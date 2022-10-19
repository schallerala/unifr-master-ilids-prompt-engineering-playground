import asyncio
import functools
import io
import logging
import math
import os
import subprocess
from email.utils import formatdate
from itertools import groupby
from logging import getLogger
from typing import Dict, List

import numpy as np
from decord import VideoReader, cpu
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from starlette._compat import md5_hexdigest

from prompt_playground import ILIDS_PATH
from prompt_playground.actionclip import (
    VARIATION_NAMES,
    images_features_df,
    get_text_model,
    get_images_features,
)
from prompt_playground.actionclip_similarities import (
    TopClassificationMethod,
    SimilarityParams,
    clips_texts_similarities,
)
from prompt_playground.ilids import sequences_df
from prompt_playground.monitoring import SHARED_MONITORING_LOGGER_NAME
from prompt_playground.precache_registry import PrecacheRegistry
from prompt_playground.tsne import get_text_tsne, get_image_tsne

logger = getLogger(__name__)

getLogger(SHARED_MONITORING_LOGGER_NAME).setLevel(logging.DEBUG)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cache_registry = PrecacheRegistry()
# prepare to cache a few functions that will speed up the first queries instead of have an even
# slower first request
cache_registry.register(sequences_df)
cache_registry.register(images_features_df, VARIATION_NAMES)
cache_registry.register(get_text_model, VARIATION_NAMES)
cache_registry.register(get_images_features, VARIATION_NAMES)

cache_registry.logger.setLevel(logging.INFO)


@app.get("/variations")
def get_variations() -> List[str]:
    return VARIATION_NAMES


class TextsRequest(BaseModel):
    texts: List[str]
    classifications: List[bool]
    model_variation: str


@functools.cache
def get_image_headers(image_name: str):
    stat_result = os.stat(ILIDS_PATH / "data" / "sequences" / image_name)

    content_length = str(stat_result.st_size)
    last_modified = formatdate(stat_result.st_mtime, usegmt=True)
    etag_base = str(stat_result.st_mtime) + "-" + str(stat_result.st_size)
    etag = md5_hexdigest(etag_base.encode(), usedforsecurity=False)

    return {
        # "content-length": content_length,
        "last-modified": last_modified,
        "etag": etag,
    }


@functools.cache
def get_image_array(name: str) -> np.ndarray:
    vr = VideoReader(str(ILIDS_PATH / "data" / "sequences" / name), cpu(0))
    center_frame_idx = math.floor(len(vr) / 2)

    return vr[center_frame_idx].asnumpy()  # W, H, C


@functools.cache
def get_cached_image_response(image_name: str) -> Response:
    im = Image.fromarray(get_image_array(image_name))

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    headers = {
        "Content-Disposition": f'inline; filename="{image_name}"',
        **get_image_headers(image_name),
    }
    return Response(im_bytes, headers=headers, media_type="image/png")


@app.get("/text-classification")
def get_similarity_text_classification() -> List[str]:
    return [e.value for e in TopClassificationMethod]


@app.post("/similarity")
def get_similarities(params: SimilarityParams):
    return clips_texts_similarities(params)


@app.get("/tsne-images/{model_variation}")
def get_tsne_images_features(model_variation: str, text_count: int = 30):
    tsne_result, index = get_image_tsne(model_variation, text_count)

    groups = {
        k: list(map(lambda i: i[1:], g))
        for k, g in groupby(
            sorted(
                zip(
                    images_features_df(model_variation)["category"],
                    index,
                    tsne_result,
                )
            ),
            lambda e: e[0],
        )
    }
    groups = {
        k: {
            "text": list(map(lambda i: i[0], clips)),
            "x": list(map(lambda i: i[1][0], clips)),
            "y": list(map(lambda i: i[1][1], clips)),
        }
        for k, clips in groups.items()
    }

    return groups


@app.post("/tsne-texts")
def get_default_tsne_images_features(request: TextsRequest):
    texts = request.texts
    classifications = request.classifications

    assert (
        len(texts) == len(classifications)
        and len(texts) > 0
        and request.model_variation in VARIATION_NAMES
    )

    tsne_result = get_text_tsne(tuple(texts), request.model_variation)

    # group results for easier plotting on the frontend with multiple traces
    groups = {
        k: list(map(lambda i: i[1:], g))
        for k, g in groupby(
            sorted(zip(classifications, texts, tsne_result)), lambda e: e[0]
        )
    }
    groups = {
        k: {
            "text": list(map(lambda i: i[0], texts)),
            "x": list(map(lambda i: i[1][0], texts)),
            "y": list(map(lambda i: i[1][1], texts)),
        }
        for k, texts in groups.items()
    }

    return groups


@app.post("/play/{video_filename}")
def play_video(video_filename: str):
    video_path = ILIDS_PATH / "data" / "sequences" / video_filename
    assert video_path.exists() and video_path.is_file()

    returncode = subprocess.run(["open", video_path]).returncode

    assert returncode == 0


@app.get("/image/{image_name}")
def get_image(image_name: str) -> Response:
    return get_cached_image_response(image_name)


@app.get("/images")
def get_all_images_and_categories() -> Dict[str, List[str]]:
    df = sequences_df()

    clip_indexes = {
        "index": df.index.to_list(),
        "categories": df["category"].to_list(),
        "distances": df["Distance"].fillna(np.nan).replace([np.nan], [None]).to_list(),
        "approaches": df["SubjectApproachType"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
        "descriptions": df["SubjectDescription"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
    }

    return clip_indexes


@app.on_event("startup")
def cache_io_blocking_methods():
    asyncio.create_task(cache_registry.call_all_async())
