import asyncio
import functools
import glob
import io
import math
import os
import subprocess
from email.utils import formatdate
from itertools import groupby
from math import trunc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import open_clip
import pandas as pd
import torch
from decord import VideoReader, cpu
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ilids.models.actionclip.factory import create_models_and_transforms
from PIL import Image
from pydantic import BaseModel
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from starlette._compat import md5_hexdigest

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

RANDOM_STATE = 16896375

ILIDS_PATH = Path(os.path.dirname(os.getcwd())) / "ilids"

FEATURES_COLUMNS_INDEXES = pd.RangeIndex.from_range(range(512))

VARIATION_PATHS = list(
    map(
        lambda result_file: Path(result_file),
        glob.glob(str(ILIDS_PATH / "results" / "actionclip" / "*.pkl")),
    )
)
VARIATION_NAMES = sorted(
    list(map(lambda result_path: result_path.stem, VARIATION_PATHS))
)

tp_fp_sequences_path = (
    ILIDS_PATH / "data" / "handcrafted-metadata" / "tp_fp_sequences.csv"
)
SEQUENCES_DF = pd.read_csv(tp_fp_sequences_path, index_col=0)
# Only keep relevant columns
SEQUENCES_DF = SEQUENCES_DF[
    [
        "Classification",
        "Duration",
        "Distance",
        "SubjectApproachType",
        "SubjectDescription",
        "Distraction",
        "Stage",
    ]
]


def load_variation_image_features_df(movinet_variation: str):
    pickle_file = ILIDS_PATH / "results" / "actionclip" / f"{movinet_variation}.pkl"
    features_df = pd.read_pickle(pickle_file)
    features_df.set_index(features_df.index.str.lstrip("data/sequences/"), inplace=True)

    # TODO norm

    # Drop NaN, in case a sequence wasn't fed to the model as it didn't have enough frames
    df = SEQUENCES_DF.join(features_df).dropna(subset=FEATURES_COLUMNS_INDEXES)

    df["Alarm"] = df["Classification"] == "TP"
    # For each sample, get the highest feature/signal
    df["Activation"] = df[FEATURES_COLUMNS_INDEXES].max(axis=1)

    df["category"] = None  # "create" a new column
    df.loc[df["Distraction"].notnull(), "category"] = "Distraction"
    df.loc[~df["Distraction"].notnull(), "category"] = "Background"
    df.loc[df["Classification"] == "TP", "category"] = "Alarm"

    return df


ALL_IMAGES_FEATURES_DF = {
    variation_name: load_variation_image_features_df(variation_name)
    for variation_name in VARIATION_NAMES
}

model_text = create_models_and_transforms(
    actionclip_pretrained_ckpt=ILIDS_PATH
    / "ckpt"
    / "actionclip"
    / f"{VARIATION_NAMES[0]}.pt",
    openai_model_name="ViT-B-16",
    extracted_frames=8,
    device=torch.device("cpu"),
)[1]


@functools.cache
def get_text_features(text: str) -> np.ndarray:
    tokenized_text = open_clip.tokenize([text])

    with torch.no_grad():
        return model_text(tokenized_text).numpy().ravel()


TEXT_FEATURES_LEN = len(get_text_features(""))

TEXT_DF_COLUMNS = ["classification"] + list(range(TEXT_FEATURES_LEN))

text_features_df = pd.DataFrame([], columns=TEXT_DF_COLUMNS)


@functools.cache
def update_texts_dataframe(new_text: str, classification: bool) -> np.ndarray:
    features = get_text_features(new_text)
    new_entry = pd.Series([classification, *features], index=TEXT_DF_COLUMNS)

    text_features_df.loc[new_text] = new_entry

    return features


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


@app.get("/image/{image_name}")
def get_image(image_name: str) -> Response:
    return get_cached_image_response(image_name)


@app.get("/images")
def get_all_images_and_categories() -> Dict[str, List[str]]:
    model_variation = VARIATION_NAMES[0]

    clip_indexes = {
        "index": ALL_IMAGES_FEATURES_DF[model_variation]
        .index
        .to_list(),
        "categories": ALL_IMAGES_FEATURES_DF[model_variation]["category"].to_list(),
        "distances": ALL_IMAGES_FEATURES_DF[model_variation]["Distance"].fillna(np.nan).replace([np.nan], [None]).to_list(),
        "approaches": ALL_IMAGES_FEATURES_DF[model_variation]["SubjectApproachType"].fillna(np.nan).replace([np.nan], [None]).to_list(),
        "descriptions": ALL_IMAGES_FEATURES_DF[model_variation]["SubjectDescription"].fillna(np.nan).replace([np.nan], [None]).to_list()
    }

    return clip_indexes


@functools.cache
def get_images_features(model_variation: str) -> torch.Tensor:
    features = torch.from_numpy(
        ALL_IMAGES_FEATURES_DF[model_variation][FEATURES_COLUMNS_INDEXES].to_numpy(
            dtype=np.float64
        )
    )
    features /= features.norm(dim=-1, keepdim=True)

    return features


def get_all_text_features() -> torch.Tensor:
    features = torch.from_numpy(
        text_features_df[FEATURES_COLUMNS_INDEXES].to_numpy(dtype=np.float64)
    )
    features /= features.norm(dim=-1, keepdim=True)

    return features


@functools.cache
def get_image_tsne(model_variation: str = VARIATION_NAMES[0]) -> Tuple[List, List]:
    features = get_images_features(model_variation)

    sequences_projections_2d = TSNE(
        n_components=2, random_state=RANDOM_STATE, init="pca"
    ).fit_transform(features)

    # fig = px.scatter(
    #     sequences_projections_2d, x=0, y=1,
    #     color=projection_color_df["category"],
    #     render_mode='svg',
    #     hover_data={"sequence": ALL_DF["vit-b-16-8f"].index},
    # )

    return (
        sequences_projections_2d.tolist(),
        ALL_IMAGES_FEATURES_DF[model_variation]
        .index
        .to_list(),
    )


class ClipSimilarity(BaseModel):
    text: str
    classification: bool  # text classification "as reminder"
    similarity: float


class ConfusionTopK(BaseModel):
    tp: int
    fn: int
    fp: int
    tn: int
    topk_text_classification: Dict[str, bool]  # by clip file name (key), give the text classification


class SimilarityResponse(BaseModel):
    similarities: Dict[str, List[ClipSimilarity]]  # key: str, being the clip file name
    confusion: Dict[int, ConfusionTopK]


@app.get("/similarity")
def get_similarity() -> SimilarityResponse:
    # TODO
    model_variation = VARIATION_NAMES[0]
    images_features = get_images_features(model_variation)
    texts_features = get_all_text_features()

    similarities = 100.0 * images_features @ texts_features.T
    softmax_similarities: List[List[float]] = similarities.softmax(dim=-1).tolist()

    clips = (
        ALL_IMAGES_FEATURES_DF[model_variation]
        .index
        .tolist()
    )

    texts_len = len(text_features_df)
    clips_len = len(ALL_IMAGES_FEATURES_DF[model_variation])

    clips_classification = ALL_IMAGES_FEATURES_DF[model_variation]["Classification"] == "TP"

    # ClipIndex, ClipClassification, TextClassification, TextSoftmax
    similarity_df = ALL_IMAGES_FEATURES_DF[model_variation].index.repeat(texts_len).to_frame(name="clip")
    similarity_df["ClipClassification"] = clips_classification.repeat(texts_len)
    similarity_df["TextClassification"] = np.tile(text_features_df["classification"], clips_len)
    similarity_df["TextSoftmax"] = np.array(softmax_similarities).ravel()
    similarity_df.reset_index(drop=True, inplace=True)
    similarity_df["TextSoftmaxRank"] = similarity_df.groupby("clip")["TextSoftmax"].rank(method="first", ascending=False)

    texts = list(
        zip(
            text_features_df.index.to_list(),
            text_features_df["classification"].to_list(),
        )
    )

    clips_similarity = {
        clip: [
            ClipSimilarity(
                text=text,
                classification=classification,
                similarity=clip_softmax_similarity,
            )
            for (text, classification), clip_softmax_similarity in zip(
                texts, clip_softmax_similarities
            )
        ]
        for clip, clip_softmax_similarities in zip(clips, softmax_similarities)
    }

    topks = [k for k in [1, 3, 5] if k <= texts_len]

    def _get_topk_classification_and_confusion_matrix(topk: int) -> ConfusionTopK:
        topk_text_classification = similarity_df[similarity_df["TextSoftmaxRank"] <= topk].groupby("clip")["TextClassification"].agg(
            pd.Series.mode)

        tn, fp, fn, tp = confusion_matrix(clips_classification, topk_text_classification).ravel()

        return ConfusionTopK(
            tp=tp,
            fn=fn,
            fp=fp,
            tn=tn,
            # Count most frequent occurrence of text classification by clip
            topk_text_classification=topk_text_classification.to_dict()
        )

    topk_confusion = {
        topk: _get_topk_classification_and_confusion_matrix(topk)
        for topk in topks
    }

    return SimilarityResponse(similarities=clips_similarity, confusion=topk_confusion)


def get_text_tsne() -> Tuple[List, List, List]:
    assert len(text_features_df) >= 5
    text_features = torch.from_numpy(
        text_features_df[FEATURES_COLUMNS_INDEXES].to_numpy(dtype=np.float64)
    )
    text_features /= text_features.norm(dim=-1, keepdim=True)

    texts_projections_2d = TSNE(
        n_components=2,
        random_state=16896375,
        perplexity=min(30.0, math.floor(len(text_features) - 1)),
    ).fit_transform(text_features)

    return (
        texts_projections_2d.tolist(),
        text_features_df.index.to_list(),
        text_features_df["classification"].to_list(),
    )


@app.get("/tsne-images/{model_variation}")
def get_tsne_images_features(model_variation: str):
    tsne_result, index = get_image_tsne(model_variation)

    groups = {
        k: list(map(lambda i: i[1:], g))
        for k, g in groupby(
            sorted(
                zip(
                    ALL_IMAGES_FEATURES_DF[model_variation]["category"],
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


@app.get("/tsne-images")
def get_default_tsne_images_features():
    model_variation = VARIATION_NAMES[0]

    return get_tsne_images_features(model_variation)


@app.get("/tsne-texts")
def get_default_tsne_images_features():
    tsne_result, texts, classifications = get_text_tsne()

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


@app.get("/text")
def get_all_texts() -> Dict[str, List[Union[str, bool]]]:
    return {
        "text": text_features_df.index.to_list(),
        "classification": text_features_df["classification"].to_list(),
    }


class AddTextRequest(BaseModel):
    text: str
    classification: bool


class UpdateTextRequest(BaseModel):
    text: str
    classification: bool


class AddAllTextRequest(BaseModel):
    texts: List[str]
    classifications: List[bool]


class RemoveTextRequest(BaseModel):
    text: str


@app.post("/text/add")
def add_new_text(request: AddTextRequest):
    update_texts_dataframe(request.text, request.classification)


@app.put("/text")
def add_new_text(request: UpdateTextRequest):
    text_features_df.loc[request.text, "classification"] = request.classification


@app.post("/text/add-all")
def add_all_new_text(request: AddAllTextRequest):
    assert len(request.texts) == len(request.classifications)
    for text, classification in zip(request.texts, request.classifications):
        update_texts_dataframe(text, classification)


@app.delete("/text")
def delete_text(request: RemoveTextRequest):
    text_features_df.drop(index=request.text, inplace=True)


@app.post("/play/{video_filename}")
def play_video(video_filename: str):
    video_path = ILIDS_PATH / "data" / "sequences" / video_filename
    assert video_path.exists() and video_path.is_file()

    returncode = subprocess.run(["open", video_path]).returncode

    assert returncode == 0
