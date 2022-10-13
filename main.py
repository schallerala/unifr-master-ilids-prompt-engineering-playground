import functools
import glob
import io
import math
import os
import subprocess
from email.utils import formatdate
from enum import auto, Enum
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import open_clip
import pandas as pd
import torch
from decord import VideoReader, cpu
from fastapi import FastAPI, Response, Query
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


class TextRequest(BaseModel):
    texts: List[str]
    classification: List[bool]


@functools.cache
def get_text_features_cached(texts: Tuple[str, ...]) -> torch.Tensor:
    tokenized_texts = open_clip.tokenize(list(texts))

    with torch.no_grad():
        return model_text(tokenized_texts)


def get_text_features(texts: List[str]) -> torch.Tensor:
    return get_text_features_cached(tuple(texts))


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
        "index": ALL_IMAGES_FEATURES_DF[model_variation].index.to_list(),
        "categories": ALL_IMAGES_FEATURES_DF[model_variation]["category"].to_list(),
        "distances": ALL_IMAGES_FEATURES_DF[model_variation]["Distance"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
        "approaches": ALL_IMAGES_FEATURES_DF[model_variation]["SubjectApproachType"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
        "descriptions": ALL_IMAGES_FEATURES_DF[model_variation]["SubjectDescription"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
    }

    return clip_indexes


@functools.cache
def get_images_features(model_variation: str, normalized: bool = True) -> torch.Tensor:
    features = torch.from_numpy(
        ALL_IMAGES_FEATURES_DF[model_variation][FEATURES_COLUMNS_INDEXES].to_numpy(
            dtype=np.float32
        )
    )
    if normalized:
        features /= features.norm(dim=-1, keepdim=True)

    return features


def get_all_text_features(texts: List[str], normalized: bool = True) -> torch.Tensor:
    features = get_text_features(texts)
    if normalized:
        features /= features.norm(dim=-1, keepdim=True)

    return features


@functools.cache
def get_image_tsne(model_variation: str) -> Tuple[List, List]:
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
        ALL_IMAGES_FEATURES_DF[model_variation].index.to_list(),
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
    topk_text_classification: Dict[
        str, bool
    ]  # by clip file name (key), give the text classification


class SimilarityResponse(BaseModel):
    similarities: Dict[str, List[ClipSimilarity]]  # key: str, being the clip file name
    confusion: Dict[int, ConfusionTopK]


class TopClassificationMethod(str, Enum):
    mode = "mode"  # text classification with the most frequent class in top k
    max_sum = "max_sum"  # sum all similarity by text class and pick biggest
    any = "any"  # any text classification in top k is defined as alarm


@app.get("/text-classification")
def get_similarity_text_classification() -> List[str]:
    return [e.value for e in TopClassificationMethod]


class TextsRequest(BaseModel):
    texts: List[str]
    classifications: List[bool]


class SimilarityRequest(TextsRequest):
    model_variation: str
    text_classification_method: TopClassificationMethod


@app.post("/similarity")
def get_similarity(request: SimilarityRequest) -> SimilarityResponse:
    texts = request.texts
    classifications = request.classifications
    variation = request.model_variation
    text_classification = request.text_classification_method

    assert variation in ALL_IMAGES_FEATURES_DF
    assert len(texts) == len(classifications) and len(texts) > 0

    images_features = get_images_features(variation)
    texts_features = get_all_text_features(texts)

    similarities = 100.0 * images_features @ texts_features.T
    softmax_similarities: List[List[float]] = similarities.softmax(dim=-1).tolist()

    clips = ALL_IMAGES_FEATURES_DF[variation].index.tolist()

    texts_len = len(texts)
    clips_len = len(ALL_IMAGES_FEATURES_DF[variation])

    clips_classification = ALL_IMAGES_FEATURES_DF[variation]["Classification"] == "TP"

    # ClipIndex, ClipClassification, TextClassification, TextSoftmax
    similarity_df = (
        ALL_IMAGES_FEATURES_DF[variation].index.repeat(texts_len).to_frame(name="clip")
    )
    similarity_df["ClipClassification"] = clips_classification.repeat(texts_len)
    similarity_df["Text"] = np.tile(texts, clips_len)
    similarity_df["TextClassification"] = np.tile(classifications, clips_len)
    similarity_df["TextSoftmax"] = np.array(softmax_similarities).ravel()
    similarity_df.reset_index(drop=True, inplace=True)
    similarity_df["TextSoftmaxRank"] = similarity_df.groupby("clip")[
        "TextSoftmax"
    ].rank(method="first", ascending=False)

    clips_similarity = {
        clip: [
            ClipSimilarity(
                text=row["Text"],
                classification=row["TextClassification"],
                similarity=row["TextSoftmax"],
            )
            for i, row in rows_df.iterrows()
        ]
        for clip, rows_df in similarity_df[
            ["clip", "Text", "TextClassification", "TextSoftmax"]
        ]
        .sort_values("TextSoftmax", ascending=False)
        .groupby("clip")
    }

    topks = [k for k in [1, 3, 5] if k <= texts_len]

    def _get_topk_classification_and_confusion_matrix(topk: int) -> ConfusionTopK:
        if text_classification == TopClassificationMethod.mode:
            topk_text_classification = (
                similarity_df[similarity_df["TextSoftmaxRank"] <= topk]
                .groupby("clip")["TextClassification"]
                .agg(pd.Series.mode)
            )
        elif text_classification == TopClassificationMethod.any:
            topk_text_classification = (
                similarity_df[similarity_df["TextSoftmaxRank"] <= topk]
                .groupby("clip")["TextClassification"]
                .any()
            )
        elif text_classification == TopClassificationMethod.max_sum:
            sum_similarities = (
                similarity_df[similarity_df["TextSoftmaxRank"] <= topk]
                .groupby(["clip", "TextClassification"])["TextSoftmax"]
                .sum()
            )
            topk_text_classification = (
                sum_similarities.to_frame()
                .sort_values("TextSoftmax", ascending=False)
                .reset_index()
                .drop_duplicates(subset=["clip"], keep="first")
                .set_index("clip", drop=True)["TextClassification"]
                .loc[clips_classification.index]
            )
        else:
            raise ValueError(text_classification)

        tn, fp, fn, tp = confusion_matrix(
            clips_classification, topk_text_classification
        ).ravel()

        return ConfusionTopK(
            tp=tp,
            fn=fn,
            fp=fp,
            tn=tn,
            # Count most frequent occurrence of text classification by clip
            topk_text_classification=topk_text_classification.to_dict(),
        )

    topk_confusion = {
        topk: _get_topk_classification_and_confusion_matrix(topk) for topk in topks
    }

    return SimilarityResponse(similarities=clips_similarity, confusion=topk_confusion)


def get_text_tsne(texts: List[str]) -> List[List[float]]:
    assert len(texts) >= 5

    text_features = get_text_features(texts)

    texts_projections_2d = TSNE(
        n_components=2,
        random_state=16896375,
        perplexity=min(30.0, math.floor(len(texts) - 1)),
    ).fit_transform(text_features)

    return texts_projections_2d.tolist()


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


@app.post("/tsne-texts")
def get_default_tsne_images_features(request: TextsRequest):
    texts = request.texts
    classifications = request.classifications

    assert len(texts) == len(classifications) and len(texts) > 0

    tsne_result = get_text_tsne(texts)

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


class UpdateTextRequest(BaseModel):
    text: str
    classification: bool


class AddAllTextRequest(BaseModel):
    texts: List[str]
    classifications: List[bool]


class RemoveTextRequest(BaseModel):
    text: str


@app.get("/variations")
def get_variations() -> List[str]:
    return VARIATION_NAMES


@app.post("/play/{video_filename}")
def play_video(video_filename: str):
    video_path = ILIDS_PATH / "data" / "sequences" / video_filename
    assert video_path.exists() and video_path.is_file()

    returncode = subprocess.run(["open", video_path]).returncode

    assert returncode == 0
