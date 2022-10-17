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
from typing import Dict, List, Set, Tuple, Union, Optional

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

from timeit import timeit

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


@timeit()
@functools.cache
def sequences_df() -> pd.DataFrame:
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
    ].head(400)

    return SEQUENCES_DF


@timeit()
@functools.cache
def images_features_df(variation: str) -> pd.DataFrame:
    pickle_file = ILIDS_PATH / "results" / "actionclip" / f"{variation}.pkl"
    features_df = pd.read_pickle(pickle_file)
    features_df.set_index(features_df.index.str.lstrip("data/sequences/"), inplace=True)

    # Drop NaN, in case a sequence wasn't fed to the model as it didn't have enough frames
    df = sequences_df().join(features_df).dropna(subset=FEATURES_COLUMNS_INDEXES)

    df["Alarm"] = df["Classification"] == "TP"
    # For each sample, get the highest feature/signal
    df["Activation"] = df[FEATURES_COLUMNS_INDEXES].max(axis=1)

    df["category"] = None  # "create" a new column
    df.loc[df["Distraction"].notnull(), "category"] = "Distraction"
    df.loc[~df["Distraction"].notnull(), "category"] = "Background"
    df.loc[df["Classification"] == "TP", "category"] = "Alarm"

    return df


OPENAI_BASE_MODEL_NAME = {
    "vit-b-16-16f": "ViT-B-16",
    "vit-b-16-32f": "ViT-B-16",
    "vit-b-16-8f": "ViT-B-16",
    "vit-b-32-8f": "ViT-B-32"
}


@timeit()
@functools.cache
def get_text_model(variation: str) -> torch.nn.Module:
    return create_models_and_transforms(
        actionclip_pretrained_ckpt=ILIDS_PATH
        / "ckpt"
        / "actionclip"
        / f"{variation}.pt",
        openai_model_name=OPENAI_BASE_MODEL_NAME[variation],
        extracted_frames=8,
        device=torch.device("cpu"),
    )[1]


class TextRequest(BaseModel):
    texts: List[str]
    classification: List[bool]


@functools.cache
def get_text_features(text: str, model_variation: str) -> torch.Tensor:
    tokenized_texts = open_clip.tokenize([text])

    with torch.no_grad():
        return get_text_model(model_variation)(tokenized_texts).squeeze()


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


@app.get("/")
def get_image():
    return "Hello"

@app.get("/image/{image_name}")
def get_image(image_name: str) -> Response:
    return get_cached_image_response(image_name)


@app.get("/images")
def get_all_images_and_categories() -> Dict[str, List[str]]:
    model_variation = VARIATION_NAMES[0]

    clips_features_df = images_features_df(model_variation)

    clip_indexes = {
        "index": clips_features_df.index.to_list(),
        "categories": clips_features_df["category"].to_list(),
        "distances": clips_features_df["Distance"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
        "approaches": clips_features_df["SubjectApproachType"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
        "descriptions": clips_features_df["SubjectDescription"]
        .fillna(np.nan)
        .replace([np.nan], [None])
        .to_list(),
    }

    return clip_indexes


@functools.cache
def get_images_features(model_variation: str, normalized: bool = True) -> torch.Tensor:
    features = torch.from_numpy(
        images_features_df(model_variation)[FEATURES_COLUMNS_INDEXES].to_numpy(
            dtype=np.float32
        )
    )
    if normalized:
        normalize_features(features)

    return features


def get_all_text_features(texts: List[str], model_variation: str, normalized: bool = True) -> torch.Tensor:
    # this way the results can be cached - might lose performance of vectorization but gaining by
    # caching results
    features = torch.vstack(tuple(get_text_features(text, model_variation) for text in texts))
    if normalized:
        features = normalize_features(features)

    return features


def normalize_features(features):
    features /= features.norm(dim=-1, keepdim=True)
    return features


@functools.cache
def get_image_tsne(model_variation: str, text_count: int) -> Tuple[List, List]:
    features = get_images_features(model_variation)

    sequences_projections_2d = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        # setting to "auto" won't have a result comparable
        # between clips and texts, as the input N is much larger/smaller
        learning_rate=200,
        init="pca",
        perplexity=get_tsne_perplexity(text_count),
    ).fit_transform(features)

    # fig = px.scatter(
    #     sequences_projections_2d, x=0, y=1,
    #     color=projection_color_df["category"],
    #     render_mode='svg',
    #     hover_data={"sequence": ALL_DF["vit-b-16-8f"].index},
    # )

    return (
        sequences_projections_2d.tolist(),
        images_features_df(model_variation).index.to_list(),
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
    model_variation: str


class SimilarityRequest(TextsRequest):
    text_classification_method: TopClassificationMethod
    texts_to_subtract: Optional[List[str]] = None


def _audit_confusion(request: SimilarityRequest, tp: int, fn: int, fp: int, tn: int):
    history_path = Path("history.csv")
    open_mode, add_header = ("a", False) if history_path.exists() else ("w", True)

    pd.DataFrame([dict(**request.dict(), tp=tp, fn=fn, fp=fp, tn=tn)]).to_csv(
        history_path, header=add_header, index=False, mode=open_mode
    )


@app.post("/similarity")
def get_similarity(request: SimilarityRequest) -> SimilarityResponse:
    texts = request.texts
    classifications = request.classifications
    variation = request.model_variation
    text_classification = request.text_classification_method

    assert variation in VARIATION_NAMES
    assert len(texts) == len(classifications) and len(texts) > 0

    images_features = get_images_features(variation, False)
    texts_features = get_all_text_features(texts, variation, False)

    texts_to_subtract_features_sum = (
        torch.zeros(images_features.shape[-1])
        if not request.texts_to_subtract
        else get_all_text_features(request.texts_to_subtract, variation, False).sum(dim=0)
    )

    images_features -= texts_to_subtract_features_sum
    texts_features -= texts_to_subtract_features_sum

    images_features = normalize_features(images_features)
    texts_features = normalize_features(texts_features)

    similarities = 100.0 * images_features @ texts_features.T
    softmax_similarities = similarities.softmax(dim=-1)

    clips_features_df = images_features_df(variation)

    texts_len = len(texts)
    clips_len = len(clips_features_df)

    clips_classification = clips_features_df["Classification"] == "TP"

    # ClipIndex, ClipClassification, Text, TextClassification, TextSoftmax, TextSoftmaxRank
    similarity_df = (
        clips_features_df.index.repeat(texts_len).to_frame(name="clip")
    )
    similarity_df["ClipClassification"] = clips_classification.repeat(texts_len)
    similarity_df["Text"] = np.tile(texts, clips_len)
    similarity_df["TextClassification"] = np.tile(classifications, clips_len)
    similarity_df["TextSoftmax"] = softmax_similarities.numpy().ravel()
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

        _audit_confusion(request, tp=tp, fn=fn, fp=fp, tn=tn)

        return ConfusionTopK(
            tp=tp,
            fn=fn,
            fp=fp,
            tn=tn,
            # Get text classification by clip using the given method
            topk_text_classification=topk_text_classification.to_dict(),
        )

    topk_confusion = {
        topk: _get_topk_classification_and_confusion_matrix(topk) for topk in topks
    }

    return SimilarityResponse(similarities=clips_similarity, confusion=topk_confusion)


@timeit()
@functools.cache
def get_text_tsne(texts: Tuple[str, ...], model_variation: str) -> List[List[float]]:
    assert len(texts) >= 5

    text_features = get_all_text_features(list(texts), model_variation)

    texts_projections_2d = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        # setting to "auto" won't have a result comparable
        # between clips and texts, as the input N is much larger/smaller
        learning_rate=200,
        init="pca",
        perplexity=get_tsne_perplexity(len(texts)),
    ).fit_transform(text_features)

    return texts_projections_2d.tolist()


def get_tsne_perplexity(text_count: int):
    return min(30.0, math.floor(text_count - 1))


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

    assert len(texts) == len(classifications) and len(texts) > 0 and request.model_variation in VARIATION_NAMES

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
