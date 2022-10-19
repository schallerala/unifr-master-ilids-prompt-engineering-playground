import logging
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix

from prompt_playground.actionclip import (
    VARIATION_NAMES,
    images_features_df,
    get_all_text_features,
    get_images_features,
)
from prompt_playground.monitoring import timeit, SHARED_MONITORING_LOGGER_NAME
from prompt_playground.tensor_utils import normalize_features

HISTORY_PATH = Path(os.getcwd()) / "history.csv"


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
    max: float  # max similarity
    min: float  # min similarity
    confusion: Dict[int, ConfusionTopK]


class TopClassificationMethod(str, Enum):
    mode = "mode"  # text classification with the most frequent class in top k
    max_sum = "max_sum"  # sum all similarity by text class and pick biggest
    any = "any"  # any text classification in top k is defined as alarm


class SimilarityParams(BaseModel):
    texts: List[str]
    classifications: List[bool]
    model_variation: str
    text_classification_method: TopClassificationMethod
    texts_to_subtract: Optional[List[str]] = None
    apply_softmax: bool = True


@timeit(logger_name=SHARED_MONITORING_LOGGER_NAME)
def _get_similarities(
    variation: str,
    texts: List[str],
    apply_softmax: bool,
    texts_to_subtract: Optional[List[str]] = None,
):
    images_features = get_images_features(variation)
    texts_features = get_all_text_features(texts, variation, False)

    texts_to_subtract_features_sum = (
        torch.zeros(images_features.shape[-1])
        if not texts_to_subtract
        else get_all_text_features(texts_to_subtract, variation, False).sum(dim=0)
    )
    images_features -= texts_to_subtract_features_sum
    texts_features -= texts_to_subtract_features_sum

    images_features = normalize_features(images_features)
    texts_features = normalize_features(texts_features)

    similarities = images_features @ texts_features.T
    if apply_softmax:
        similarities = (100.0 * similarities).softmax(dim=-1)

    return similarities


@timeit(logger_name=SHARED_MONITORING_LOGGER_NAME)
def _get_clips_similarity_dict(similarity_df):
    clips_similarity = defaultdict(list)

    # https://stackoverflow.com/a/54383480/3771148
    # iterating over numpy values is faster than to_dict or iterrows
    values = similarity_df.sort_values("TextSimilarity", ascending=False)[
        ["clip", "Text", "TextClassification", "TextSimilarity"]
    ].values

    for row in values:
        clips_similarity[row[0]].append(
            ClipSimilarity.construct(
                text=row[1],
                classification=row[2],
                similarity=row[3],
            )
        )

    return clips_similarity


@timeit(logger_name=SHARED_MONITORING_LOGGER_NAME)
def _get_topk_classification_and_confusion_matrix(
    text_classification: TopClassificationMethod,
    clips_index: pd.Index,
    y_true: pd.Series,
    similarity_df: pd.DataFrame,
) -> ConfusionTopK:
    if text_classification == TopClassificationMethod.mode:
        topk_text_classification = similarity_df.groupby("clip")[
            "TextClassification"
        ].agg(pd.Series.mode)
    elif text_classification == TopClassificationMethod.any:
        topk_text_classification = similarity_df.groupby("clip")[
            "TextClassification"
        ].any()
    elif text_classification == TopClassificationMethod.max_sum:
        sum_similarities = similarity_df.groupby(["clip", "TextClassification"])[
            "TextSimilarity"
        ].sum()
        topk_text_classification = (
            sum_similarities.to_frame()
            .sort_values("TextSimilarity", ascending=False)
            .reset_index()
            .drop_duplicates(subset=["clip"], keep="first")
            .set_index("clip", drop=True)["TextClassification"]
            # to order it back to normal
            .loc[clips_index]
        )
    else:
        raise ValueError(text_classification)

    tn, fp, fn, tp = confusion_matrix(y_true, topk_text_classification).ravel()

    return ConfusionTopK(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        # Get text classification by clip using the given method
        topk_text_classification=topk_text_classification.to_dict(),
    )


def _audit_confusion(
    request: SimilarityParams, topk: int, tp: int, fn: int, fp: int, tn: int
):
    open_mode, add_header = ("a", False) if HISTORY_PATH.exists() else ("w", True)

    pd.DataFrame(
        [dict(topk=topk, **request.dict(), tp=tp, fn=fn, fp=fp, tn=tn)]
    ).to_csv(HISTORY_PATH, header=add_header, index=False, mode=open_mode)


def clips_texts_similarities(params: SimilarityParams) -> SimilarityResponse:
    texts = params.texts
    classifications = params.classifications
    variation = params.model_variation
    text_classification = params.text_classification_method

    assert variation in VARIATION_NAMES
    assert len(texts) == len(classifications) and len(texts) > 0

    similarities = _get_similarities(
        variation, texts, params.apply_softmax, params.texts_to_subtract
    )

    clips_features_df = images_features_df(variation)

    texts_len = len(texts)
    clips_len = len(clips_features_df)

    # ClipIndex, ClipClassification, Text, TextClassification, TextSimilarity, TextSimilarityRank
    similarity_df = clips_features_df.index.repeat(texts_len).to_frame(name="clip")
    similarity_df["ClipClassification"] = clips_features_df["Alarm"].repeat(texts_len)
    similarity_df["Text"] = np.tile(texts, clips_len)
    similarity_df["TextClassification"] = np.tile(classifications, clips_len)
    similarity_df["TextSimilarity"] = similarities.numpy().ravel()
    similarity_df.reset_index(drop=True, inplace=True)

    min_similarity = float(similarities.min())
    max_similarity = float(similarities.max())

    clips_similarity = _get_clips_similarity_dict(similarity_df)

    topks = [k for k in [1, 3, 5] if k <= texts_len]

    similarity_df["TextSimilarityRank"] = similarity_df.groupby("clip")[
        "TextSimilarity"
    ].rank(method="first", ascending=False)

    topk_confusion: Dict[int, ConfusionTopK] = dict()

    # go in reversed order, to always reduce the dataframe size and "speed up" a bit the selection
    # for the top K
    for topk in reversed(topks):
        similarity_df = similarity_df[similarity_df["TextSimilarityRank"] <= topk]
        confusion = _get_topk_classification_and_confusion_matrix(
            text_classification,
            clips_features_df.index,
            clips_features_df["Alarm"],
            similarity_df,
        )

        topk_confusion[topk] = confusion

        _audit_confusion(
            params,
            topk=topk,
            tp=confusion.tp,
            fn=confusion.fn,
            fp=confusion.fp,
            tn=confusion.tn,
        )

    return SimilarityResponse.construct(
        similarities=clips_similarity,
        max=max_similarity,
        min=min_similarity,
        confusion=topk_confusion,
    )
