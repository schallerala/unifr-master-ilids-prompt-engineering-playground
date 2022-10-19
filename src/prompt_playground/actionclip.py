import functools
import glob
import logging
from pathlib import Path
from typing import List

import numpy as np
import open_clip
import pandas as pd
import torch
from ilids.models.actionclip.factory import create_models_and_transforms

from prompt_playground import ILIDS_PATH
from prompt_playground.ilids import sequences_df
from prompt_playground.monitoring import timeit, SHARED_MONITORING_LOGGER_NAME
from prompt_playground.tensor_utils import normalize_features

_FEATURES_COLUMNS_INDEXES = pd.RangeIndex.from_range(range(512))

_VARIATION_PATHS = list(
    map(
        lambda result_file: Path(result_file),
        glob.glob(str(ILIDS_PATH / "results" / "actionclip" / "*.pkl")),
    )
)

VARIATION_NAMES = sorted(
    list(map(lambda result_path: result_path.stem, _VARIATION_PATHS))
)

_OPENAI_BASE_MODEL_NAMES = {
    "vit-b-16-16f": "ViT-B-16",
    "vit-b-16-32f": "ViT-B-16",
    "vit-b-16-8f": "ViT-B-16",
    "vit-b-32-8f": "ViT-B-32",
}


@functools.cache
def images_features_df(variation: str) -> pd.DataFrame:
    pickle_file = ILIDS_PATH / "results" / "actionclip" / f"{variation}.pkl"
    features_df = pd.read_pickle(pickle_file)
    features_df.set_index(features_df.index.str.lstrip("data/sequences/"), inplace=True)

    # Drop NaN, in case a sequence wasn't fed to the model as it didn't have enough frames
    df = sequences_df().join(features_df).dropna(subset=_FEATURES_COLUMNS_INDEXES)

    # For each sample, get the highest feature/signal
    df["Activation"] = df[_FEATURES_COLUMNS_INDEXES].max(axis=1)

    return df


# @cache_registry.cache(VARIATION_NAMES)
@functools.cache
def get_text_model(variation: str) -> torch.nn.Module:
    return create_models_and_transforms(
        actionclip_pretrained_ckpt=ILIDS_PATH
        / "ckpt"
        / "actionclip"
        / f"{variation}.pt",
        openai_model_name=_OPENAI_BASE_MODEL_NAMES[variation],
        extracted_frames=8,
        device=torch.device("cpu"),
    )[1]


# @cache_registry.cache(VARIATION_NAMES)
@functools.cache
def get_images_features(model_variation: str) -> torch.Tensor:
    features = torch.from_numpy(
        images_features_df(model_variation)[_FEATURES_COLUMNS_INDEXES].to_numpy(
            dtype=np.float32
        )
    )

    return features


@functools.cache
def get_text_features(text: str, model_variation: str) -> torch.Tensor:
    tokenized_texts = open_clip.tokenize([text])

    with torch.no_grad():
        return get_text_model(model_variation)(tokenized_texts).squeeze()


@timeit(logger_name=SHARED_MONITORING_LOGGER_NAME)
def get_all_text_features(
    texts: List[str], model_variation: str, normalized: bool = True
) -> torch.Tensor:
    # this way the results can be cached - might lose performance of vectorization but gaining by
    # caching results
    features = torch.vstack(
        tuple(get_text_features(text, model_variation) for text in texts)
    )
    if normalized:
        features = normalize_features(features)

    return features
