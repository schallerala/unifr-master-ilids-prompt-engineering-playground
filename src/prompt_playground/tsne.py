import functools
import math
from typing import Tuple, List

from sklearn.manifold import TSNE

import torch

from prompt_playground import RANDOM_STATE
from prompt_playground.actionclip import (
    get_images_features,
    images_features_df,
    get_all_text_features,
)
from prompt_playground.tensor_utils import normalize_features


@functools.cache
def get_fit_transform_embeddings(model_variation: str, texts: Tuple[str, ...]) -> Tuple[Tuple[List, List], List[List[float]]]:
    """As the t-SNE has an internal state and should be fit and transform only once,
    concat the visual and text embeddings and return the fit_transform results as tuples.
    """
    images_features = normalize_features(get_images_features(model_variation))
    text_features = get_all_text_features(list(texts), model_variation)

    features = torch.vstack([images_features, text_features])

    projections_2d = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        # setting to "auto" won't have a result comparable
        # between clips and texts, as the input N is much larger/smaller
        learning_rate=200,
        init="pca",
    ).fit_transform(features)

    return (
        (projections_2d[: len(images_features)].tolist(), images_features_df(model_variation).index.to_list()),
        projections_2d[len(images_features):].tolist(),
    )



def get_image_tsne(model_variation: str, texts: Tuple[str, ...]) -> Tuple[List, List]:
    projections, clips = get_fit_transform_embeddings(model_variation, texts)[0]

    return projections, clips


def get_text_tsne(model_variation: str, texts: Tuple[str, ...]) -> List[List[float]]:
    projections = get_fit_transform_embeddings(model_variation, texts)[1]

    return projections
