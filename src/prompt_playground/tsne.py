import functools
import math
from typing import Tuple, List

from sklearn.manifold import TSNE

from prompt_playground import RANDOM_STATE
from prompt_playground.actionclip import (
    get_images_features,
    images_features_df,
    get_all_text_features,
)
from prompt_playground.tensor_utils import normalize_features


def get_tsne_perplexity(text_count: int):
    return min(30.0, math.floor(text_count - 1))


@functools.cache
def get_image_tsne(model_variation: str, text_count: int) -> Tuple[List, List]:
    features = normalize_features(get_images_features(model_variation))

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
