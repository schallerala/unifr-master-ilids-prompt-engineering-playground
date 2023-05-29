import functools
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from prompt_playground.actionclip import get_all_text_features, get_images_features, images_features_df

from prompt_playground.tensor_utils import normalize_features


@functools.cache
def roc_auc(model_variation: str, texts: Tuple[str, ...], text_class: Tuple[bool, ...]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    # 1. similarity matrix
    # 2. score -> for each clip, get the average similarity score
    # 3. roc curve
    # 4. auc

    # 0. order the texts by text_class (False are first, True are last)
    ordered_texts_by_class = pd.Series(text_class, index=texts).sort_values()
    count_of_false_texts = ordered_texts_by_class[ordered_texts_by_class == False].count()

    # 1. similarity matrix
    normalized_visual_features = normalize_features(get_images_features(model_variation))
    normalized_text_features = get_all_text_features(ordered_texts_by_class.index.to_list(), model_variation)

    # 512 x 512, text x visual
    similarities = normalized_text_features @ normalized_visual_features.T

    # 2. score
    # create the score (negative are above the positive)
    #   - foreach column, cut after the number of positive texts
    #       - mean the values (get a value for the positive and the negative texts)
    #       - divide both values
    negative_texts_similarity_score_mean = similarities[:count_of_false_texts].mean(axis=0)
    positive_texts_similarity_score_mean = similarities[count_of_false_texts:].mean(axis=0)

    y_true = images_features_df(model_variation)["y_true"].values
    ratio_score = positive_texts_similarity_score_mean / negative_texts_similarity_score_mean

    # compute the ROC and AUC score
    fpr, tpr, thresholds = roc_curve(y_true, ratio_score)
    roc_auc = auc(fpr, tpr)

    return (fpr, tpr, thresholds), roc_auc
