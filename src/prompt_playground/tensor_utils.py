def normalize_features(features):
    features /= features.norm(dim=-1, keepdim=True)

    return features
