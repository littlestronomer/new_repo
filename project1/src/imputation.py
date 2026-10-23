from sklearn.impute import KNNImputer, SimpleImputer


def get_imputer(method="median", **params):
    if method == "median":
        return SimpleImputer(strategy="median", **params)
    elif method == "mean":
        return SimpleImputer(strategy="mean", **params)
    elif method == "knn":
        return KNNImputer(**params)
    else:
        raise ValueError(f"Unknown imputer: {method}")
