from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer


def get_imputer(method="median", **params):
    if method in ("median", "mean", "most_frequent"):
        return SimpleImputer(strategy=method, **params)
    elif method == "knn":
        return KNNImputer(**params)
    elif method == "iterative":
        params.setdefault("random_state", 8004)
        return IterativeImputer(**params)
    elif method == "passthrough":
        return FunctionTransformer()
    else:
        raise ValueError(f"Unknown imputer: {method}")