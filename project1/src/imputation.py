from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor


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
    elif method == "iterative_forest":
        params.setdefault("max_iter", 5)
        params.setdefault("n_nearest_features", 30)
        params.setdefault("random_state", 8004)

        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=50,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=8004,
            ),
            **params,
        )
    else:
        raise ValueError(f"Unknown imputer: {method}")