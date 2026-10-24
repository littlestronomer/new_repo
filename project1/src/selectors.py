from functools import partial

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    RFE,
    f_regression,
    mutual_info_regression,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import ExtraTreesRegressor


def get_selector(method="passthrough", **params):
    if method == "passthrough":
        return FunctionTransformer()

    elif method == "pca":
        params.setdefault("n_components", 100)
        return PCA(**params)

    elif method == "lasso":
        params.setdefault("alpha", 0.5)
        params.setdefault("max_iter", 10000)
        return SelectFromModel(Lasso(**params))

    elif method == "kbest":
        params.setdefault("k", 100)
        return SelectKBest(score_func=f_regression, **params)

    elif method == "mutual_info":
        params.setdefault("k", 100)
        score_func = partial(mutual_info_regression, random_state=8004)
        return SelectKBest(score_func=score_func, **params)

    elif method == "tree":
        threshold = params.pop("threshold", "median")
        return SelectFromModel(
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=8004,
                n_jobs=-1,
            ),
            threshold=threshold,
            **params,
        )

    elif method == "rfe":
        params.setdefault("n_features_to_select", 100)
        params.setdefault("step", 0.1)
        return RFE(estimator=Ridge(), **params)

    else:
        raise ValueError(f"Unknown selector: {method}")