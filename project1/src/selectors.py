from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Lasso


def get_selector(method="passthrough", **params):
    if method == "passthrough":
        return FunctionTransformer()
    elif method == "pca":
        params.setdefault("n_components", 100)
        return PCA(**params)
    elif method == "lasso":
        params.setdefault("alpha", 0.5)
        return SelectFromModel(Lasso(max_iter=10000, **params))
    elif method == "kbest":
        params.setdefault("k", 100)
        return SelectKBest(score_func=f_regression, **params)
    else:
        raise ValueError(f"Unknown selector: {method}")
