import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Lasso


class UnsupervisedFilter(BaseEstimator, TransformerMixin):
    """Drop low-variance features and highly correltaed features"""

    def __init__(self, max_correlation=0.95, min_variance=1e-8):
        self.max_correlation = max_correlation
        self.min_variance = min_variance

    def fit(self, X, y=None):
        # identify small variance
        X = np.asarray(X)
        keep = X.var(axis=0) > self.min_variance
        idx = np.flatnonzero(keep)

        # identify high correlation
        corr = np.abs(np.corrcoef(X[:, idx], rowvar=False))
        redundant = (np.triu(corr, k=1) > self.max_correlation).any(axis=0)

        # build feature
        keep[idx[redundant]] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


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
    elif method == "unsupervised":
        return UnsupervisedFilter(**params)
    else:
        raise ValueError(f"Unknown selector: {method}")
