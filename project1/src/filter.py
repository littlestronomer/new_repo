import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


class UnsupervisedFilter(BaseEstimator, TransformerMixin):
    """Drop columns without a correlated partner, then one of each near-duplicate pair."""

    def __init__(self, min_partner=0.0, max_correlation=1.0):
        self.min_partner = min_partner
        self.max_correlation = max_correlation

    def fit(self, X, y=None):
        corr = np.nan_to_num(pd.DataFrame(X).corr().abs().to_numpy())
        np.fill_diagonal(corr, 0)

        keep = corr.max(axis=0) >= self.min_partner
        idx = np.flatnonzero(keep)
        pairs = np.triu(corr[np.ix_(idx, idx)], k=1)
        keep[idx[(pairs > self.max_correlation).any(axis=0)]] = False

        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def get_filter(method="passthrough", **params):
    if method == "passthrough":
        return FunctionTransformer()
    elif method == "unsupervised":
        return UnsupervisedFilter(**params)
    elif method == "variance":
        return VarianceThreshold(**params)
    elif method == "combined":
        steps = params["steps"]
        return Pipeline([
            (f"filter_{i}", get_filter(**step))
            for i, step in enumerate(steps)
        ])
    else:
        raise ValueError(f"Unknown filter: {method}")
