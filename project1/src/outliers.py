import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import chi2


def _prepare_X(X):
    # replace missing values with median, or 0 if column is empty
    X = pd.DataFrame(X)
    return X.fillna(X.median()).fillna(0).to_numpy()


def mahalanobis_outliers(X, alpha=0.01, variance=0.8, scaler=RobustScaler):
    # impute and scale
    X = _prepare_X(X)
    X = scaler().fit_transform(X)

    # pca
    pca = PCA(n_components=variance, svd_solver="full")
    X_pca = pca.fit_transform(X)

    # compute mahalanobis distances
    mean = np.mean(X_pca, axis=0)
    cov = np.atleast_2d(np.cov(X_pca, rowvar=False))
    cov_inv = np.linalg.pinv(cov)

    diff = X_pca - mean
    d_squared = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)

    # significance level threshold
    threshold = chi2.ppf(1 - alpha, df=X_pca.shape[1])

    return d_squared > threshold


def isolation_forest_outliers(
    X, contamination=0.01, n_estimators=200, random_state=8004
):
    # impute missing values
    X = _prepare_X(X)

    # fit isolation forest
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    return model.fit_predict(X) == -1


def lof_outliers(
    X, contamination=0.01, n_neighbors=20, variance=0.8
):
    # impute and scale
    X = _prepare_X(X)
    X = RobustScaler().fit_transform(X)

    # optional pca
    if variance is not None:
        pca = PCA(n_components=variance, svd_solver="full")
        X = pca.fit_transform(X)

    # fit local outlier factor
    model = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=min(n_neighbors, len(X) - 1),
        n_jobs=-1,
    )

    return model.fit_predict(X) == -1


def get_outliers(X, method="passthrough", **params):
    if method == "passthrough":
        return np.zeros(len(X), dtype=bool)
    elif method == "mahalanobis":
        return mahalanobis_outliers(X, **params)
    elif method == "isolation_forest":
        return isolation_forest_outliers(X, **params)
    elif method == "lof":
        return lof_outliers(X, **params)
    else:
        raise ValueError(f"Unknown method: {method}")