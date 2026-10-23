import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


def mahalanobis_outliers(X, alpha=0.01, variance=0.8, scaler=RobustScaler):
    # replace na
    X_outliers = X.fillna(X.median())

    # scale
    X_outliers = scaler().fit_transform(X_outliers)

    # pca
    pca = PCA(n_components=variance, svd_solver="full")
    X_pca = pca.fit_transform(X_outliers)

    # compute mahalanobis distance
    mean = np.mean(X_pca, axis=0)
    cov = np.cov(X_pca, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    # get distances
    distances = np.array([mahalanobis(point, mean, cov_inv) for point in X_pca])
    d_squared = distances**2

    # significance level threshold
    threshold = chi2.ppf(1 - alpha, df=X_pca.shape[1])

    return d_squared > threshold


def get_outliers(X, method="passthrough", **params):
    if method == "passthrough":
        return np.zeros(len(X), dtype=bool)

    elif method == "mahalanobis":
        return mahalanobis_outliers(X, **params)

    else:
        raise ValueError(f"Unknown method: {method}")
