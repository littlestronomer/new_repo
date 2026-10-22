import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


def identify_outliers(X, method, alpha, variance):
    if method == "none":
        return np.zeros(len(X), dtype=bool)

    elif method == "mahalanobis":
        # replace na
        X_outliers = X.fillna(X.mean())

        # scale
        sc = StandardScaler()
        X_outliers = sc.fit_transform(X_outliers)

        # pca
        pca = PCA(n_components=variance, svd_solver="full")
        X_pca = pca.fit_transform(X_outliers)

        # compute mahalanobis distance
        mean = np.mean(X_pca, axis=0)
        cov = np.cov(X_pca, rowvar=False)
        cov_inv = np.linalg.inv(cov)

        distances = np.array([mahalanobis(point, mean, cov_inv) for point in X_pca])

        # shape and significance level
        p = X_pca.shape[1]
        threshold = chi2.ppf(1 - alpha, df=p)
        d_squared = distances**2

        return d_squared > threshold

    else:
        raise ValueError(f"Unknown method '{method}'.")
