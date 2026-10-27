from operator import add
from functools import reduce
from project1.src.config import config

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    WhiteKernel,
    RationalQuadratic,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def _standardize_y(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def _build_gpr(
    kernel="rbf",
    nu=1.5,
    rq_alpha=1.0,
    sigma_0=1.0,
    length_scale=10.0,
    length_scale_bounds=(1.0, 1e3),
    noise=0.3,
    noise_bounds=(1e-3, 1e1),
    **params,
):
    # default parts
    parts = {
        "rbf": RBF(length_scale, length_scale_bounds),
        "matern": Matern(length_scale, length_scale_bounds, nu=nu),
        "rq": RationalQuadratic(length_scale, rq_alpha, length_scale_bounds),
        "linear": DotProduct(sigma_0=sigma_0),
    }

    # split names
    kernels = kernel.split("+")

    # assert all kernels available
    unknown = set(kernels) - parts.keys()
    if unknown:
        raise ValueError(f"Unknown GPR kernel {sorted(unknown)}. Use {list(parts)}.")

    # build kernel and set defaults
    base = reduce(add, (parts[n] for n in kernels))

    # set default parameters
    params.setdefault("normalize_y", True)
    params.setdefault("n_restarts_optimizer", 0)
    params.setdefault("random_state", config.seed)
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * base + WhiteKernel(noise, noise_bounds),
        **params,
    )


def get_model(method="krr", n_features=None, **params):
    if method == "linear":
        return LinearRegression()
    elif method == "ridge":
        return Ridge(**params)
    elif method == "krr":
        params.setdefault("kernel", "rbf")
        if "gamma_mult" in params:
            if n_features is None:
                raise ValueError("KRR requires n_features for gamma_mult")
            params["gamma"] = params.pop("gamma_mult") / n_features
        return _standardize_y(KernelRidge(**params))
    elif method == "gpr":
        if n_features is None:
            raise ValueError("GPR requires n_features")
        params.setdefault("length_scale", (n_features / 3) ** 0.5)
        return _build_gpr(**params)
    elif method == "svr":
        params.setdefault("kernel", "rbf")
        return _standardize_y(SVR(**params))
    elif method == "hgb":
        return HistGradientBoostingRegressor(random_state=8004, **params)
    elif method == "extra_trees":
        return ExtraTreesRegressor(random_state=8004, n_jobs=-1, **params)
    elif method == "random_forest":
        return RandomForestRegressor(random_state=8004, n_jobs=-1, **params)
    elif method == "elastic_net":
        return _standardize_y(ElasticNet(max_iter=20000, **params))
    elif method == "catboost":
        params.setdefault("iterations", 500)
        params.setdefault("random_seed", config.seed)
        params.setdefault("thread_count", 2)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return CatBoostRegressor(**params)
    elif method == "lightgbm":
        params.setdefault("n_estimators", 500)
        params.setdefault("random_state", config.seed)
        params.setdefault("n_jobs", 2)
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)
    elif method == "xgboost":
        params.setdefault("n_estimators", 500)
        params.setdefault("random_state", config.seed)
        params.setdefault("n_jobs", 2)
        params.setdefault("tree_method", "hist")
        return XGBRegressor(**params)
    elif method == "tabpfn":
        from tabpfn import TabPFNRegressor

        params.setdefault("random_state", config.seed)
        params.setdefault("device", "auto")
        if n_features is not None and n_features > 500:
            # 832 features exceed the TabPFN v2.0 soft limit (TabPFN-2.5 allows 2000)
            params.setdefault("ignore_pretraining_limits", True)
        return TabPFNRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {method}")
