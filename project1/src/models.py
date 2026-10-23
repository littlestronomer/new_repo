from operator import add
from functools import reduce

from sklearn.compose import TransformedTargetRegressor
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
)


def _standardize_y(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def _build_gpr(
    kernel="rbf",
    nu=1.5,
    sigma_0=1.0,
    length_scale=25.0,
    noise=0.1,
    noise_min=1e-5,
    **params,
):
    # default parts
    parts = {
        "rbf": RBF(length_scale),
        "matern": Matern(length_scale, nu=nu),
        "linear": DotProduct(sigma_0=sigma_0),
    }

    # split names
    kernels = kernel.split("+")

    # assert all kernels available
    unknown = set(kernels) - parts.keys()
    assert not unknown, f"Unknown GPR kernel {sorted(unknown)}"

    # build kernel and set defaults
    base = reduce(add, (parts[n] for n in kernels))

    # set default parameters
    params.setdefault("normalize_y", True)
    params.setdefault("n_restarts_optimizer", 0)
    params.setdefault("random_state", 8004)
    return GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * base
        + WhiteKernel(noise, noise_level_bounds=(noise_min, 1e1)),
        **params,
    )


def get_model(method="krr", **params):
    if method == "linear":
        return LinearRegression()
    elif method == "ridge":
        return Ridge(**params)
    elif method == "krr":
        params.setdefault("kernel", "rbf")
        return _standardize_y(KernelRidge(**params))
    elif method == "gpr":
        return _build_gpr(**params)
    elif method == "svr":
        params.setdefault("kernel", "rbf")
        return _standardize_y(SVR(**params))
    else:
        raise ValueError(f"Unknown model: {method}")
