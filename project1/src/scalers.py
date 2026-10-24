from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    MinMaxScaler,
)


def get_scaler(method="passthrough", **params):
    if method == "passthrough":
        return FunctionTransformer()
    elif method == "standard":
        return StandardScaler(**params)
    elif method == "robust":
        return RobustScaler(**params)
    elif method == "quantile":
        params.setdefault("output_distribution", "normal")
        params.setdefault("n_quantiles", 200)
        return QuantileTransformer(**params)
    elif method == "power":
        params.setdefault("method", "yeo-johnson")
        return PowerTransformer(**params)
    elif method == "minmax":
        return MinMaxScaler(**params)
    else:
        raise ValueError(f"Unknown scaler: {method}")