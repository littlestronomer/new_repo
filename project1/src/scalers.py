from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler


def get_scaler(method="passthrough", **params):
    if method == "passthrough":
        return FunctionTransformer()
    elif method == "standard":
        return StandardScaler(**params)
    elif method == "robust":
        return RobustScaler(*params)
    else:
        raise ValueError(f"Unknown scaler: {method}")
