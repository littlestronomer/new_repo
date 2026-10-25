from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 8004
    n_splits: int = 5
    n_repeats: int = 1
    n_jobs: int = -1


config = Config()

from sklearn.model_selection import ParameterGrid


# available methods and parameters
OPTIONS = {
    "outliers": {
        "passthrough": {"method": ["passthrough"]},
        "mahalanobis": {
            "method": ["mahalanobis"],
            "alpha": [0.001, 0.01],
            "variance": [0.8, 0.95],
        },
        "isolation_forest": {
            "method": ["isolation_forest"],
            "contamination": [0.005, 0.01, 0.03],
            "n_estimators": [200],
        },
        "lof": {
            "method": ["lof"],
            "contamination": [0.005, 0.01, 0.03],
            "n_neighbors": [20, 40],
            "variance": [0.8],
        },
    },

    "scalers": {
        "passthrough": {"method": ["passthrough"]},
        "standard": {"method": ["standard"]},
        "robust": {"method": ["robust"]},
        "quantile": {
            "method": ["quantile"],
            "n_quantiles": [100, 200],
        },
        "power": {"method": ["power"]},
        "minmax": {"method": ["minmax"]},
    },

    "imputers": {
        "passthrough": {"method": ["passthrough"]},
        "median": {"method": ["median"]},
        "mean": {"method": ["mean"]},
        "most_frequent": {"method": ["most_frequent"]},
        "knn": {
            "method": ["knn"],
            "n_neighbors": [5, 15, 30],
        },
        "iterative": {
            "method": ["iterative"],
            "max_iter": [5],
            "n_nearest_features": [20, 50],
        },
        "iterative_forest": {
            "method": ["iterative_forest"],
            "max_iter": [5],
            "n_nearest_features": [20, 30],
        },
    },

    "filters": {
        "passthrough": {"method": ["passthrough"]},
        "variance": {
            "method": ["variance"],
            "threshold": [0.0, 1e-4],
        },
        "unsupervised": {
            "method": ["unsupervised"],
            "min_partner": [0.0, 0.3],
            "max_correlation": [0.95, 0.99],
        },
        "combined": {
            "method": ["combined"],
            "steps": [[
                {"method": "variance", "threshold": 0.0},
                {"method": "unsupervised", "min_partner": 0.0, "max_correlation": 0.99},
            ]],
        },
    },

    "selectors": {
        "passthrough": {"method": ["passthrough"]},
        "pca": {
            "method": ["pca"],
            "n_components": [50, 100, 150],
        },
        "lasso": {
            "method": ["lasso"],
            "alpha": [0.01, 0.03, 0.1],
        },
        "kbest": {
            "method": ["kbest"],
            "k": [50, 100, 150, 200],
        },
        "mutual_info": {
            "method": ["mutual_info"],
            "k": [50, 100, 150, 200],
        },
        "tree": {
            "method": ["tree"],
            "threshold": ["mean", "median"],
        },
        "rfe": {
            "method": ["rfe"],
            "n_features_to_select": [50, 100],
            "step": [0.1],
        },
    },

    "models": {
        "linear": {"method": ["linear"]},
        "ridge": {
            "method": ["ridge"],
            "alpha": [0.1, 1.0, 10.0],
        },
        "elastic_net": {
            "method": ["elastic_net"],
            "alpha": [0.01, 0.1],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "krr": {
            "method": ["krr"],
            "kernel": ["rbf"],
            "alpha": [0.003, 0.01, 0.03],
            "gamma_mult": [2, 2.5, 3, 3.5],
        },
        "gpr": {
            "method": ["gpr"],
            "kernel": ["rbf", "matern", "rq", "linear", "rbf+linear"],
            "nu": [2.5],
        },
        "svr": {
            "method": ["svr"],
            "C": [1, 10, 100],
            "epsilon": [0.01, 0.1],
        },
        "hgb": {
            "method": ["hgb"],
            "max_iter": [100, 300],
            "learning_rate": [0.05, 0.1],
        },
        "extra_trees": {
            "method": ["extra_trees"],
            "n_estimators": [200, 500],
            "min_samples_leaf": [1, 3],
        },
        "random_forest": {
            "method": ["random_forest"],
            "n_estimators": [200, 500],
            "min_samples_leaf": [1, 3],
        },
        "catboost": {
            "method": ["catboost"],
            "iterations": [300, 700],
            "depth": [4, 6],
            "learning_rate": [0.03, 0.1],
        },
        "lightgbm": {
            "method": ["lightgbm"],
            "n_estimators": [300, 700],
            "num_leaves": [7, 15],
            "learning_rate": [0.03, 0.1],
        },
        "xgboost": {
            "method": ["xgboost"],
            "n_estimators": [300, 700],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.1],
        },
    },
}

