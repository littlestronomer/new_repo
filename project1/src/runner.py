import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from itertools import product

from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from project1.src.outliers import get_outliers
from project1.src.imputation import get_imputer
from project1.src.scalers import get_scaler
from project1.src.filter import get_filter
from project1.src.selectors import get_selector
from project1.src.models import get_model
from project1.src.config import config

def run_experiment(
    X_train,
    y_train,
    outliers,
    scalers,
    imputers,
    filters,
    selectors,
    models,
):
    # create cv sets
    splits = list(
        RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
        ).split(X_train)
    )
    n_fits = (
        len(outliers)
        * len(scalers)
        * len(imputers)
        * len(filters)
        * len(selectors)
        * len(models)
        * len(splits)
    )
    print("Model fits:", n_fits)

    # init loop
    out_file = "project1/runs/" + datetime.now().strftime("run_%Y%m%d_%H%M.csv")
    pbar = tqdm(total=n_fits, desc="model fits")
    rows = []

    # outliers loop (does not depend on model)
    for o in outliers:
        mask_outliers = [get_outliers(X_train.iloc[tr], **o) for tr, _ in splits]
        n_outliers = np.mean([m.sum() for m in mask_outliers])

        # scaler, imputer, selectors loop
        for scaler, imputer, filt, selector in product(
            scalers, imputers, filters, selectors
        ):
            folds = []

            # cv loop
            for (tr, val), mask in zip(splits, mask_outliers):
                # drop outliers in train
                X_tr, y_tr = X_train.iloc[tr][~mask], y_train.iloc[tr][~mask]

                # build pipeline
                prep = Pipeline(
                    [
                        ("filter", get_filter(**filt)),
                        ("scaler", get_scaler(**scaler)),
                        ("imputer", get_imputer(**imputer)),
                        ("selector", get_selector(**selector)),
                    ]
                )

                # transform
                Z_tr = prep.fit_transform(X_tr, y_tr)
                Z_val = prep.transform(X_train.iloc[val])
                folds.append((Z_tr, y_tr, Z_val, y_train.iloc[val]))

            # modle loop
            for m in models:
                # fit model
                scores = [
                    r2_score(
                        y_val,
                        get_model(**m, n_features=Z_tr.shape[1])
                        .fit(Z_tr, y_tr)
                        .predict(Z_val),
                    )
                    for Z_tr, y_tr, Z_val, y_val in folds
                ]

                # update progress
                pbar.update(len(splits))

                # store results
                rows.append(
                    {
                        "outliers": o,
                        "outliers_detected": n_outliers,
                        "scaler": scaler,
                        "filter": filt,
                        "imputer": imputer,
                        "selector": selector,
                        "model": m,
                        "r2_mean": np.mean(scores),
                        "r2_std": np.std(scores),
                        "r2_folds": scores,
                        "n_features": folds[0][0].shape[1],
                    }
                )

                # print best score
                pbar.set_postfix(best=round(max(r["r2_mean"] for r in rows), 3))

            pd.json_normalize(rows).to_csv(out_file, index=False)

    pbar.close()
    return pd.json_normalize(rows).sort_values(
        "r2_mean", ascending=False
    )

