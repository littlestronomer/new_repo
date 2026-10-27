"""Hierarchical age modeling: classification-based routing + group specialists.

Variant A: TabPFN classifier on age bins -> expectation over bin centroids.
Variant B: 3-group soft routing {young <55, middle 55-84, old 85+} with a
TabPFN regressor specialist per group, combined with group probabilities.
Both reuse the champion preprocessing (winsorize -> robust -> KNN -> lasso)
and the runner's CV protocol. No pseudo-labeling here (isolates the effect).
"""

from dotenv import load_dotenv

load_dotenv()  # loads TABPFN_TOKEN etc. from .env

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from project1.src.data import data
from project1.src.filter import get_filter
from project1.src.scalers import get_scaler
from project1.src.imputation import get_imputer
from project1.src.selectors import get_selector
from project1.src.outliers import get_outliers
from project1.src.config import config

OUTLIER_SETTING = {"method": "isolation_forest", "contamination": 0.01, "n_estimators": 200}
GROUPS = [("young", 42, 55), ("middle", 55, 85), ("old", 85, 98)]


def make_prep():
    return Pipeline(
        [
            ("filter", get_filter(method="winsorize", z_threshold=10)),
            ("scaler", get_scaler(method="robust")),
            ("imputer", get_imputer(method="knn", n_neighbors=15)),
            ("selector", get_selector(method="lasso", alpha=0.1)),
        ]
    )


def group_of(age):
    for g, lo, hi in GROUPS:
        if lo <= age < hi:
            return g
    return "old"


def main():
    import gc

    import torch
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    def clear_vram():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    X, y, _, _ = data()
    y = np.asarray(y, dtype=float)
    y_grp = pd.Series([group_of(a) for a in y], index=X.index)
    grp_names = [g for g, _, _ in GROUPS]

    splits = list(
        RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
        ).split(X)
    )
    mask_outliers = [get_outliers(X.iloc[tr], **OUTLIER_SETTING) for tr, _ in splits]

    # bin variants for variant A: (name, width in years)
    bin_defs = [("bins5", 5), ("bins8", 8)]

    oof = {
        "regressor": pd.Series(np.nan, index=X.index),
        "bins5": pd.Series(np.nan, index=X.index),
        "bins8": pd.Series(np.nan, index=X.index),
        "hier3": pd.Series(np.nan, index=X.index),
        "hier3_hard": pd.Series(np.nan, index=X.index),
    }

    for k, ((tr, val), mask) in enumerate(zip(splits, mask_outliers)):
        X_tr, y_tr = X.iloc[tr][~mask], y[tr][~mask]
        X_val = X.iloc[val]
        g_tr, g_val = y_grp.iloc[tr][~mask], y_grp.iloc[val]
        print(f"--- fold {k + 1}/{len(splits)} ---")

        prep = make_prep()
        Z_tr = prep.fit_transform(X_tr, y_tr)
        Z_val = prep.transform(X_val)

        # plain regressor reference (same preprocessing)
        reg = TabPFNRegressor(
            n_estimators=8, random_state=config.seed, keep_cache_on_device=False
        )
        reg.fit(Z_tr, y_tr)
        oof["regressor"].loc[X_val.index] = reg.predict(Z_val)
        del reg
        clear_vram()

        # variant A: fine bins -> expectation over bin centroids
        for name, width in bin_defs:
            edges = np.arange(40, 100 + width, width)
            b_tr = pd.Series(
                pd.cut(y_tr, bins=edges, right=False, labels=False), index=X_tr.index
            )
            valid = b_tr.notna()
            clf = TabPFNClassifier(
                n_estimators=8, random_state=config.seed, keep_cache_on_device=False
            )
            clf.fit(Z_tr[valid], b_tr[valid].astype(int))
            proba = clf.predict_proba(Z_val)
            classes = clf.classes_
            # centroid of each bin = mean train age inside it
            cent = {
                c: y_tr[b_tr == c].mean() for c in classes if (b_tr == c).any()
            }
            pred = sum(p * cent[c] for c, p in zip(classes, proba.T) if c in cent)
            oof[name].loc[X_val.index] = pred
            del clf
            clear_vram()
            print(f"  {name} done")

        # variant B: 3-group soft routing + specialists
        clf = TabPFNClassifier(
            n_estimators=8, random_state=config.seed, keep_cache_on_device=False
        )
        clf.fit(Z_tr, g_tr)
        proba = clf.predict_proba(Z_val)
        cls = list(clf.classes_)
        del clf
        clear_vram()

        spec_pred = {}
        for g in grp_names:
            m = (g_tr == g).to_numpy()
            if m.sum() < 10:
                spec_pred[g] = np.full(len(X_val), y_tr[m].mean() if m.any() else y_tr.mean())
                continue
            spec = TabPFNRegressor(
                n_estimators=8, random_state=config.seed, keep_cache_on_device=False
            )
            spec.fit(Z_tr[m], y_tr[m])
            spec_pred[g] = spec.predict(Z_val)
            del spec
            clear_vram()

        soft = np.zeros(len(X_val))
        hard_g = [cls[i] for i in proba.argmax(axis=1)]
        hard = np.array([spec_pred[g][i] for i, g in enumerate(hard_g)])
        for i, c in enumerate(cls):
            soft += proba[:, i] * spec_pred[c]
        oof["hier3"].loc[X_val.index] = soft
        oof["hier3_hard"].loc[X_val.index] = hard
        print("  hier3 done")

    # %% report

    print("\n=== OOF R2 ===")
    scores = {n: r2_score(y, s.loc[X.index]) for n, s in oof.items()}
    for n, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {n}: {s:.4f}")

    print("\n=== age-binned MAE (regressor vs hier3) ===")
    bins = [42, 55, 65, 75, 85, 98]
    lab = ["42-54", "55-64", "65-74", "75-84", "85-97"]
    grp = pd.cut(y, bins=bins, labels=lab, right=False)
    for name in ["regressor", "hier3", "hier3_hard"]:
        mae = pd.Series(np.abs(y - oof[name].loc[X.index]), index=X.index).groupby(grp).mean()
        print(f"  {name}: " + ", ".join(f"{l}={mae[l]:.2f}" for l in lab))

    runs_dir = Path(__file__).resolve().parent / "runs"
    (runs_dir / "hierarchical_result.json").write_text(
        json.dumps({n: float(s) for n, s in scores.items()}, indent=2)
    )
    pd.DataFrame({n: s for n, s in oof.items()}).to_csv(runs_dir / "hierarchical_oof.csv")
    print(f"\nsaved: {runs_dir / 'hierarchical_result.json'}")


if __name__ == "__main__":
    main()
