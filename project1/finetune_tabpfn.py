"""Finetune TabPFN on the brain-age task; compare against zero-shot on a holdout.

Uses the best preprocessing configuration found by the CV experiments:
isolation-forest outlier rows are NOT dropped here (row removal is handled by
the runner); we apply winsorize(10) -> robust scale -> KNN(15) impute ->
lasso(0.1) select, fit on the train split only, and evaluate both the
zero-shot TabPFNRegressor and the finetuned FinetunedTabPFNRegressor.
"""

from dotenv import load_dotenv

load_dotenv()  # loads TABPFN_TOKEN etc. from .env

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from project1.src.data import data
from project1.src.filter import get_filter
from project1.src.scalers import get_scaler
from project1.src.imputation import get_imputer
from project1.src.selectors import get_selector
from project1.src.models import get_model
from project1.src.config import config

SEED = config.seed

# %% load and split

X, y, _, _ = data()
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %% preprocessing (fit on the train split only)

prep = Pipeline(
    [
        ("filter", get_filter(method="winsorize", z_threshold=10)),
        ("scaler", get_scaler(method="robust")),
        ("imputer", get_imputer(method="knn", n_neighbors=15)),
        ("selector", get_selector(method="lasso", alpha=0.1)),
    ]
)
Z_tr = prep.fit_transform(X_tr, y_tr)
Z_val = prep.transform(X_val)
y_tr = np.asarray(y_tr, dtype=float)
y_val = np.asarray(y_val, dtype=float)
print(f"preprocessed: {Z_tr.shape}")

# %% zero-shot baseline

t0 = time.time()
zs = get_model("tabpfn", n_features=Z_tr.shape[1], n_estimators=8)
zs.fit(Z_tr, y_tr)
r2_zs = r2_score(y_val, zs.predict(Z_val))
print(f"zero-shot  R2: {r2_zs:.4f}  ({time.time() - t0:.0f}s)")

# %% finetune

from tabpfn.finetuning import FinetunedTabPFNRegressor

runs_dir = Path(__file__).resolve().parent / "runs" / "finetune"
runs_dir.mkdir(parents=True, exist_ok=True)

ft = FinetunedTabPFNRegressor(
    device="cuda",
    epochs=60,
    time_limit=900,  # 15 min safety cap
    learning_rate=1e-4,
    weight_decay=0.01,
    crps_loss_weight=0.0,
    mse_loss_weight=1.0,
    random_state=SEED,
)
t0 = time.time()
ft.fit(Z_tr, y_tr, output_dir=runs_dir)
r2_ft = r2_score(y_val, ft.predict(Z_val))
print(f"finetuned R2: {r2_ft:.4f}  ({time.time() - t0:.0f}s)")

print(f"\ndelta vs zero-shot: {r2_ft - r2_zs:+.4f}")
