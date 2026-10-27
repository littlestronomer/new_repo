# NEXT STEPS — Project 1: Brain Age Prediction (Task 1)

**Status (2026-10-25):** CV R² = **0.6740** (5-fold, TabPFN ensemble + pseudo-labeling).
Submission file ready: `project1/runs/submission.csv` (regenerate anytime with `uv run python -m project1.predict`).

| Milestone | CV R² |
|---|---|
| GBM pipeline baseline | 0.503 |
| Zero-shot TabPFN + preprocessing | 0.654 |
| + EDA-driven filters (winsorize/dedup) | 0.660 |
| + selector sweep (lasso 260 / tree 414) | 0.661 |
| Ensemble v1 (2 members) | 0.6656 |
| Ensemble v2 (pseudo-labeling) | 0.6740 |
| Ensemble v3 (round-2 pseudo) | 0.6739 (converged) |

---

## 0. Immediate actions (do these first)

1. **Upload `project1/runs/submission.csv` to the course leaderboard.**
   Record the returned test R² — it calibrates every decision below
   (EDA found no train/test shift, so expect ≈ CV ± 0.03).
2. **Check the course rules on external/public data.**
   This single fact decides Path A vs Path B below.

---

## Path A — OASIS-3 external-data route *(only if rules allow external data)*

Evidence gathered: the data is almost certainly **OASIS-3** session-level FreeSurfer
morphometry (age range 42–97 matches exactly; ~25–50 same-brain longitudinal pairs
inside train; zero train/test subject overlap; perturbations = duplicated/affine-copied
columns, 3 extreme-rescaled columns, injected cell outliers, MCAR missingness 5–10%).

Steps:
1. Request OASIS-3 access (free, data-use agreement): https://sites.wustl.edu/oasisbrains / https://www.nitrc.org/projects/oasis3
2. Download: (a) FreeSurfer cross-sectional morphometry tables (aseg + aparc per session),
   (b) demographics with **age at each session** (not screening age).
3. **Column matching** (the perturbations renamed/shuffled/scaled columns):
   - Clean our features: drop constant cols, winsorize cells at robust |z|>10, column z-score.
   - For each of our 828 columns, compute its correlation *profile* against all other columns.
   - Match to OASIS-3 feature profiles (rank-correlation on profiles, Hungarian algorithm).
   - Affine copies (slope/scaling) are invariant under rank correlation → duplicates match the same source column.
   - Validate: matched columns must produce coherent groups (our 5 duplicate groups should collapse onto single OASIS-3 features).
4. **Row matching**: match our train rows to OASIS-3 sessions (column-matched deviation-space correlation, as in `diagnostics.py` style analysis). Success criterion: >90% of train rows find a partner with r > 0.9 and its age within ±2y of `y_train` — that *proves* the mapping and unlocks test rows.
5. Two ways to use it (check rules for which is permissible):
   - **A1 (augmentation, usually safer)**: append matched OASIS-3 rows (features→our column order, ages as labels) to training; rerun the ensemble v2/v3 protocol. Thousands of extra rows also make TabPFN **finetuning** viable (it failed on 970 rows alone).
   - **A2 (direct lookup, only if explicitly allowed)**: match test rows → use the session's true age directly.
6. Regenerate submission with the augmented model (extend `project1/predict.py`).

Expected: R² 0.75–0.85 if matching succeeds.

## Path B — Close-out *(if external data is forbidden)*

We have strong evidence 0.674 ≈ the information ceiling of these features
(structurally near-identical brains span ±18y — pathology mimics aging in OASIS-3;
residuals carry no signal; extremes unrankable). Close-out:

1. **Confirmation run** with repeated CV for an honest final number:
   - In `project1/src/config.py` set `n_repeats=3` (15 folds), rerun `ensemble_v2.py`-style protocol on the champion members (~45 min GPU).
   - Re-optimize blend weights *per repeat* and average (avoids OOF weight overfit).
2. Regenerate `runs/submission.csv` with confirmed weights; upload final submission.
3. Write the report (structure below).

### Report outline (ready to paste)
- **Preprocessing (required 3 steps)**: outlier detection (iForest 1% rows + winsorize |z|>10 cells — x742/x413/x115 sabotage), feature selection (lasso α=0.1 → 260 / tree → 414; ≤200 hurts, all-828 hurts), imputation (KNN k=15; MCAR verified, TabPFN-native NaN ≈ KNN, median much worse).
- **Model**: TabPFN (HF tabular foundation model, in-context learning, sklearn API) — why: +0.15 over tuned GBMs.
- **Ensembling & semi-supervision**: 4-member blend + 1 round of pseudo-labeling on X_test (+0.008–0.014), converged at round 2.
- **Negative results** (with numbers): TabPFN finetuning (−0.005/−0.010), hierarchical classifier-routing (−0.03), linear/isotonic calibration (0/−0.005), GBM ensemble members (zero blend weight), duplicate-column averaging (signal-diluting).
- **Analysis**: age-extremes diagnosis (42–54 bias −6.9y, 85–97 +8.5y; corr ≈ 0 within extremes), same-brain ±18y pairs → information-ceiling argument.

---

## Repo guide

```bash
uv sync                      # install deps
cp .env.example .env         # if present; else create .env with TABPFN_TOKEN=<key>
uv run python -m project1.main            # grid experiments (CV, cached)
uv run python -m project1.ensemble_v2     # OOF ensemble + pseudo-labeling
uv run python -m project1.predict         # regenerate submission.csv
uv run python -m project1.diagnostics     # residual + outlier analysis (CPU)
```

| File | Purpose |
|---|---|
| `project1/main.py` | grid experiments via the runner (cached in `runs/cache/`) |
| `project1/ensemble.py` / `_v2` / `_v3` | OOF ensembles; v2 adds pseudo-labeling, v3 round-2 |
| `project1/predict.py` | final submission generator (champion recipe) |
| `project1/diagnostics.py` | residual analysis, calibration test, outlier audit |
| `project1/hierarchical.py` | classifier-routing experiment (negative result) |
| `project1/finetune_tabpfn.py` | TabPFN finetuning experiment (negative result) |
| `project1/notebooks/playground.ipynb` | perturbation EDA |
| `project1/src/*` | framework: config, data, filters, imputers, scalers, selectors, outliers, models, runner |

**Environment notes**
- GPU (CUDA) required for practical TabPFN runs; first fit downloads weights from HuggingFace.
- `.env` must contain `TABPFN_TOKEN` (one-time license at https://ux.priorlabs.ai; never commit it — already in `.gitignore`).
- All experiment scripts must run **from the repo root**.
- Results land in `project1/runs/` (per-run CSVs + `ensemble_*.json` + `hierarchical_result.json`).
