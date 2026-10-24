# Advanced Machine Learning — ETH Zürich

Code and experiments for the **Advanced Machine Learning (AML)** course at ETH Zürich.

## Project 1 — Regression

A modular machine learning framework for experimenting with regression models and preprocessing pipelines.

**Features**
- Configurable preprocessing: outlier detection, imputation, scaling, filtering, and feature selection.
- Multiple regression models, including Gaussian Processes, Kernel Ridge Regression, SVR, and tree-based ensembles.
- Automated hyperparameter experiments with cross-validation.
- Reproducible evaluation using R² scores, experiment caching, and automatic leaderboards.

## Repository Structure

| Path | Description |
|---|---|
| `project1/main.py` | Experiment configurations and entry point |
| `project1/src/` | Preprocessing, models, and evaluation framework |
| `project1/data/` | Training and test datasets |
| `project1/runs/` | Experiment results and leaderboards |
| `project1/notebooks/` | Exploratory analysis |

## Setup

Dependencies are managed using [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lemaurer/AML-Projects.git
cd AML-Projects
uv sync
```

Experiment configurations and model selection are defined in `project1/main.py`.