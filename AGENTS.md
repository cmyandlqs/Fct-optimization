# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `scripts/`:
- `model.py`: shared `SubModel`, `MODEL_CONFIG`, routing (`route_model`), and path constants.
- `train_fct_predictor.py`: training pipeline for 4 independent models.
- `predict_optimal_threshold.py`: inference + gradient-descent threshold optimization.
- `check_env.py`, `eda.py`, and data utilities support setup and analysis.

Data is under `dataset/` (`raw_data/`, `cleaned_data/`, `all_data/`, `new_data/`). Trained artifacts are saved to `models/<model_name>/`. Plots and reports go to `analysis/`. SwanLab local run logs go to `logs/train/` and `logs/inference/`.

## Build, Test, and Development Commands
Use conda env `DL`, then run from `scripts/`:
```bash
conda activate DL
cd scripts
python check_env.py                  # Validate runtime dependencies
python train_fct_predictor.py        # Train and save 4 sub-models
python predict_optimal_threshold.py  # Run threshold optimization inference
python eda.py                        # Generate exploratory charts in analysis/
```

## Coding Style & Naming Conventions
Follow Python conventions already used in `scripts/`:
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep shared model/routing logic in `scripts/model.py`; avoid duplicating constants across scripts.
- Prefer type hints for new public helpers and keep docstrings concise.
- Keep file/path handling via `pathlib.Path`.

## Testing Guidelines
There is no dedicated unit-test framework yet. Before submitting changes:
- Run `python check_env.py`.
- Run at least one end-to-end script relevant to your change (`train_fct_predictor.py` or `predict_optimal_threshold.py`).
- Verify new outputs in `analysis/` or `models/` and summarize key metrics (MAE/RMSE/MAPE or predicted FCT) in the PR.

## Commit & Pull Request Guidelines
Current history is minimal (`学习率不变的版本`), so keep commits short and task-focused:
- Use one-line, imperative summaries (Chinese or English), e.g. `adjust cache optimizer lr`.
- Keep related code/data-path changes in the same commit.

PRs should include:
- What changed and why.
- Commands run locally.
- Before/after metrics or sample inference output.
- Linked issue or experiment context when applicable.

## Configuration Tips
`BASE_DIR` defaults to repo root in `scripts/model.py` and can be overridden via env var `FCT_BASE_DIR`.

## TODO
1. Measure real FCT values for dynamic optimal thresholds (not only predicted FCT).
2. Add more samples for low-data scenarios (`search`, `mine`).
