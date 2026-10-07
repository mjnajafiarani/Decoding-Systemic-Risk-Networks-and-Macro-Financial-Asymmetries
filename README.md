# Macro-financial network paper

Current reproducibility materials are organized as follows:

- `src/`: audited analysis code.
- `scripts/`: runnable analysis scripts.
- `paper/`: manuscript-facing workspace.
- `paper/results/`: numerical tables used by the paper.
- `data/README.md`: expected input data structure.

The workflow includes the 71-edge candidate graph, temporal validation, Ridge/XGBoost model selection, out-of-fold SHAP attribution, grouped ablation, recursive response analysis, and first-difference sensitivity.

The two own lags remain in each fitted equation as autoregressive self-links and are reported separately from the cross-variable network.

See `paper/README.md` for the manuscript-to-result map and `CITATION.cff` for citation metadata.
