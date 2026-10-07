# Macro-financial network paper

Current reproducibility materials are organized as follows:

- `src/`: audited analysis code.
- `scripts/`: runnable analysis scripts.
- `paper/`: manuscript-facing workspace.
- `paper/results/`: numerical tables used by the paper.
- `data/README.md`: expected input data structure.

Legacy ZIP/RAR files in the repository root are retained from earlier development. The current paper uses the folders listed above.

The workflow includes the candidate graph, temporal validation, Ridge/XGBoost model selection, SHAP attribution, grouped ablation, recursive response analysis, and first-difference sensitivity.

See `paper/README.md` for the result map and `CITATION.cff` for citation metadata.
