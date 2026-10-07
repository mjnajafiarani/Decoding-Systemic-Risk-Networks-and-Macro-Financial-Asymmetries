# Paper workspace

This folder contains the manuscript-facing material for the current paper.

## Result map

- `results/performance_summary.csv`: outer-fold predictive performance.
- `results/self_history_summary.csv`: self-history and cross-variable attribution shares.
- `results/W_selection_recurrent_edges.csv`: the 19 selection-recurrent W edges.
- `results/response_main_edges.csv`: main three-year fitted-system responses.
- `results/first_difference_performance.csv`: first-difference sensitivity.
- `results/model_family_summary.csv`: Ridge/XGBoost fixed-feature diagnostic.

The audited implementation is in `../src/macro_network_pipeline.py`; runnable entry points are in `../scripts/`.

The reported outer-fold evaluation ends in 2022. The two own lags are autoregressive self-links kept separate from the reported cross-variable edge set.

For the starred incremental rule in the manuscript, both recurrent positive drop-column support and a positive mean error change are required.

Appendix A should document data sources, original frequencies, annualization rules, and the final candidate-parent specification.
