# Reproducible pre-2023 diagnostics for the macro-financial network.
# IMPORTANT: all diagnostics are restricted to years <= 2022.
#
# Test settings:
# ADF: constant, maxlag <= 2, autolag="AIC"
# KPSS: constant, nlags="auto"
# First-difference diagnostic: ADF on Δx
# Zivot-Andrews: regression="ct", trim=0.15, maxlag <= 2, autolag="AIC"
# Pesaran CD: common complete years across 5 countries; levels and first differences
# Cointegration: Engle-Granger (target on source), trend="c", maxlag <= 2, autolag="aic";
# only when both country series are classified I(1) or likely I(1); Fisher aggregation is descriptive.
#
# The full numeric outputs are retained in the diagnostics workbook used for the paper.
