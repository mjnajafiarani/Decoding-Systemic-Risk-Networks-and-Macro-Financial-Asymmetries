from pathlib import Path
import sys, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import macro_network_pipeline as m

src = ROOT / "results" / "primary" / "by_target"
out = ROOT / "results" / "primary" / "aggregate"
out.mkdir(parents=True, exist_ok=True)

def read_many(suffix, dtype):
    files = sorted(src.glob(f"*_{suffix}.csv"))
    if not files:
        raise FileNotFoundError(f"No *_{suffix}.csv files found. Run target scripts first.")
    return pd.concat([pd.read_csv(f, dtype=dtype) for f in files], ignore_index=True)

perf = read_many("performance", {"target":"string","model":"string","transform":"string","params":"string","selected_sources":"string"})
sel  = read_many("selection", {"target":"string","source":"string","model":"string","transform":"string"})
shap = read_many("shap", {"target":"string","model":"string","transform":"string","feature":"string"})
sup  = read_many("support", {"target":"string","source":"string","model":"string","transform":"string"})
for name, df in [("ALL_performance", perf), ("ALL_selection", sel), ("ALL_shap", shap), ("ALL_support", sup)]:
    df.to_csv(out / f"{name}.csv", index=False)

aggw = m.aggregate_w(shap, sup, sel)
for k, v in aggw.items():
    v.to_csv(out / f"{k}.csv", index=False)


full = m.load_panel(m.DATA_FILE)
bundles = {1:{}, 2:{}, 3:{}}
for _, r in perf[perf.reliable == True].iterrows():
    t, fold = str(r["target"]), int(r["fold"])
    d, features, groups = m.build_target_frame(full, t)
    sp = m.outer_splits(d, t)[fold - 1]
    ms = m.ModelSpec(str(r["model"]), json.loads(str(r["params"])), str(r["transform"]))
    srcs = [] if pd.isna(r["selected_sources"]) else [x for x in str(r["selected_sources"]).split(";") if x]
    bundles[fold][t] = m.fit_bundle(sp["train"], t, m.features_from_sources(groups, srcs), ms)

resp = m.response_from_outer_folds(full, bundles, horizon=3)
resp.to_csv(out / "R_responses_raw.csv", index=False)
aggr = m.aggregate_r(resp, horizon=3)
for k, v in aggr.items():
    v.to_csv(out / f"{k}.csv", index=False)

pr = perf[perf.reliable == True].copy()
summary = pr.groupby("target").agg(
    n_folds=("fold","count"), r2_mean=("r2","mean"), r2_sd=("r2","std"),
    bnmae_mean=("balanced_nmae","mean"), bnmae_sd=("balanced_nmae","std"),
    persistence_bnmae_mean=("persistence_balanced_nmae","mean"),
    skill_vs_persistence_mean=("skill_vs_persistence","mean"),
    persistence_r2_mean=("persistence_r2","mean"),
    n_sources_mean=("n_selected_sources","mean")
).reset_index()
summary["models_by_fold"] = summary.target.map(
    pr.groupby("target").model.apply(lambda x: ";".join(x.astype(str))).to_dict()
)
summary.to_csv(out / "performance_summary.csv", index=False)

W = aggw["W_edges"].copy()
W["stable"] = W.S_select.fillna(0) >= 2/3
W["incrementally_supported"] = W.S_incremental.fillna(0) >= 2/3
W.sort_values(["stable","W"], ascending=[False,False]).to_csv(out / "W_edges_sorted.csv", index=False)

R = aggr["R_global"].copy()
R["absR"] = R.R.abs()
R["self_edge"] = R.source == R.target
R.sort_values("absR", ascending=False).to_csv(out / "R_global_sorted.csv", index=False)

sf = shap.copy()
def typ(f):
    f = str(f)
    if f.startswith("C_"): return "country"
    if "_self_lag" in f: return "self"
    if "_lag" in f: return "cross"
    return "other"

sf["type"] = sf.feature.map(typ)
rows = []
for (t, fold), g in sf.groupby(["target","fold"]):
    sums = g.groupby("type").mean_abs_shap.sum()
    dyn = sums.get("cross",0) + sums.get("self",0)
    rows.append({
        "target":t,"fold":fold,
        "cross_share":sums.get("cross",0)/dyn if dyn else 0,
        "self_share":sums.get("self",0)/dyn if dyn else 0
    })
decomp = pd.DataFrame(rows)
decomp.to_csv(out / "target_dynamic_shares_by_fold.csv", index=False)
decomp.groupby("target").agg(
    cross_share_mean=("cross_share","mean"),
    cross_share_sd=("cross_share","std"),
    self_share_mean=("self_share","mean")
).reset_index().to_csv(out / "target_dynamic_shares_summary.csv", index=False)

print("Primary aggregation DONE")
