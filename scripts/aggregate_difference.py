from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import macro_network_pipeline as m

src = ROOT / "results" / "robustness_first_difference" / "by_target"
out = ROOT / "results" / "robustness_first_difference" / "aggregate"
out.mkdir(parents=True, exist_ok=True)

def read_many(suffix, dtype):
    files = sorted(src.glob(f"*_{suffix}.csv"))
    if not files:
        raise FileNotFoundError(f"No *_{suffix}.csv files found")
    return pd.concat([pd.read_csv(f, dtype=dtype) for f in files], ignore_index=True)

perf = read_many("performance", {"target":"string","model":"string","transform":"string","params":"string","selected_sources":"string"})
sel = read_many("selection", {"target":"string","source":"string","model":"string","transform":"string"})
shap = read_many("shap", {"target":"string","model":"string","transform":"string","feature":"string"})

agg = m.aggregate_w(shap, pd.DataFrame(), sel)
for k, v in agg.items():
    v.to_csv(out / f"{k}.csv", index=False)

pr = perf[perf.reliable == True]
summary = pr.groupby("target").agg(
    n_folds=("fold","count"),
    r2_mean=("r2","mean"),
    r2_sd=("r2","std"),
    bnmae_mean=("balanced_nmae","mean"),
    skill_persist_mean=("skill_vs_persistence","mean"),
    n_sources_mean=("n_selected_sources","mean")
).reset_index()

full = m.load_panel(m.DATA_FILE)
for v in m.VARS10:
    z = np.arcsinh(full[v].astype(float)) if v == "INF" else full[v].astype(float)
    full[v] = pd.Series(z, index=full.index).groupby(full.Country).diff()

rows = []
for t in m.VARS10:
    d, _, _ = m.build_target_frame(full, t)
    for sp in m.outer_splits(d, t):
        if not sp["reliable"]:
            continue
        tr, va = sp["train"], sp["valid"]
        y = pd.to_numeric(va[t], errors="coerce").to_numpy(float)
        pred = np.zeros(len(y))
        scales = m.country_scales(tr, t)
        rows.append({
            "target":t,
            "fold":sp["fold"],
            "zero_bnmae":m.balanced_nmae(y, pred, va.Country, scales),
            "zero_r2":r2_score(y, pred)
        })

zero = pd.DataFrame(rows)
zero_summary = zero.groupby("target").agg(
    zero_bnmae_mean=("zero_bnmae","mean"),
    zero_r2_mean=("zero_r2","mean")
).reset_index()

summary = summary.merge(zero_summary, on="target")
summary["skill_vs_zero"] = 1 - summary.bnmae_mean / summary.zero_bnmae_mean
summary.to_csv(out / "performance_summary.csv", index=False)

level = pd.read_csv(
    ROOT / "results" / "primary" / "aggregate" / "W_edges_sorted.csv",
    dtype={"source":"string","target":"string"}
)[["source","target","W","S_select"]].rename(
    columns={"W":"W_level","S_select":"S_level"}
)
diff = agg["W_edges"][["source","target","W","S_select"]].rename(
    columns={"W":"W_diff","S_select":"S_diff"}
)
comparison = level.merge(diff, on=["source","target"], how="outer").fillna(
    {"W_level":0,"S_level":0,"W_diff":0,"S_diff":0}
)
comparison["stable_level"] = comparison.S_level >= 2/3
comparison["stable_diff"] = comparison.S_diff >= 2/3
comparison["both"] = comparison.stable_level & comparison.stable_diff
comparison.to_csv(out / "level_diff_edge_comparison.csv", index=False)

print("Difference aggregation DONE")
