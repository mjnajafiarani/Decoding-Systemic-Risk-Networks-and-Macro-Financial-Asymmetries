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

