from pathlib import Path
import sys, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import macro_network_pipeline as m

target = sys.argv[1].upper()
if target not in m.VARS10:
    raise SystemExit(f"Unknown target {target}. Choose from {m.VARS10}")
outdir = ROOT / "results" / "primary" / "by_target"
outdir.mkdir(parents=True, exist_ok=True)
full = m.load_panel(m.DATA_FILE)
d, features, groups = m.build_target_frame(full, target)
perf, sels, shaps, supports = [], [], [], []
for sp in m.outer_splits(d, target):
    fold, tr, va = sp["fold"], sp["train"], sp["valid"]
    if not sp["reliable"]:
        perf.append({"target": target, "fold": fold, "reliable": False,
                     "n_train": len(tr), "n_valid": len(va)})
        continue
    sel_spec, selected_sources, selected_features, inner_score, _ = m.select_pipeline_nested(tr, target, groups)
    b = m.fit_bundle(tr, target, selected_features, sel_spec)
    met = m.evaluate_bundle(b, tr, va, target)
    perf.append({"target": target, "fold": fold, "reliable": True,
                 "n_train": len(tr), "n_valid": len(va),
                 "model": sel_spec.kind, "transform": sel_spec.transform,
                 "params": json.dumps(sel_spec.params, sort_keys=True),
                 "inner_balanced_nmae": inner_score,
                 "n_selected_sources": len(selected_sources),
                 "selected_sources": ";".join(selected_sources), **met})
    for src in m.REL[target]:
        sels.append({"target": target, "source": src, "fold": fold,
                     "selected": src in selected_sources,
                     "model": sel_spec.kind, "transform": sel_spec.transform})
    shaps.append(m.shap_feature_table(b, va, fold))
    sup = m.grouped_edge_support(b, tr, va, target, groups)
    sup["fold"] = fold; sup["model"] = sel_spec.kind; sup["transform"] = sel_spec.transform
    supports.append(sup)

pd.DataFrame(perf).to_csv(outdir / f"{target}_performance.csv", index=False)
pd.DataFrame(sels).to_csv(outdir / f"{target}_selection.csv", index=False)
(pd.concat(shaps, ignore_index=True) if shaps else pd.DataFrame()).to_csv(outdir / f"{target}_shap.csv", index=False)
(pd.concat(supports, ignore_index=True) if supports else pd.DataFrame()).to_csv(outdir / f"{target}_support.csv", index=False)
print(target, "DONE")
