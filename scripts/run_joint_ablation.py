from pathlib import Path
import sys, json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import macro_network_pipeline as m

full = m.load_panel(m.DATA_FILE)
perf = pd.read_csv(ROOT/'results'/'primary'/'aggregate'/'ALL_performance.csv',dtype={'target':'string','model':'string','transform':'string','params':'string','selected_sources':'string'})
rows=[]
for _,r in perf[perf.reliable==True].iterrows():
    t=str(r['target']); fold=int(r['fold']); d,features,groups=m.build_target_frame(full,t); sp=m.outer_splits(d,t)[fold-1]; tr,va=sp['train'],sp['valid']
    ms=m.ModelSpec(str(r['model']),json.loads(str(r['params'])),str(r['transform']))
    srcs=[] if pd.isna(r['selected_sources']) else [x for x in str(r['selected_sources']).split(';') if x]
    fullb=m.fit_bundle(tr,t,m.features_from_sources(groups,srcs),ms); fullm=m.evaluate_bundle(fullb,tr,va,t)
    selfb=m.fit_bundle(tr,t,m.features_from_sources(groups,[]),ms); selfm=m.evaluate_bundle(selfb,tr,va,t)
    rows.append({'target':t,'fold':fold,'full_bnmae':fullm['balanced_nmae'],'self_bnmae':selfm['balanced_nmae'],
                 'delta_self_minus_full':selfm['balanced_nmae']-fullm['balanced_nmae'],
                 'full_r2':fullm['r2'],'self_r2':selfm['r2'],'selected_sources':';'.join(srcs)})
df=pd.DataFrame(rows)
out=ROOT/'results'/'primary'/'aggregate'; out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/'joint_cross_ablation.csv',index=False)
s=df.groupby('target').agg(mean_delta=('delta_self_minus_full','mean'),median_delta=('delta_self_minus_full','median'),support_fraction=('delta_self_minus_full',lambda x:float((x>0).mean())),n_folds=('fold','count')).reset_index()
s.to_csv(out/'joint_cross_ablation_summary.csv',index=False)
print('Joint ablation DONE')
