"""Select already-trained candidates using training-human blocks only.
Human CSV: split (train/test), schedule, q, distance, mt, peak_speed, success.
Candidates are produced with train.py --config FILE --out ROOT/candidate_NAME,
then evaluate.py --root ROOT/candidate_NAME. No retraining on held-out people.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def means(d):
    d=d[(d.success==1)&(d.mt>0)&np.isfinite(d.peak_speed)].copy()
    d['log_mt']=np.log(d.mt)
    d['distance_bin']=pd.cut(d.distance,[0,.10,.14,np.inf],labels=[0,1,2],right=False)
    return d.groupby(['schedule','q','distance_bin'],observed=True)[['log_mt','peak_speed']].mean(),d

def loss(sim,human,sd):
    aligned=sim.reindex(human.index)
    if aligned.isna().any().any(): return float('inf')
    return float((((aligned-human)/sd)**2).to_numpy().sum())

def main():
    p=argparse.ArgumentParser();p.add_argument('--human',required=True)
    p.add_argument('--candidates',required=True);p.add_argument('--out',default='human_fit')
    a=p.parse_args(); h=pd.read_csv(a.human)
    required={'split','schedule','q','distance','mt','peak_speed','success'}
    if not required.issubset(h): p.error(f'Required columns: {sorted(required)}')
    if not {'train','test'}.issubset(set(h.split)): p.error('Provide preassigned train and test blocks')
    hm,train=means(h[h.split=='train']); test,_=means(h[h.split=='test'])
    if hm.empty or test.empty: p.error('No usable human trials in one split')
    sd=train[['log_mt','peak_speed']].std()
    if not np.isfinite(sd).all() or (sd<=0).any(): p.error('Training metrics require positive finite SD')
    candidates={}
    for f in Path(a.candidates).rglob('evaluation/trials.csv'):
        meta=json.loads((f.parent.parent/'config.json').read_text())
        if meta.get('smoke'): continue
        key=json.dumps({'model':meta['model'],'config':meta['config']},sort_keys=True)
        candidates.setdefault(key,[]).append(pd.read_csv(f))
    if not candidates: p.error('No non-smoke evaluated candidates found')
    scores=[]; summaries={}
    for key,frames in candidates.items():
        # Mean per training seed first; then average seeds equally.
        m=pd.concat([means(f)[0] for f in frames]).groupby(level=[0,1,2],observed=True).mean()
        summaries[key]=m; scores.append(dict(candidate=key,train_loss=loss(m,hm,sd),n_seeds=len(frames)))
    scores=pd.DataFrame(scores).sort_values('train_loss')
    if not np.isfinite(scores.iloc[0].train_loss): p.error('No candidate covers all training-human bins')
    # Select one winner per objective without consulting the test split.
    scores['model']=scores.candidate.map(lambda x:json.loads(x)['model'])
    winners=[]
    for model,g in scores.groupby('model'):
        row=g.iloc[0].to_dict(); row['test_loss']=loss(summaries[row['candidate']],test,sd);winners.append(row)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    scores.to_csv(out/'training_selection.csv',index=False)
    pd.DataFrame(winners).to_csv(out/'heldout_results.csv',index=False)
    print(pd.DataFrame(winners)[['model','train_loss','test_loss','n_seeds']].to_string(index=False))
if __name__=='__main__':main()
