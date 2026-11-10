"""Descriptive summaries. Seeds are training replicates, not human subjects."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); a=p.parse_args()
    root=Path(a.root); out=root/'comparison'; out.mkdir(exist_ok=True)
    frames=[]; blocks=[]
    for f in root.rglob('trials.csv'):
        if 'comparison' in f.parts: continue
        tag=str(f.parent.relative_to(root)); d=pd.read_csv(f); d['run']=tag; frames.append(d)
        b=pd.read_csv(f.parent/'blocks.csv'); b['run']=tag; blocks.append(b)
    if not frames: p.error('Run baseline.py or evaluate.py first')
    d=pd.concat(frames,ignore_index=True); b=pd.concat(blocks,ignore_index=True)
    d['distance_bin']=pd.cut(d.distance,[0,.10,.14,np.inf],right=False)
    good=d[(d.success==1)&(d.mt>0)].copy(); good['log_mt']=np.log(good.mt)
    bybin=good.groupby(['run','schedule','q','distance_bin'],observed=True)[['mt','log_mt','peak_speed']].mean()
    bybin.to_csv(out/'distance_bin_means.csv')
    # Equal-weight only bins present in all four cells of a given run.
    adjusted=[]
    for run,g in bybin.reset_index().groupby('run'):
        count=g.groupby('distance_bin',observed=True).size()
        bins=count[count==4].index
        z=g[g.distance_bin.isin(bins)].groupby(['schedule','q'],observed=True)[['mt','log_mt','peak_speed']].mean().reset_index()
        z['run']=run; z['matched_bins']=len(bins); adjusted.append(z)
    means=pd.concat(adjusted,ignore_index=True) if adjusted else pd.DataFrame()
    means.to_csv(out/'matched_distance_means.csv',index=False)
    b.groupby(['run','schedule','q'])[['points','utility','success_rate','waiting_distance']].mean().to_csv(out/'blocks_summary.csv')
    effects=[]
    if not means.empty:
        for run,g in means.groupby('run'):
            z=g.set_index(['schedule','q']).log_mt
            if all(k in z.index for k in [('A',1),('A',10),('B',1),('B',10)]):
                da=z.loc['A',10]-z.loc['A',1]; db=z.loc['B',10]-z.loc['B',1]
                effects.append(dict(run=run,delta_A=da,delta_B=db,interaction=da-db))
        learned=means[means.run.str.contains('/seed_')].copy()
        if not learned.empty:
            learned['label']=learned.run.str.split('/').str[0]
            plotdata=learned.groupby(['label','schedule','q'])[['mt','peak_speed']].mean().reset_index()
        else:
            plotdata=means[means.run.str.startswith('baseline/external/')].copy()
            plotdata['label']=plotdata.run.str.split('/').str[-1]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for (label,schedule),g in plotdata.groupby(['label','schedule']):
            for ax,metric in zip(axes,['mt','peak_speed']):
                ax.plot(g.q,g[metric],marker='o',linestyle='-' if schedule=='A' else '--',
                        label=f'{label} {schedule}')
        axes[0].set_ylabel('Movement time (s)'); axes[1].set_ylabel('Peak speed (m/s)')
        for ax in axes: ax.set_xlabel('Points per target'); ax.set_xticks([1,10])
        axes[1].legend(fontsize=7,loc='best')
        fig.suptitle('Descriptive simulation averages; inspect success rates before interpreting')
        fig.tight_layout(); fig.savefig(out/'behavior.png',dpi=180); plt.close(fig)
    pd.DataFrame(effects).to_csv(out/'reward_effects.csv',index=False)
    # Per-run PPO curves (shaped training return is explicitly labelled).
    for f in root.glob('*/seed_*/progress.csv'):
        z=pd.read_csv(f)
        if 'rollout/ep_rew_mean' in z:
            fig,ax=plt.subplots(); ax.plot(z['time/total_timesteps'],z['rollout/ep_rew_mean'])
            ax.set(xlabel='Training transitions',ylabel='Shaped training return',title=str(f.parent.relative_to(root)))
            fig.tight_layout(); fig.savefig(out/(f.parent.parent.name+'_'+f.parent.name+'_learning.png')); plt.close(fig)
    print(f'Saved summaries to {out}. Missing matched bins are not imputed; inspect success rates.')
if __name__=='__main__': main()
