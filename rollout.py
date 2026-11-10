"""Shared evaluation; export bounded-memory per-block step traces."""
from pathlib import Path
import numpy as np
import pandas as pd
from env import STEP_COLUMNS

def collect(env,policy,folder,episodes=100,seed_start=100000,save_steps=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    blocks=[]; trials=[]
    for schedule in ['A','B']:
        for q in [1,10]:
            for n in range(episodes):
                seed=seed_start+n; block_id=f'{schedule}_{q}_{seed}'
                obs,_=env.reset(seed=seed,options=dict(schedule=schedule,q=q))
                while True:
                    obs,_,done,_,info=env.step(policy(obs,env))
                    if done: break
                blocks.append(dict(block_id=block_id,env_seed=seed,**info['block']))
                trials.extend(dict(block_id=block_id,env_seed=seed,**t) for t in env.trials)
                if save_steps:
                    d=folder/'steps'; d.mkdir(exist_ok=True)
                    # Unicode array loads with allow_pickle=False; columns specify types.
                    np.savez_compressed(d/f'{block_id}.npz',columns=np.array(STEP_COLUMNS),
                        data=np.asarray(env.rows,dtype=str),schedule=schedule,q=q)
            print(f'{folder}: {schedule}-{q}: {episodes} blocks',flush=True)
    pd.DataFrame(blocks).to_csv(folder/'blocks.csv',index=False)
    pd.DataFrame(trials).to_csv(folder/'trials.csv',index=False)
    env.close()
