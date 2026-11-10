"""PPO fitting; each policy sees four conditions, one condition per episode."""
import argparse,json,subprocess,sys
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from env import ReachEnv,configuration

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['external','completion'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--steps',type=int,default=2_000_000)
    p.add_argument('--out',default='runs'); p.add_argument('--config',default='config.json')
    p.add_argument('--smoke',action='store_true',help='Short execution check, NOT scientific training')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args(); c=configuration(a.config); torch.set_num_threads(a.threads)
    for seed in a.seeds:
        folder=Path(a.out)/a.model/f'seed_{seed}'
        if folder.exists(): raise FileExistsError(f'{folder} exists; choose a fresh --out')
        folder.mkdir(parents=True)
        nenv=2 if a.smoke else 8
        vec=DummyVecEnv([lambda i=i: Monitor(ReachEnv(c,a.model,shaping=True),
                          str(folder/f'env_{i}.monitor.csv')) for i in range(nenv)])
        vec.seed(seed)
        settings=dict(learning_rate=3e-4,n_steps=128 if a.smoke else 6000,
          batch_size=64 if a.smoke else 500,n_epochs=2 if a.smoke else 10,
          gamma=1.,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5)
        metadata=dict(config=c,model=a.model,seed=seed,smoke=a.smoke,ppo=settings,n_envs=nenv)
        (folder/'config.json').write_text(json.dumps(metadata,indent=2))
        (folder/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        model=PPO('MlpPolicy',vec,device='cpu',seed=seed,verbose=1,**settings,
            policy_kwargs=dict(activation_fn=torch.nn.Tanh,net_arch=dict(pi=[64,64],vf=[64,64])))
        model.set_logger(configure(str(folder),['stdout','csv']))
        callback=CheckpointCallback(save_freq=max(1,240000//nenv),save_path=str(folder/'checkpoints'))
        try:
            model.learn(total_timesteps=a.steps,callback=callback)
            model.save(str(folder/'model'))
            metadata['actual_steps']=model.num_timesteps
            (folder/'config.json').write_text(json.dumps(metadata,indent=2))
        finally: vec.close()
if __name__=='__main__': main()
