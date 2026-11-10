import argparse,json
from pathlib import Path
from stable_baselines3 import PPO
from env import ReachEnv
from rollout import collect

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs')
    p.add_argument('--episodes',type=int,default=100); p.add_argument('--seed-start',type=int,default=100000)
    p.add_argument('--save-steps',action='store_true'); a=p.parse_args()
    paths=list(Path(a.root).glob('*/seed_*/model.zip'))
    if not paths: p.error('No trained model.zip files under root')
    for path in paths:
        meta=json.loads((path.parent/'config.json').read_text())
        model=PPO.load(path,device='cpu')
        env=ReachEnv(meta['config'],meta['model'],record=True)
        collect(env,lambda obs,e:model.predict(obs,deterministic=True)[0],path.parent/'evaluation',
                a.episodes,a.seed_start,a.save_steps)
if __name__=='__main__': main()
