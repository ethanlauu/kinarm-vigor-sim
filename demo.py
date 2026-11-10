"""Watch one block using a PD controller or a saved PPO model."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from env import ReachEnv,configuration,controller

def main():
    p=argparse.ArgumentParser(); p.add_argument('--schedule',choices=['A','B'],default='B')
    p.add_argument('--q',type=int,choices=[1,10],default=10); p.add_argument('--model')
    p.add_argument('--seed',type=int,default=100000); p.add_argument('--config',default='config.json')
    a=p.parse_args(); agent=None
    if a.model:
        import json
        from pathlib import Path
        from stable_baselines3 import PPO
        agent=PPO.load(a.model,device='cpu')
        meta=json.loads((Path(a.model).parent/'config.json').read_text()); cfg=meta['config']
    else: cfg=configuration(a.config)
    env=ReachEnv(cfg); obs,_=env.reset(seed=a.seed,options=dict(schedule=a.schedule,q=a.q))
    fig,ax=plt.subplots(); ax.set(xlim=(-.22,.22),ylim=(-.22,.22),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    target=Circle(env.g,cfg['radius'],color='teal'); ax.add_patch(target)
    cursor,=ax.plot([],[],'ko'); title=ax.set_title('')
    def frame(_):
        nonlocal obs
        for _ in range(3):
            if env.done: break
            u=agent.predict(obs,deterministic=True)[0] if agent else controller(env)
            obs,*_=env.step(u)
        target.center=env.g; target.set_visible(env.active)
        cursor.set_data([env.x[0]],[env.x[1]])
        title.set_text(f'{a.schedule}-{a.q} | {env.tick*cfg["dt"]:.1f}s | points={env.points:.0f}')
        return cursor,target,title
    animation=FuncAnimation(fig,frame,frames=range((env.K+2)//3),interval=30,repeat=False,cache_frame_data=False)
    plt.show(); env.close()
if __name__=='__main__': main()
