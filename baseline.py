import argparse,subprocess,sys
from pathlib import Path
from env import ReachEnv,configuration,controller
from rollout import collect

def main():
    p=argparse.ArgumentParser(); p.add_argument('--check',action='store_true')
    p.add_argument('--episodes',type=int,default=100); p.add_argument('--out',default='runs')
    p.add_argument('--config',default='config.json'); p.add_argument('--gains',nargs='+',type=float,default=[10,25,50,100,200,400])
    a=p.parse_args()
    if a.check: subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
    for kind in ['external','completion']:
        for kp in a.gains:
            collect(ReachEnv(configuration(a.config),kind,record=True),lambda obs,e:controller(e,kp),
                    Path(a.out)/'baseline'/kind/f'kp_{kp:g}',a.episodes)
if __name__=='__main__': main()
