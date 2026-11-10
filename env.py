"""Finite-horizon reaching task. Physical units: metres, seconds, newtons."""
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

DEFAULT = dict(dt=.01, duration=60., interval=5., timeout=10., dwell=.15,
               radius=.01, mass=1., damping=10., force=5., c=1.,
               beta0=1., beta1=0., kappa=1.)

def configuration(path=None):
    c=DEFAULT.copy()
    if path: c.update(json.loads(Path(path).read_text()))
    unknown=set(c)-set(DEFAULT)
    if unknown: raise ValueError(f'Unknown settings: {unknown}')
    for k in ('dt','duration','interval','timeout','dwell','radius','mass','force'):
        if c[k]<=0: raise ValueError(f'{k} must be positive')
    for k in ('c','damping','beta0','beta1','kappa'):
        if c[k]<0: raise ValueError(f'{k} must be nonnegative')
    for k in ('duration','interval','timeout','dwell'):
        if not np.isclose(c[k]/c['dt'],round(c[k]/c['dt'])):
            raise ValueError(f'{k} must be an integer number of ticks')
    return c

class ReachEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, config=None, model='external', shaping=False, record=False):
        super().__init__()
        self.cfg=configuration()
        if config: self.cfg.update(config)
        if model not in ('external','completion'): raise ValueError(model)
        self.model,self.shaping,self.record=model,shaping,record
        self.action_space=spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(13,),dtype=np.float32)
        self.K=round(self.cfg['duration']/self.cfg['dt'])
        self.B=round(self.cfg['interval']/self.cfg['dt'])
        self.A=round(self.cfg['timeout']/self.cfg['dt'])
        self.H=round(self.cfg['dwell']/self.cfg['dt'])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o=options or {}
        # Separate target RNG: paired evaluation seeds yield identical geometry,
        # regardless of forced condition or number of acquisitions.
        target_seed=int(self.np_random.integers(0,2**32-1))
        rng=np.random.default_rng(target_seed)
        self.schedule=o.get('schedule',self.np_random.choice(['A','B']))
        self.q=int(o.get('q',self.np_random.choice([1,10])))
        if self.schedule not in ('A','B') or self.q not in (1,10): raise ValueError(o)
        angle=rng.uniform(0,2*np.pi)
        targets=[.12*np.array([np.cos(angle),np.sin(angle)])]
        for _ in range(max(500,self.K//self.H+2)-1):
            while True:
                d=rng.choice([.08,.12,.16]); a=rng.uniform(0,2*np.pi)
                g=targets[-1]+d*np.array([np.cos(a),np.sin(a)])
                if np.max(np.abs(g))<=.18: break
            targets.append(g)
        self.targets=np.array(targets)
        self.x=np.zeros(2); self.v=np.zeros(2); self.tick=0; self.target_id=-1
        self.points=self.bonus=self.effort=self.wait_distance=0.
        self.successes=0; self.rows=[]; self.trials=[]; self.motion=[]
        self.done=False; self._spawn()
        return self._obs(), {}

    def _spawn(self):
        self.target_id+=1; self.g=self.targets[self.target_id].copy()
        self.onset=self.tick; self.active=True; self.dwell=0
        self.start_distance=float(np.linalg.norm(self.g-self.x))
        self.motion=[(self.tick,float(np.linalg.norm(self.v)))] if self.record else []

    def _finish(self, reason):
        if not self.record: return
        onset=None
        for j in range(max(0,len(self.motion)-2)):
            if all(v>.005 for _,v in self.motion[j:j+3]):
                onset=self.motion[j][0]*self.cfg['dt']; break
        entry=(self.tick-self.H)*self.cfg['dt'] if reason=='success' else np.nan
        mt=entry-onset if onset is not None and np.isfinite(entry) else np.nan
        if np.isfinite(mt) and mt<=0: mt=np.nan
        peak=max((v for k,v in self.motion if onset is not None and
                  k*self.cfg['dt']>=onset and k*self.cfg['dt']<=entry),default=np.nan)
        self.trials.append(dict(target_id=self.target_id,schedule=self.schedule,q=self.q,
            target_onset=self.onset*self.cfg['dt'],distance=self.start_distance,
            movement_onset=onset,entry=entry,end_time=self.tick*self.cfg['dt'],
            acquisition_time=self.tick*self.cfg['dt'] if reason=='success' else np.nan,
            reaction_time=onset-self.onset*self.cfg['dt'] if onset is not None else np.nan,
            mt=mt,peak_speed=peak,success=int(reason=='success'),reason=reason))

    def _obs(self):
        c=self.cfg; t=self.tick*c['dt']
        w=(self.B-self.tick%self.B)*c['dt'] if self.schedule=='B' and not self.done else 0.
        return np.array([* (self.x/.2),*(self.v/.5),*(self.g/.2),float(self.active),
            self.dwell/c['dwell']*c['dt'],(self.tick-self.onset)*c['dt']/10,
            w/5,(c['duration']-t)/60,self.q/10,float(self.schedule=='B')],dtype=np.float32)

    def potential(self):
        return 0. if self.done or not self.active else -float(np.linalg.norm(self.g-self.x))/.12

    def step(self, action):
        if self.done: raise RuntimeError('Reset after termination')
        c=self.cfg; dt=c['dt']; u=np.asarray(action,dtype=float)
        if u.shape!=(2,) or not np.isfinite(u).all(): raise ValueError('Action must be finite shape (2,)')
        u=np.clip(u,-1,1); phi=self.potential()
        old=self.x.copy(); active=self.active; target_id=self.target_id; target=self.g.copy()
        self.v+=(c['force']*u-c['damping']*self.v)/c['mass']*dt
        self.x+=self.v*dt; self.tick+=1
        effort=c['c']*float(u@u)*dt; points=bonus=0.; event='move' if active else 'wait'
        self.effort+=effort
        if not active: self.wait_distance+=float(np.linalg.norm(self.x-old))
        if active and self.record: self.motion.append((self.tick,float(np.linalg.norm(self.v))))
        if self.tick==self.K:
            if active: self._finish('block_end')
            self.done=True; self.active=False; self.dwell=0; event='block_end'
        elif self.schedule=='B' and self.tick%self.B==0:
            if active: self._finish('expired')
            self._spawn(); event='scheduled_onset'
        elif active and self.schedule=='A' and self.tick-self.onset>=self.A:
            self._finish('expired'); self._spawn(); event='timeout'
        elif active:
            inside=np.linalg.norm(old-target)<=c['radius'] and np.linalg.norm(self.x-target)<=c['radius']
            self.dwell=self.dwell+1 if inside else 0
            if self.dwell>=self.H:
                D=(self.tick-self.onset)*dt; points=float(self.q)
                if self.model=='completion': bonus=(c['beta0']+c['beta1']*self.q)*np.exp(-c['kappa']*D)
                self.points+=points; self.bonus+=bonus; self.successes+=1
                self._finish('success'); event='success'
                if self.schedule=='A': self._spawn()
                else: self.active=False; self.dwell=0
        raw=points+bonus-effort; shaping=self.potential()-phi
        info=dict(points=points,completion_bonus=float(bonus),effort_cost=effort,
                  raw_reward=raw,shaping=shaping,event=event)
        if self.record:
            self.rows.append([self.tick,target_id,*old,*self.x,*self.v,*u,*target,
                              int(active),self.dwell,points,bonus,effort,shaping,event])
        if self.done:
            info['block']=dict(schedule=self.schedule,q=self.q,points=self.points,
                completion_bonus=self.bonus,effort_cost=self.effort,utility=self.points+self.bonus-self.effort,
                successes=self.successes,opportunities=self.target_id+1,
                success_rate=self.successes/(self.target_id+1),waiting_distance=self.wait_distance)
        return self._obs(),raw+(shaping if self.shaping else 0.),self.done,False,info

STEP_COLUMNS=['tick','target_id','old_x','old_y','x','y','vx','vy','ux','uy','gx','gy',
              'active_before','dwell_after','points','completion_bonus','effort_cost','shaping','event']

def controller(env,kp=100.):
    if not env.active: return np.zeros(2)
    c=env.cfg
    return np.clip((kp*(env.g-env.x)-2*np.sqrt(c['mass']*kp)*env.v)/c['force'],-1,1)
