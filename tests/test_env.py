import numpy as np
import pytest
from env import ReachEnv,controller

def test_gym_contract():
    from stable_baselines3.common.env_checker import check_env
    check_env(ReachEnv(),warn=True)

def test_fixed_condition_and_geometry():
    e=ReachEnv(); a,_=e.reset(seed=123,options={'schedule':'A','q':1}); targets=e.targets.copy()
    e.reset(seed=123,options={'schedule':'B','q':10})
    np.testing.assert_equal(e.targets,targets)
    for _ in range(100):
        o,*_=e.step([0,0]); assert o[-1]==1 and o[-2]==1

def test_B_twelve_and_no_late_reward():
    e=ReachEnv(record=True); e.reset(seed=1,options={'schedule':'B','q':10})
    e.tick=499; e.x=e.g.copy(); e.v[:]=0; e.dwell=14
    _,r,_,_,i=e.step([0,0]); assert i['points']==0 and e.target_id==1
    while not e.done: e.step([0,0])
    assert e.target_id+1==12
    assert e.trials[0]['reason']=='expired'

def test_dwell_and_A_replacement():
    e=ReachEnv(); e.reset(seed=1,options={'schedule':'A','q':1}); e.x=e.g.copy()
    for _ in range(14): assert e.step([0,0])[-1]['points']==0
    assert e.step([0,0])[-1]['points']==1
    assert e.target_id==1 and e.dwell==0 and e.tick==15

def test_B_success_total_and_waiting():
    e=ReachEnv(); e.reset(seed=1,options={'schedule':'B','q':10})
    while not e.done:
        if e.active: e.x=e.g.copy(); e.v[:]=0
        _,_,_,_,info=e.step([0,0])
    assert e.tick==6000 and e.successes==12 and e.points==120
    assert info['block']['opportunities']==12
    with pytest.raises(RuntimeError): e.step([0,0])

def test_terminal_priority():
    e=ReachEnv(); e.reset(seed=1,options={'schedule':'A','q':1})
    e.tick=5999; e.onset=5990; e.x=e.g.copy(); e.dwell=14
    _,_,done,trunc,i=e.step([0,0]); assert done and not trunc and i['points']==0

def test_shaping_telescopes_and_reward_decomposition():
    e=ReachEnv(shaping=True); e.reset(seed=7,options={'schedule':'B','q':1})
    phi=e.potential(); added=raw=ret=0.
    while not e.done:
        _,r,_,_,i=e.step(controller(e)); added+=i['shaping']; raw+=i['raw_reward']; ret+=r
    assert np.isclose(added,-phi)
    assert np.isclose(ret-raw,-phi)
    assert np.isclose(raw,e.points+e.bonus-e.effort)

def test_constant_bonus():
    e=ReachEnv({'kappa':0,'beta0':2},model='completion'); e.reset(seed=1,options={'schedule':'B','q':1})
    while not e.done: e.step(controller(e))
    assert np.isclose(e.bonus,2*e.successes)

def test_metric_timing():
    e=ReachEnv(record=True); e.reset(seed=4,options={'schedule':'A','q':1})
    while not e.done: e.step(controller(e))
    for t in e.trials:
        if t['success']:
            assert t['mt']>0
            assert np.isclose(t['acquisition_time']-t['entry'],.15)
            assert np.isclose(t['acquisition_time']-t['target_onset'],t['reaction_time']+t['mt']+.15)
