from gym.envs.registration import register
from d4rl.utils.wrappers import NormalizedBoxEnv, ProxyEnv
from d4rl.gym_mujoco.gym_envs import OfflineWalker2dEnv, OfflineHalfCheetahEnv, OfflineHopperEnv


class DelayedEnv(ProxyEnv):
    def __init__(self, env):
        ProxyEnv.__init__(self, env)
        self.step_counter = 0
        self.reward_counter = 0
        
    def step(self, action):
        self.step_counter += 1
        next_state, reward, done, info = super().step(action)
        self.reward_counter += reward
        if done:
            return next_state, self.reward_counter, done, info
        else:
            return next_state, 0, done, info
        
    def reset(self, **kwargs):
        self.step_counter = 0
        self.reward_counter = 0
        return super().reset(**kwargs)
    

def get_delayed_walker2d(**kwargs):
    return NormalizedBoxEnv(DelayedEnv(OfflineWalker2dEnv(**kwargs)))

def get_delayed_hopper(**kwargs):
    return NormalizedBoxEnv(DelayedEnv(OfflineHopperEnv(**kwargs)))

def get_delayed_halfcheetah(**kwargs):
    return NormalizedBoxEnv(DelayedEnv(OfflineWalker2dEnv(**kwargs)))

REF_MIN_SCORE = {
    'walker2d-stochastic-0.35-v3': 1.8695256663349098,
    'hopper-stochastic-0.2-v3': 18.613661584802880,
    'halfcheetah-stochastic-0.35-v3': -284.93601542829816,
}

REF_MAX_SCORE = {
    'walker2d-stochastic-0.35-v3': 4349.11057456666,
    'hopper-stochastic-0.2-v3': 2008.6098363010358,
    'halfcheetah-stochastic-0.35-v3': 6103.9156506577065,
}
