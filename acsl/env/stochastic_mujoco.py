from gym.envs.registration import register
from d4rl.utils.wrappers import NormalizedBoxEnv, ProxyEnv
from gym.envs.mujoco import SwimmerEnv, HumanoidEnv
from d4rl.gym_mujoco.gym_envs import OfflineWalker2dEnv, OfflineHalfCheetahEnv, OfflineHopperEnv, OfflineAntEnv
from d4rl import offline_env
import numpy as np

class OfflineSwimmerEnv(SwimmerEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        SwimmerEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHumanoidEnv(HumanoidEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HumanoidEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class StochasticEnv(ProxyEnv):
    def __init__(self, env, sigma):
        ProxyEnv.__init__(self, env)
        self.step_counter=0
        self.sigma=sigma

    def step(self, action):
        noise_std = np.abs((1-np.exp(-0.01*self.step_counter))*np.sin(self.step_counter)*self.sigma)
        self.step_counter+=1
        a=action+np.random.normal(0,noise_std,action.shape)
        a=np.clip(a,-1,1)
        return super().step(a)

    def reset(self, **kwargs):
        self.step_counter=0
        return super().reset(**kwargs)

def get_stochastic_hopper(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineHopperEnv(**kwargs), kwargs['noise_level']))

def get_stochastic_halfcheetah(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineHalfCheetahEnv(**kwargs), kwargs['noise_level']))

def get_stochastic_ant(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineAntEnv(**kwargs), kwargs['noise_level']))

def get_stochastic_walker2d(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineWalker2dEnv(**kwargs), kwargs['noise_level']))

def get_stochastic_swimmer(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineSwimmerEnv(**kwargs), kwargs['noise_level']))

def get_stochastic_humanoid(**kwargs):
    return NormalizedBoxEnv(StochasticEnv(OfflineHumanoidEnv(**kwargs), kwargs['noise_level']))

# REF_MIN_SCORE and REF_MAX_SCORE
# walker2d, hopper, halfcheetah, ant are copied from https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/infos.py
# swimmer and humanoid are not included in D4RL, so we write them ourselves
REF_MIN_SCORE = {
    'walker2d-stochastic-v3': 1.629008,
    'hopper-stochastic-v3': -20.272305,
    'halfcheetah-stochastic-v3': -280.178953,
    'ant-stochastic-v3': -325.6,
    'swimmer-stochastic-v3': 0.0,
    'humanoid-stochastic-v3': 0.0
}

REF_MAX_SCORE = {
    'walker2d-stochastic-v3': 4592.3,
    'hopper-stochastic-v3': 3234.3,
    'halfcheetah-stochastic-v3': 12135.0,
    'ant-stochastic-v3': 3879.7,
    'swimmer-stochastic-v3': 100.0,
    'humanoid-stochastic-v3': 6000.0
}

stochastic_mujoco_envs=[]

for agent in [
    ('hopper', 0.1), ('halfcheetah', 0.1), ('walker2d', 0.1), ('ant', 0.1), ('swimmer', 0.1), ('humanoid', 0.1),
    ('hopper', 0.2), ('halfcheetah', 0.2), ('walker2d', 0.2), ('ant', 0.2), ('swimmer', 0.2), ('humanoid', 0.2),
    ('hopper', 0.3), ('halfcheetah', 0.3), ('walker2d', 0.3), ('ant', 0.3), ('swimmer', 0.3), ('humanoid', 0.3),
]:
    for dataset in ['medium']:
        stochastic_mujoco_envs.append(f'{agent[0]}-stochastic-{agent[1]}-{dataset}-v3')
        register(
            id=f'{agent[0]}-stochastic-{agent[1]}-{dataset}-v3',
            entry_point=f'acsl.env.stochastic_mujoco:get_stochastic_{agent[0]}',
            max_episode_steps=1000,
            kwargs={
                'deprecated': False,
                'ref_min_score': REF_MIN_SCORE[f'{agent[0]}-stochastic-v3'],
                'ref_max_score': REF_MAX_SCORE[f'{agent[0]}-stochastic-v3'],
                'dataset_url': f'https://datasets.caomingjun.com/offline_rl/stochastic_mujoco/{agent[0]}_{str(agent[1])}_{dataset}.hdf5',
                'noise_level': agent[1]
            }
        )

for agent in [
    ('hopper', 0.1), ('halfcheetah', 0.1), ('walker2d', 0.1), ('ant', 0.1), ('swimmer', 0.1), ('humanoid', 0.1),
    ('hopper', 0.2), ('halfcheetah', 0.2), ('walker2d', 0.2), ('ant', 0.2), ('swimmer', 0.2), ('humanoid', 0.2),
    ('hopper', 0.15), ('halfcheetah', 0.15), ('walker2d', 0.15), ('ant', 0.15), ('swimmer', 0.15), ('humanoid', 0.15),
]:
    for dataset in ['replay']:
        stochastic_mujoco_envs.append(f'{agent[0]}-stochastic-{agent[1]}-{dataset}-v3')
        register(
            id=f'{agent[0]}-stochastic-{agent[1]}-{dataset}-v3',
            entry_point=f'acsl.env.stochastic_mujoco:get_stochastic_{agent[0]}',
            max_episode_steps=1000,
            kwargs={
                'deprecated': False,
                'ref_min_score': REF_MIN_SCORE[f'{agent[0]}-stochastic-v3'],
                'ref_max_score': REF_MAX_SCORE[f'{agent[0]}-stochastic-v3'],
                'dataset_url': f'https://datasets.caomingjun.com/offline_rl/stochastic_mujoco/{agent[0]}_{str(agent[1])}_{dataset}.hdf5',
                'noise_level': agent[1]
            }
        )
