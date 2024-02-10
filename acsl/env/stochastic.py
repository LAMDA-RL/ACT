import numpy as np


def process_gambling_dataset(trajs):
    total_length = sum([len(t.obs) for t in trajs])
    state_dim = 7
    action_dim = 3

    observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    actions = np.zeros([total_length, action_dim], dtype=np.uint8)
    next_observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    rewards = np.zeros([total_length, ], dtype=np.float32)
    terminals = np.zeros([total_length,], dtype=np.bool8)

    cur = 0
    for t in trajs:
        t_len = len(t.obs)
        observations[cur:cur+t_len] = t.obs
        actions[np.arange(cur, cur+t_len), t.actions] = 1
        next_observations[cur:cur+t_len-1] = t.obs[1:]
        rewards[cur:cur+t_len] = t.rewards
        terminals[cur+t_len-1] = True
        cur += t_len
    return {
        "observations": observations, 
        "actions": actions, 
        "next_observations": next_observations, 
        "rewards": rewards, 
        "terminals": terminals, 
        "ends": terminals.copy()
    }

def process_2048_dataset(trajs):
    total_length = sum([len(t.obs) for t in trajs])
    state_dim = 4*4*8
    action_dim = 4

    observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    actions = np.zeros([total_length, action_dim], dtype=np.uint8)
    next_observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    rewards = np.zeros([total_length, ], dtype=np.float32)
    terminals = np.zeros([total_length,], dtype=np.bool8)

    cur = 0
    for t in trajs:
        t_len = len(t.obs)
        t_obs = [o.reshape(-1) for o in t.obs]
        t_actions = np.asarray(t.actions)
        observations[cur:cur+t_len] = t_obs
        actions[np.arange(cur, cur+t_len), t_actions] = 1
        next_observations[cur:cur+t_len-1] = t_obs[1:]
        rewards[cur:cur+t_len] = t.rewards
        terminals[cur+t_len-1] = True
        cur += t_len
    return {
        "observations": observations, 
        "actions": actions, 
        "next_observations": next_observations, 
        "rewards": rewards, 
        "terminals": terminals, 
        "ends": terminals.copy()
    }

def process_connect4_dataset(trajs):
    total_length = sum([len(t.obs) for t in trajs])
    state_dim = 2*7*6
    action_dim = 7

    observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    actions = np.zeros([total_length, action_dim], dtype=np.uint8)
    next_observations = np.zeros([total_length, state_dim], dtype=np.uint8)
    rewards = np.zeros([total_length, ], dtype=np.float32)
    terminals = np.zeros([total_length,], dtype=np.bool8)

    cur = 0
    for t in trajs:
        t_len = len(t.obs)
        t_obs = [o["grid"].reshape(-1) for o in t.obs]
        t_actions = np.asarray(t.actions)
        observations[cur:cur+t_len] = t_obs
        actions[np.arange(cur, cur+t_len), t_actions] = 1
        next_observations[cur:cur+t_len-1] = t_obs[1:]
        rewards[cur:cur+t_len] = t.rewards
        terminals[cur+t_len-1] = True
        cur += t_len
    return {
        "observations": observations, 
        "actions": actions, 
        "next_observations": next_observations, 
        "rewards": rewards, 
        "terminals": terminals, 
        "ends": terminals.copy()
    }



def get_stochastic_dataset(task):
    if task == "gambling-v0":
        from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
        gambling_task = GamblingOfflineEnv()
        env = gambling_task.env_cls()
        dataset = process_gambling_dataset(gambling_task.trajs)
    elif task == "2048-v0":
        from stochastic_offline_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
        tfe_task = TFEOfflineEnv()
        env = tfe_task.env_cls()
        dataset = process_2048_dataset(tfe_task.trajs)
    elif task == "connect4-v0":
        from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        c4_task = ConnectFourOfflineEnv()
        env = c4_task.env_cls()
        dataset = process_connect4_dataset(c4_task.trajs)
    elif task == "cliffwalking-v0":
        from acsl.env.cliff_walking import CliffWalkingEnv
        env = CliffWalkingEnv()
        dataset = process_cliffwalking_dataset()
    return env, dataset
