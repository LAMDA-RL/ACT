from typing import Callable, Dict, List

import gym
import numpy as np
import torch
import torch.nn as nn

from acsl.policy.model_free.sequence_rvs import SequenceRvS


@torch.no_grad()
def eval_sequence_rvs(
    env, 
    policy: SequenceRvS, 
    n_episode: int, 
    seed, 
    score_func=None
):
    if score_func is None:
        score_func = env.get_normalized_score
    try:
        env.reset(seed=seed)
    except: 
        pass  # some custom envs may not support seeding 
    policy.eval()
    episode_lengths = []
    episode_returns = []
    timesteps = np.arange(policy.episode_len, dtype=int)[None, :]
    for _ in range(n_episode):
        states = np.zeros([1, policy.episode_len+1, policy.state_dim], dtype=np.float32)
        actions = np.zeros([1, policy.episode_len+1, policy.action_dim], dtype=np.float32)
        rewards = np.zeros([1, policy.episode_len+1, 1], dtype=np.float32)
        agent_advs = np.zeros([1, policy.episode_len+1, 1], dtype=np.float32)
        state, done = env.reset(), False
        
        states[:, 0] = state.reshape(-1)
        
        episode_return = episode_length = 0
        for step in range(policy.episode_len):
            command = policy.select_command(states=states[:, step])
            agent_advs[:, step] = command
            action = policy.select_action(
                states=states[:, :step+1][:, -policy.seq_len: ], 
                actions=actions[:, :step+1][:, -policy.seq_len: ], 
                agent_advs=agent_advs[:, :step+1][:, -policy.seq_len: ], 
                timesteps=timesteps[:, :step+1][:, -policy.seq_len: ], 
                deterministic=False
            )
            next_state, reward, done, info = env.step(action)
            if len(action.shape) == 0:
                # discrete control
                actions[:, step, action] = 1
            else:
                # continuous control
                actions[:, step] = action
            rewards[:, step] = reward
            states[:, step+1] = next_state.reshape(-1)

            episode_return += reward
            episode_length += 1
            if done:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                break 
           
    policy.train()
    episode_returns = np.asarray(episode_returns)
    episode_lengths = np.asarray(episode_lengths)
    return {"normalized_score_mean": score_func(episode_returns.mean()) * 100} 
    
@torch.no_grad()
def eval_act_gpt(
    relabel, 
    env: gym.Env, 
    adt: nn.Module,
    n_episode: int, 
    seed: int, 
    score_func=None,  
):
    if score_func is None:
        score_func = env.get_normalized_score
    env.reset(seed=seed)
    adt.eval()
    episode_lengths = []
    episode_returns = []
    timesteps = np.arange(adt.episode_len, dtype=int)[None, :]
    for _ in range(n_episode):
        states = np.zeros([1, adt.episode_len+1, adt.state_dim], dtype=np.float32)
        actions = np.zeros([1, adt.episode_len+1, adt.action_dim], dtype=np.float32)
        rewards = np.zeros([1, adt.episode_len+1, 1], dtype=np.float32)
        agent_advs = np.zeros([1, adt.episode_len+1, 1], dtype=np.float32)
        model_advs = np.zeros([1, adt.episode_len+1, 1], dtype=np.float32)
        state, done = env.reset(), False
        
        states[:, 0] = state
        
        episode_return = episode_length = 0
        for step in range(adt.episode_len):
            command = adt.select_command(states=states[:, step])
            agent_advs[:, step] = command
            action = adt.select_action(
                states=states[:, :step+1][:, -adt.seq_len: ], 
                actions=actions[:, :step+1][:, -adt.seq_len: ],
                agent_advs=agent_advs[:, :step+1][:, -adt.seq_len: ], 
                model_advs=model_advs[:, :step+1][:, -adt.seq_len: ], 
                timesteps=timesteps[:, :step+1][:, -adt.seq_len: ], 
                deterministic = True
            )
            next_state, reward, done, info = env.step(action)

            if len(action.shape) == 0:
                # discrete control
                actions[:, step, action] = 1
            else:
                # continuous control
                actions[:, step] = action
            rewards[:, step] = reward
            states[:, step+1] = next_state
            
            # lets update the past agent advantages
            if relabel:
                update_advantage = adt.compute_agent_adv(states[:, :step+2], actions[:, :step+2], rewards[:, :step+1])
                agent_advs[:, :step+1] = update_advantage

            episode_return += reward
            episode_length += 1
            if done:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                break

    adt.train()
    episode_returns = np.asarray(episode_returns)
    episode_lengths = np.asarray(episode_lengths)
    return {"normalized_score_mean": score_func(episode_returns.mean()) * 100}


@torch.no_grad()
def eval_decision_transformer(
    env: gym.Env, actor: nn.Module, target_returns: List[float], return_scale: float, n_episode: int, delayed_reward: bool=False, seed: int=0, score_func=None
) -> Dict[str, float]:
    
    def eval_one_return(target_return, score_func=None):
        if score_func is None:
            score_func = env.get_normalized_score
        env.seed(seed)
        actor.eval()
        episode_lengths = []
        episode_returns = []
        timesteps = np.arange(actor.episode_len, dtype=int)[None, :]
        for _ in range(n_episode):
            states = np.zeros([1, actor.episode_len+1, actor.state_dim], dtype=np.float32)
            actions = np.zeros([1, actor.episode_len+1, actor.action_dim], dtype=np.float32)
            returns_to_go = np.zeros([1, actor.episode_len+1, 1], dtype=np.float32)
            state, done = env.reset(), False
            
            states[:, 0] = state.reshape(-1)
            returns_to_go[:, 0] = target_return*return_scale
            
            episode_return = episode_length = 0
            for step in range(actor.episode_len):
                action = actor.select_action(
                    states[:, :step+1][:, -actor.seq_len: ], 
                    actions[:, :step+1][:, -actor.seq_len: ], 
                    returns_to_go[:, :step+1][:, -actor.seq_len: ], 
                    timesteps[:, :step+1][:, -actor.seq_len: ]
                )
                if len(action.shape) == 0:
                    # discrete control
                    actions[:, step, action] = 1
                else:
                    # continuous control
                    actions[:, step] = action
                next_state, reward, done, info = env.step(action)
                actions[:, step] = action
                states[:, step+1] = next_state.reshape(-1)
                if delayed_reward:
                    returns_to_go[:, step+1] = returns_to_go[:, step]
                else:
                    returns_to_go[:, step+1] = returns_to_go[:, step] - reward*return_scale
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    episode_returns.append(score_func(episode_return)*100)
                    episode_lengths.append(episode_length)
                    break
                
        actor.train()
        episode_returns = np.asarray(episode_returns)
        episode_lengths = np.asarray(episode_lengths)
        return {
            "normalized_score_mean_target{:.1f}".format(target_return): episode_returns.mean(), 
            "normalized_score_std_target{:.1f}".format(target_return): episode_returns.std(), 
            "length_mean_target{:.1f}".format(target_return): episode_lengths.mean(), 
            "length_std_target{:.1f}".format(target_return): episode_lengths.std()
        }
    
    ret = {}
    for target in target_returns:
        ret.update(eval_one_return(target, score_func))
    return ret

    
@torch.no_grad()
def dump_dt_actions(
    env: gym.Env, actor: nn.Module, target: float, return_scale: float, n_episode: int, seed: int
) -> Dict[str, float]:
    
    def eval_one_return(target_return):
        env.seed(seed)
        actor.eval()
        timesteps = np.arange(actor.episode_len, dtype=int)[None, :]
        all_obss = []
        all_actions = []
        for _ in range(n_episode):
            states = np.zeros([1, actor.episode_len+1, actor.state_dim], dtype=np.float32)
            actions = np.zeros([1, actor.episode_len+1, actor.action_dim], dtype=np.float32)
            returns_to_go = np.zeros([1, actor.episode_len+1, 1], dtype=np.float32)
            state, done = env.reset(), False
            
            states[:, 0] = state.reshape(-1)
            returns_to_go[:, 0] = target_return*return_scale
            
            episode_return = episode_length = 0
            for step in range(actor.episode_len):
                action = actor.select_action(
                    states[:, :step+1][:, -actor.seq_len: ], 
                    actions[:, :step+1][:, -actor.seq_len: ], 
                    returns_to_go[:, :step+1][:, -actor.seq_len: ], 
                    timesteps[:, :step+1][:, -actor.seq_len: ]
                )
                actions[:, step, action] = 1
                next_state, reward, done, info = env.step(action)
                states[:, step+1] = next_state.reshape(-1)
                returns_to_go[:, step+1] = returns_to_go[:, step] - reward*return_scale
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    all_obss.append(states)
                    all_actions.append(actions)
                    break
                
        actor.train()
        return all_obss, all_actions
    return eval_one_return(target)

    
@torch.no_grad()
def dump_sequence_rvs_actions(
    env, 
    policy: SequenceRvS, 
    n_episode: int, 
    seed=0, 
):
    try:
        env.reset(seed=seed)
    except: 
        pass  # some custom envs may not support seeding 
    policy.eval()
    timesteps = np.arange(policy.episode_len, dtype=int)[None, :]
    all_obss = []
    all_actions = []
    for _ in range(n_episode):
        states = np.zeros([1, policy.episode_len+1, policy.state_dim], dtype=np.float32)
        actions = np.zeros([1, policy.episode_len+1, policy.action_dim], dtype=np.float32)
        rewards = np.zeros([1, policy.episode_len+1, 1], dtype=np.float32)
        agent_advs = np.zeros([1, policy.episode_len+1, 1], dtype=np.float32)
        state, done = env.reset(), False
        
        states[:, 0] = state.reshape(-1)
        
        episode_return = episode_length = 0
        for step in range(policy.episode_len):
            command = policy.select_command(states=states[:, step])
            agent_advs[:, step] = command
            action = policy.select_action(
                states=states[:, :step+1][:, -policy.seq_len: ], 
                actions=actions[:, :step+1][:, -policy.seq_len: ], 
                agent_advs=agent_advs[:, :step+1][:, -policy.seq_len: ], 
                timesteps=timesteps[:, :step+1][:, -policy.seq_len: ], 
                deterministic=False
            )
            next_state, reward, done, info = env.step(action)
            if len(action.shape) == 0:
                # discrete control
                actions[:, step, action] = 1
            else:
                # continuous control
                actions[:, step] = action
            rewards[:, step] = reward
            states[:, step+1] = next_state.reshape(-1)

            episode_return += reward
            episode_length += 1
            if done:
                all_obss.append(states)
                all_actions.append(actions)
                break 
           
    policy.train()
    return all_obss, all_actions