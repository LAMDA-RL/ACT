import torch
import wandb
import numpy as np
from tqdm import trange
import copy 
import os

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.module.critic import Critic, DoubleCritic
# from stable_baselines3.common.vec_env import SubprocVecEnv

from acsl.module.net.attention.rvs_transformer import RvSTransformer
# from acsl.policy.model_free import ACTOnlinePolicy
from acsl.policy.model_free.act_online import OnlineACTPolicy
from acsl.buffer.online_buffer import OnlineTrajectoryBuffer, create_dataloader
from acsl.policy.model_free.command import InSampleMaxCommand
from acsl.utils.eval import eval_sequence_rvs


class Experiment():
    def __init__(self, args):
        self.args = args
        exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
        self.logger = CompositeLogger(log_path=f"./log/sequence_rvs/online/{args.name}", name=exp_name, loggers_config={
            "FileLogger": {"activate": not args.debug}, 
            "TensorboardLogger": {"activate": not args.debug}, 
            # "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
        })
        setup(args, self.logger)
        self.product_root = f"./out/sequence_rvs/{args.name}/{args.task}/seed{args.seed}/"

        offline_dataset = args.task + "-medium-v2"
        self.env, offline_dataset = get_d4rl_dataset(offline_dataset, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
        obs_shape, action_shape = np.prod(self.env.observation_space.shape), np.prod(self.env.action_space.shape)
        target_entropy = -action_shape
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.episode_len = args.episode_len
        self.seq_len = args.seq_len

        critic_q = DoubleCritic(
            backend=torch.nn.Identity(), 
            input_dim=obs_shape + action_shape, 
            hidden_dims=args.hidden_dims, 
            reduce="min"
        ).to(args.device)

        critic_v = DoubleCritic(
            backend=torch.nn.Identity(), 
            input_dim=obs_shape, 
            hidden_dims=args.hidden_dims, 
            reduce="mean"
        ).to(args.device)

        actor_transformer = RvSTransformer(
            obs_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
            seq_len=args.seq_len, episode_len=args.episode_len, 
            embed_dropout=args.embed_dropout, attention_dropout=args.attention_dropout, 
            pos_encoding=args.pos_encoding
        ).to(args.device)
        command_backend = Critic(
            backend=torch.nn.Identity(), 
            input_dim=obs_shape, 
            hidden_dims=args.hidden_dims, 
            norm_layer=args.command_norm_layer
        ).to(args.device)
        command = InSampleMaxCommand(
            command_module=command_backend, 
            is_agent=True, 
            expectile=args.expectile, 
            enhance=args.enhance, 
            device=args.device
        ).to(args.device)
        command.configure_optimizers(args.command_lr)
        
        if args.add_noise:
            from functools import partial
            exploration_noise = partial(np.random.normal, loc=0, scale=args.noise_scale, size=[action_shape, ])
        else:
            exploration_noise = None

        policy = OnlineACTPolicy(
            critic_q=critic_q, 
            critic_v=critic_v, 
            command=command, 
            actor_transformer=actor_transformer, 
            state_dim=obs_shape, 
            action_dim=action_shape, 
            embed_dim=args.embed_dim, 
            seq_len=args.seq_len, 
            episode_len=args.episode_len, 
            action_type = args.action_type, 
            discount=args.discount, 
            lambda_=args.lambda_, 
            tau=args.tau, 
            exploration_noise=exploration_noise, 
            device=args.device
        ).to(args.device)
        policy.configure_optimizers(**args.optimizer_args)
        policy.train()
        
        self.policy = policy
        self.critic_q = critic_q
        self.critic_v = critic_v

        self.trajectory_buffer = OnlineTrajectoryBuffer(
            buffer_size=args.buffer_size, 
            state_dim=obs_shape, 
            action_dim=action_shape, 
            seq_len=args.seq_len, 
            episode_len=args.episode_len, 
            discount=args.discount, 
            return_scale=args.return_scale, 
            adv_scale=args.adv_scale, 
            lambda_=args.lambda_, 
            device=args.device
        )
        self.trajectory_buffer.add_offline_traj(offline_dataset)
        

    def train_epoch(self, batch_size):
        args = self.args
        dataloader = create_dataloader(self.trajectory_buffer, batch_size=batch_size)
        for i_step in range(args.critic_step_per_epoch):
            batch = next(dataloader)
            critic_metrics = self.policy.update_critic(batch)
        self.trajectory_buffer.relabel(self.critic_q, self.critic_v)
        dataloader = create_dataloader(self.trajectory_buffer, batch_size=args.batch_size)
        for i_step in range(args.critic_step_per_epoch):
            batch = next(dataloader)
            command_metrics = self.policy.update_command(batch)
        for i_step in range(args.act_step_per_epoch):
            batch = next(dataloader)
            act_metrics = self.policy.update_actor(batch, args.clip_grad)
        critic_metrics.update(command_metrics)
        critic_metrics.update(act_metrics)
        return critic_metrics

    def rollout(self, num_rollouts):
        # online_envs = [copy.deepcopy(self.env) for i in range(num_rollouts)]
        # for i, env in enumerate(online_envs):
            # env.seed(i+100)
        # online_envs = SubprocVecEnv(online_envs)
        online_envs = self.env
        self.policy.eval()
        
        all_observations = np.zeros([num_rollouts, self.episode_len+1, self.obs_shape], dtype=np.float32)
        all_actions = np.zeros([num_rollouts, self.episode_len, self.action_shape], dtype=np.float32)
        all_rewards = np.zeros([num_rollouts, self.episode_len, 1], dtype=np.float32)
        all_terminals = np.zeros([num_rollouts, self.episode_len, 1], dtype=np.float32)
        all_agent_advs = np.zeros([num_rollouts, self.episode_len, 1], dtype=np.float32)
        all_masks = np.zeros([num_rollouts, self.episode_len], dtype=np.float32)
        timesteps = np.tile(np.arange(self.episode_len, dtype=int)[None, :], (num_rollouts, 1))
        
        num_samples = 0
        mask = np.ones([num_rollouts, ], dtype=bool)
        state = online_envs.reset()
        all_observations[:, 0, :] = state
        all_masks[:, 0] = mask
        for step in range(self.episode_len):
            num_samples += mask.sum()
            command = self.policy.select_command(states=all_observations[:, step])
            all_agent_advs[:, step] = command
            action = self.policy.select_action(
                states=all_observations[:, :step+1][:, -self.seq_len: ], 
                actions=all_actions[:, :step+1][:, -self.seq_len: ], 
                agent_advs=all_agent_advs[:, :step+1][:, -self.seq_len: ], 
                timesteps=timesteps[:, :step+1][:, -self.seq_len: ], 
                deterministic=False
            )
            next_state, reward, done, _ = online_envs.step(action)
            mask = mask & (~done)
            all_observations[:, step+1, :] = next_state
            all_actions[:, step, :] = action
            all_rewards[:, step, :] = reward
            all_terminals[:, step, :] = done
            all_agent_advs[:, step, :] = command
            all_masks[:, step] = mask
            
            if not mask.any():
                break
        
        online_envs.close()
        return {
            "observations": all_observations[:, :-1], 
            "actions": all_actions, 
            "next_observations": all_observations[:, 1:], 
            "rewards": all_rewards, 
            "terminals": all_terminals, 
            "masks": all_masks
        }, num_samples, all_rewards.sum() / num_rollouts
        
    def train(self):
        args = self.args
        self.policy.train()
        if args.load_path is not None:
            self.policy.load_state_dict(torch.load(os.path.join(args.load_path, "pretrain/pretrain.pt"), map_location="cpu"))
            self.policy.to(args.device)
        else:
            for i_epoch in trange(1, args.pretrain_epoch+1, desc="offline"):
                train_metrics = self.train_epoch(args.batch_size)
                self.logger.log_scalars("pre-train", train_metrics, step=i_epoch)
                if i_epoch % args.eval_interval == 0:
                    eval_metrics = eval_sequence_rvs(self.env, self.policy, n_episode=args.eval_episode, seed=args.seed)
                    self.logger.log_scalars("pre-eval", eval_metrics, step=i_epoch)
            self.logger.log_object(name="pretrain.pt", object=self.policy.state_dict(), path=f"{self.product_root}/pretrain/")
            
        env_steps = 0
        for i_epoch in trange(1, args.online_epoch+1, desc="online"):
            new_trajectories, num_samples, episode_return = self.rollout(args.num_rollouts)
            self.trajectory_buffer.add_online_traj(new_trajectories)
            env_steps += num_samples
            
            train_metrics = self.train_epoch(args.batch_size)
            self.logger.log_scalars("train", train_metrics, step=i_epoch)
            self.logger.log_scalars("train", {
                "num_steps": num_samples, 
                "env_steps": env_steps
            }, step=i_epoch)
            self.logger.log_scalars("train", {
                "num_steps": num_samples,
                "env_steps": env_steps, 
                "episode_return": episode_return
            }, step=i_epoch)
            if i_epoch % args.eval_interval == 0:
                eval_metrics = eval_sequence_rvs(self.env, self.policy, n_episode=args.eval_episode, seed=args.seed)
                self.logger.log_scalars("eval", eval_metrics, step=i_epoch)
    

if __name__ == "__main__":
    args = parse_args()
    experiment = Experiment(args)
    experiment.train()