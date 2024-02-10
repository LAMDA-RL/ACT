from typing import Union, Optional
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import torch

from offlinerllib.buffer.base import Buffer
from offlinerllib.utils.functional import discounted_cum_sum
from offlinerllib.module.critic import Critic, DoubleCritic

from acsl.buffer.d4rl_buffer import D4RLTrajectoryBuffer
from acsl.utils.misc import compute_gae

class ACTTrajectoryBuffer(D4RLTrajectoryBuffer):
    def __init__(
        self, 
        dataset, 
        seq_len: int, 
        discount: float=1.0, 
        return_scale: float=1.0, 
        adv_scale: Optional[float]=None, 
        lambda_: Optional[float]=None, 
        use_mean_reduce: bool=False, 
        delayed_reward: bool=False, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__(
            dataset=dataset, 
            seq_len=seq_len, 
            discount=discount, 
            return_scale=return_scale
        )
        self.lambda_ = lambda_
        self.adv_scale = adv_scale
        self.device = device

        self.values = np.zeros_like(self.rewards)
        self.agent_advs = np.zeros_like(self.rewards)

        self.is_ssl_pretrain = False
        self.use_mean_reduce = use_mean_reduce
        self.delayed_reward = delayed_reward
        
        if self.delayed_reward:
            total_returns = np.sum(self.rewards, axis=1, keepdims=True)
            total_returns = total_returns / (total_returns.max()) * 50
            self.rewards = np.zeros_like(self.rewards)
            # self.returns = total_returns * self.masks[..., None]  # this may not reconcile with the discounted return
            self.rewards[np.arange(self.traj_num), self.traj_len - 1] = total_returns[:, 0]

    @torch.no_grad()
    def relabel(self, critic_q, critic_v):
        self.values = np.zeros_like(self.rewards)
        self.agent_advs = np.zeros_like(self.rewards)
        traj_start = 0
        with torch.no_grad():
            for i in range(self.traj_num):
                if self.lambda_ is None:
                    traj_len = self.traj_len[i]
                    obss = torch.from_numpy(self.observations[i, :traj_len]).to(self.device)
                    actions = torch.from_numpy(self.actions[i, :traj_len]).to(self.device)
                    if self.use_mean_reduce:
                        vs = critic_v(obss, reduce=False).mean(0).detach().cpu().numpy()
                        qs = critic_q(obss, actions, reduce=False).mean(0).detach().cpu().numpy()
                    else:
                        vs = critic_v(obss).detach().cpu().numpy()
                        qs = critic_q(obss, actions).detach().cpu().numpy()
                    self.values[i, :traj_len] = vs
                    self.agent_advs[i, :traj_len] = qs - vs
                else:
                    traj_len = self.traj_len[i]
                    rewards = self.rewards[i, :traj_len]
                    obss = torch.from_numpy(self.observations[i, :traj_len]).to(self.device)
                    last_obs = torch.from_numpy(self.next_observations[i, traj_len-1:traj_len]).to(self.device)
                    if self.use_mean_reduce:
                        vs = critic_v(obss, reduce=False).mean(0).detach().cpu().numpy()
                        last_v = critic_v(last_obs, reduce=False).mean(0).cpu().numpy()
                    else:
                        vs = critic_v(obss).detach().cpu().numpy()
                        last_v = critic_v(last_obs).cpu().numpy()
                    gae, _ = compute_gae(rewards, vs, last_v, gamma=self.discount, lam=self.lambda_, dim=0)
                    self.values[i, :traj_len] = vs
                    self.agent_advs[i, :traj_len] = gae
        if self.adv_scale is not None:
            if self.delayed_reward:
                max_adv = self.agent_advs.max()
                self.agent_advs = self.agent_advs / max_adv * self.adv_scale
            else:
                advs_ = self.agent_advs[self.agent_advs!=0]
                std_ = advs_.std()
                self.agent_advs = (self.agent_advs / std_) * self.adv_scale
            

    def __prepare_sample(self, traj_idx, end_idx):
        start_idx = end_idx
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[start_idx:start_idx+self.seq_len], 
            "values": self.values[traj_idx, start_idx:start_idx+self.seq_len], 
            "agent_advs": self.agent_advs[traj_idx, start_idx:start_idx+self.seq_len], 
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len]
        }
    
    def __iter__(self):
        while True:
            if not self.is_ssl_pretrain:
                traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
                start_idx = np.random.choice(self.traj_len[traj_idx])
                yield self.__prepare_sample(traj_idx, start_idx)
            else:
                traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
                start_idx = np.random.choice(self.traj_len[traj_idx])
                sample = self.__prepare_sample(traj_idx, start_idx)
                
                mask_p = np.random.uniform()
                input_mask = np.random.choice([0, 1], p=[mask_p, 1-mask_p], size=[self.seq_len, ])
                while (input_mask * sample["masks"]).sum() == 0:
                    input_mask = np.random.choice([0, 1], p=[mask_p, 1-mask_p], size=[self.seq_len, ])
                prediction_mask = 1 - input_mask
                sample.update({
                    "input_masks": input_mask, 
                    "prediction_masks": prediction_mask
                })
                yield sample
            
            