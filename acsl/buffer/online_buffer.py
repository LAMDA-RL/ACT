from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from acsl.utils.misc import compute_gae

def create_dataloader(
    buffer, 
    batch_size: int, 
    num_workers=4, 
):
    return iter(DataLoader(buffer, batch_size=batch_size, num_workers=num_workers))
    

class OnlineTrajectoryBuffer(IterableDataset):
    def __init__(
        self, 
        buffer_size: int, 
        state_dim: int, 
        action_dim: int, 
        seq_len: int, 
        episode_len: int, 
        discount: float=0.99, 
        return_scale: float=1.0, 
        adv_scale: Optional[float]=None, 
        lambda_: Optional[float]=None, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.episode_len = episode_len
        self.discount = discount
        self.return_scale = return_scale
        self.adv_scale = adv_scale
        self.lambda_ = lambda_
        self.device = device
        
        max_seq_len = episode_len+seq_len
        self.data = {
            "observations": np.zeros([buffer_size, max_seq_len, state_dim], dtype=np.float32), 
            "actions": np.zeros([buffer_size, max_seq_len, action_dim], dtype=np.float32), 
            "next_observations": np.zeros([buffer_size, max_seq_len, state_dim], dtype=np.float32), 
            "rewards": np.zeros([buffer_size, max_seq_len, 1], dtype=np.float32), 
            "terminals": np.zeros([buffer_size, max_seq_len, 1], dtype=np.float32), 
            "masks": np.zeros([buffer_size, max_seq_len], dtype=np.float32), 
            "values": np.zeros([buffer_size, max_seq_len, 1], dtype=np.float32), 
            "agent_advs": np.zeros([buffer_size, max_seq_len, 1], dtype=np.float32), 
        }
        self.cur_traj = 0
        self.traj_num = 0
        self.sample_num = 0
        self.traj_len = np.zeros([buffer_size, ], dtype=np.int32)
        self.sample_prob = np.zeros([buffer_size, ])
        
    def add_offline_traj(self, dataset): 
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                this_traj_len = i+1-traj_start
                for k, v in converted_dataset.items():
                    self.data[k][self.cur_traj, :this_traj_len] = v[traj_start:traj_start+this_traj_len]
                self.data["masks"][self.cur_traj, :this_traj_len] = 1
                self.traj_num = min(self.traj_num + 1, self.buffer_size)
                self.sample_num = self.sample_num - self.traj_len[self.cur_traj] + this_traj_len
                self.traj_len[self.cur_traj] = this_traj_len
                self.cur_traj = (self.cur_traj + 1) % self.buffer_size
                traj_start = i+1
        self.sample_prob = self.traj_len / self.sample_num
        
    def add_online_traj(self, online_trajs):
        num, traj_len = online_trajs["observations"].shape[0], online_trajs["observations"].shape[1]
        index_to_go = (self.cur_traj + np.arange(num)) % self.buffer_size
        old_num_sample = self.traj_len[index_to_go].sum()
        self.traj_len[index_to_go] = online_trajs["masks"].sum(-1)
        new_num_sample = self.traj_len[index_to_go].sum()
        self.sample_num = self.sample_num - old_num_sample + new_num_sample
        self.cur_traj = (self.cur_traj+num) % self.buffer_size
        self.traj_num = min(self.traj_num+num, self.buffer_size)
        self.data["masks"][index_to_go] = 0
        for k, v in online_trajs.items():
            self.data[k][index_to_go, :traj_len, ...] = v
        self.sample_prob = self.traj_len / self.sample_num

    def __prepare_sample(self, traj_idx, start_idx):
        # note that we use a different __prepare_sample for ACT
        # in ACTBuffer we look behind samples rather than look ahead
        # start_idx = end_idx - self.seq_len + 1
        ret = {
            k: v[traj_idx, start_idx:start_idx+self.seq_len] for k, v in self.data.items()
        }
        ret["timesteps"] = np.arange(start_idx, start_idx+self.seq_len)
        return ret
        
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.buffer_size, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.__prepare_sample(traj_idx, start_idx)
        
    @torch.no_grad()
    def relabel(self, critic_q, critic_v):
        for i in range(self.traj_num):
            if self.lambda_ is None:
                traj_len = self.traj_len[i]
                obss = torch.from_numpy(self.observations[i, :traj_len]).to(self.device)
                actions = torch.from_numpy(self.actions[i, :traj_len]).to(self.device)
                if obss.shape[0] == 2:
                    obss = torch.tile(obss, (2, 1))
                    actions = torch.tile(actions, (2, 1))
                    vs = critic_v(obss).detach().cpu().numpy()[:2, ...]
                    qs = critic_q(obss, actions).detach().cpu().numpy()[:2, ...]
                else:
                    vs = critic_v(obss).detach().cpu().numpy()
                    qs = critic_q(obss, actions).detach().cpu().numpy()
                self.data["values"][i, :traj_len] = vs
                self.data["agent_advs"][i, :traj_len] = qs - vs
            else:
                traj_len = self.traj_len[i]
                rewards = self.data["rewards"][i, :traj_len]
                obss = torch.from_numpy(self.data["observations"][i, :traj_len]).to(self.device)
                last_obs = torch.from_numpy(self.data["next_observations"][i, traj_len-1:traj_len]).to(self.device)
                if obss.shape[0] == 2:
                    obss = torch.tile(obss, (2, 1))
                    vs = critic_v(obss).detach().cpu().numpy()[:2, ...]
                else:
                    vs = critic_v(obss).detach().cpu().numpy()
                last_v = critic_v(last_obs).cpu().numpy()
                gae, _ = compute_gae(rewards, vs, last_v, gamma=self.discount, lam=self.lambda_, dim=0)
                self.data["values"][i, :traj_len] = vs
                self.data["agent_advs"][i, :traj_len] = gae
        if self.adv_scale is not None:
            advs_ = self.data["agent_advs"][self.data["agent_advs"]!=0]
            std_ = advs_.std()
            self.data["agent_advs"] = (self.data["agent_advs"] / std_) * self.adv_scale