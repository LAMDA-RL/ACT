from torch.utils.data import IterableDataset
import numpy as np
from offlinerllib.utils.functional import discounted_cum_sum
import torch
from offlinerllib.module.critic import Critic, DoubleCritic

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

class ToyBuffer(IterableDataset):
    def __init__(
        self, 
        dataset, 
        seq_len, 
        discount: float=1.0, 
        return_scale: float=1.0, 
        adv_scale=None, 
        lambda_=None, 
        use_mean_reduce=None, 
        device="cpu",
        *args,  
        **kwargs
    ) -> None:
        super().__init__()
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj, traj_len = [], []
        self.seq_len = seq_len
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) / return_scale
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
        self.traj = np.array(traj, dtype=object)
        self.traj_len = np.array(traj_len)
        self.traj_num = len(self.traj_len)
        self.size = self.traj_len.sum()
        self.sample_prob = self.traj_len / self.size
        del converted_dataset
        
        
        # other attrs for act
        self.lambda_ = lambda_
        self.adv_scale = adv_scale
        self.device = device
        
        self.use_mean_reduce = use_mean_reduce
        
    
    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.traj[traj_idx]
        sample = {k: v[start_idx:start_idx+self.seq_len] for k, v in traj.items()}
        sample_len = len(sample["observations"])
        if sample_len < self.seq_len:
            sample = {k: pad_along_axis(v, pad_to=self.seq_len) for k, v in sample.items()}
        masks = np.hstack([np.ones(sample_len), np.zeros(self.seq_len-sample_len)])
        sample["masks"] = masks
        sample["timesteps"] = np.arange(start_idx, start_idx+self.seq_len)
        return sample
    
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.__prepare_sample(traj_idx, start_idx)
        
    @torch.no_grad()    
    def relabel(self, critic_q, critic_v):
        assert self.lambda_ is None
        for i in range(len(self.traj)):
            traj_len = self.traj_len[i]
            obss = torch.from_numpy(self.traj[i]["observations"]).to(self.device)
            actions = torch.from_numpy(self.traj[i]["actions"]).to(self.device)
            if isinstance(critic_v, DoubleCritic) and obss.shape[0] == critic_v.critic_num:
                critic_num = critic_v.critic_num
                obss = torch.tile(obss, (2, 1))
                actions = torch.tile(actions, (2, 1))
                vs = critic_v(obss, reduce=False).mean(0).detach().cpu().numpy()[:critic_num, ...]
                qs = critic_q(obss, actions, reduce=False).mean(0).detach().cpu().numpy()[:critic_num, ...]
            else:
                vs = critic_v(obss, reduce=False).mean(0).detach().cpu().numpy()
                qs = critic_q(obss, actions, reduce=False).mean(0).detach().cpu().numpy()
            agent_advs = (qs - vs).copy()
            self.traj[i]["agent_advs"]  = agent_advs