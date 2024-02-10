from copy import deepcopy
from operator import itemgetter
from typing import Dict, Tuple, Union
import os

import numpy as np
import torch
import torch.nn as nn

from offlinerllib.utils.functional import expectile_regression
from offlinerllib.utils.misc import convert_to_tensor, make_target


class IQL(nn.Module):
    def __init__(
        self,
        critic_q,
        critic_v,
        v_target=True,
        expectile: float = 0.7,
        tau: float = 5e-3,
        discount: float = 0.99,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.critic_q = critic_q
        self.critic_v = critic_v
        self.v_target = v_target
        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.device = device

        if self.v_target:
            self.critic_v_target = make_target(critic_v)
        else:
            self.critic_q_target = make_target(critic_q)

    def configure_optimizers(self, critic_lr):
        self.critic_q_optim = torch.optim.Adam(
            self.critic_q.parameters(), lr=critic_lr)
        self.critic_v_optim = torch.optim.Adam(
            self.critic_v.parameters(), lr=critic_lr)

    def update(self, batch):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        states, actions, rewards, next_states, terminals, masks = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks"
        )(batch)

        if self.v_target:
            with torch.no_grad():
                target = self.critic_v_target(next_states)
                target = rewards + self.discount * (1-terminals) * target
            q_pred = self.critic_q(states, actions, reduce=False)
            q_loss = ((target - q_pred) * masks.unsqueeze(-1)
                      ).pow(2).sum(0).mean()
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()

            v_pred = self.critic_v(states, reduce=False)
            v_loss = expectile_regression(
                v_pred, target, expectile=self.expectile)
            v_loss = (v_loss * masks.unsqueeze(-1)).sum(0).mean()
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
        else:
            with torch.no_grad():
                q_target = self.critic_v(next_states, reduce=False).mean(0)
                q_target = rewards + self.discount * (1-terminals) * q_target
                q_old = self.critic_q_target(states, actions)
            q_pred = self.critic_q(states, actions, reduce=False)
            q_loss = ((q_target - q_pred) *
                      masks.unsqueeze(-1)).pow(2).sum(0).mean()
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()

            v_pred = self.critic_v(states, reduce=False)
            v_loss = expectile_regression(
                v_pred, q_old, expectile=self.expectile)
            v_loss = (v_loss * masks.unsqueeze(-1)).sum(0).mean()
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
            
        self._sync_target()
        return {
            "q_loss": q_loss.item(), 
            "v_loss": v_loss.item(), 
            "q_pred": q_pred.mean().item(), 
            "v_pred": v_pred.mean().item()
        }

    def _sync_target(self):
        if self.v_target:
            for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
                o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)
        else:
            for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
                o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)

    @torch.no_grad()
    def evaluate(self, traj_buffer):
        mse_loss_v_sum = 0
        mse_loss_q_sum = 0
        exp_loss_v_sum = 0
        sample_sum = 0
        for i in range(traj_buffer.traj_num):
            obss = traj_buffer.observations[i:i+1, :, :]
            actions = traj_buffer.actions[i:i+1, :, :]
            next_obss = traj_buffer.next_observations[i:i+1, :, :]
            rewards = traj_buffer.rewards[i:i+1, :, :]
            terminals = traj_buffer.terminals[i:i+1, :, :]
            masks = traj_buffer.masks[i:i+1, :]

            obss = convert_to_tensor(obss, self.device)
            actions = convert_to_tensor(actions, self.device)
            next_obss = convert_to_tensor(next_obss, self.device)
            rewards = convert_to_tensor(rewards, self.device)
            terminals = convert_to_tensor(terminals, self.device)
            masks = convert_to_tensor(masks, self.device)
            
            td_target = rewards + self.discount * (1-terminals) * self.critic_v(next_obss, reduce=False).mean(0)
            td_v = td_target - self.critic_v(obss, reduce=False).mean(0)
            td_q = td_target - self.critic_q(obss, actions, reduce=False).mean(0)
            
            td_loss_mse_v = 0.5*((td_v) * masks.unsqueeze(-1)).pow(2).sum()
            td_loss_mse_q = 0.5*((td_q) * masks.unsqueeze(-1)).pow(2).sum()
            td_loss_exp_v = ((torch.where(td_v >= 0, self.expectile, 1-self.expectile) * td_v.pow(2)) * masks.unsqueeze(-1)).sum()
            
            # if self.v_target:
            #     td_target_by_vt = rewards + self.discount * \
            #     (1-terminals) * self.critic_v_target(next_obss, reduce=False).mean(0)
            #     td_loss_vt = ((td_target_by_vt - self.critic_v(obss, reduce=False).mean(0)).pow(2) * masks.unsqueeze(-1)).sum()
            #     loss_vt_sum += td_loss_vt.item()
            mse_loss_v_sum += td_loss_mse_v.item()
            mse_loss_q_sum += td_loss_mse_q.item()
            exp_loss_v_sum += td_loss_exp_v.item()
            sample_sum += masks.float().sum()
        # return mse_loss_v_sum / sample_sum, mse_loss_q_sum / sample_sum, exp_loss_v_sum / sample_sum
        return exp_loss_v_sum / sample_sum
    
    def save(self, root):
        if not os.path.exists(root):
            os.makedirs(root)
        torch.save(self.critic_q.state_dict(), os.path.join(root, "critic_q.pt"))
        torch.save(self.critic_v.state_dict(), os.path.join(root, "critic_v.pt"))
        if self.v_target:
            torch.save(self.critic_v_target.state_dict(), os.path.join(root, "critic_v_target.pt"))
        else:
            torch.save(self.critic_q_target.state_dict(), os.path.join(root, "critic_q_target.pt"))
    
    def load(self, root):
        self.critic_q.load_state_dict(torch.load(os.path.join(root, "critic_q.pt"), map_location="cpu"))
        self.critic_v.load_state_dict(torch.load(os.path.join(root, "critic_v.pt"), map_location="cpu"))
        if self.v_target:
            self.critic_v_target.load_state_dict(torch.load(os.path.join(root, "critic_v_target.pt"), map_location="cpu"))
        else:
            self.critic_q_target.load_state_dict(torch.load(os.path.join(root, "critic_q_target.pt"), map_location="cpu"))
            