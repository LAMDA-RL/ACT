from operator import itemgetter
from typing import Any, Dict, Optional, Union, Callable, List

import torch
import torch.nn as nn
import numpy as np

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target

from acsl.module.policy_dist import SquashedDeterministicActorDistribution, ClippedGaussianActorDistribution, CategoricalActorDistribution
from acsl.utils.misc import compute_gae

class ACTGPTPolicy(BasePolicy):
    def __init__(
        self, 
        act, 
        critic_q: nn.Module, 
        critic_v: nn.Module, 
        command: nn.Module, 
        state_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        episode_len: int, 
        action_type: str="deterministic", 
        discount: float=0.99, 
        lambda_: Optional[float]=None, 
        tau: float=5e-3, 
        device: Union[str, torch.device] = "cpu", 
        **kwargs
    ) -> None:
        super().__init__()
        self.act = act
        self.critic_q = critic_q
        self.critic_v = critic_v
        self.critic_v_target = make_target(critic_v)
        self.command = command
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        
        self._discount = discount
        self._lambda = lambda_
        self._tau = tau

        self.action_type = action_type
        
        if self.action_type == "stochastic":
            self.action_head = ClippedGaussianActorDistribution(
                input_dim=embed_dim, 
                output_dim=action_dim, 
                reparameterize=False, 
                conditioned_logstd=False, 
            )
        elif self.action_type == "deterministic":
            self.action_head = SquashedDeterministicActorDistribution(
                input_dim=embed_dim, 
                output_dim=action_dim, 
            )
        elif self.action_type == "categorical": 
            self.action_head = CategoricalActorDistribution(
                input_dim=embed_dim, 
                output_dim=action_dim
            )

        self.to(device)
        
    def configure_optimizers(
        self, 
        critic_lr, 
        actor_lr, 
        critic_weight_decay, 
        actor_weight_decay, 
        critic_betas, 
        actor_betas, 
        critic_lr_scheduler_fn, 
        actor_lr_scheduler_fn
    ) -> None:
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_lr)
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=critic_lr)
        decay_param, nodecay_param = self.act.configure_params()
        self.act_optim = torch.optim.AdamW([
            {"params": [*decay_param, *self.action_head.parameters()], "weight_decay": actor_weight_decay}, 
            {"params": nodecay_param, "weight_decay": 0.0}
        ], lr=actor_lr, betas=actor_betas)
        self.act_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.act_optim, actor_lr_scheduler_fn)

    @torch.no_grad()
    def select_command(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        return self.command.select_command(states).detach().cpu().numpy()

    @torch.no_grad()
    def select_action(
        self, 
        states, 
        actions, 
        agent_advs, 
        timesteps, 
        deterministic=False, 
        *args, **kwargs
    ):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len: ].to(self.device)
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len: ].to(self.device)
        agent_advs = torch.from_numpy(agent_advs).float().reshape(1, -1, 1)[:, -self.seq_len: ].to(self.device)
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len: ].to(self.device)
        
        B, L, *_ = states.shape
        out = self.act(
            states=states, 
            actions=actions, 
            agent_advs=agent_advs, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=None
        )
        return self.action_head.sample(out[:, 1::3], deterministic=deterministic)[0][0, L-1].cpu().squeeze().numpy()
            
    def update_command(self, batch: Dict[str, Any]):
        loss_metrics = self.command.update(batch)
        return loss_metrics
        
    def update_critic(self, batch: Dict[str, Any], clip_grad=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        states, actions, rewards, next_states, terminals, masks = \
            [convert_to_tensor(v, self.device) for v in itemgetter("observations", "actions", "rewards", "next_observations", "terminals", "masks")(batch)]
        B, L, *_ = states.shape
        
        with torch.no_grad():
            target = self.critic_v_target(next_states)
            target = rewards + self._discount * (1-terminals) * target
        # v_pred = self.critic_v(states)
        # v_loss = ((target_v - v_pred) * masks.unsqueeze(-1)).pow(2).mean()
        # self.critic_v_optim.zero_grad()
        # v_loss.backward()
        # self.critic_v_optim.step()
        
        # for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
        #     o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

        if self._lambda is None:  # if lambda_ is None, then update q
            q_pred = self.critic_q(states, actions, reduce=False)
            q_loss = ((target - q_pred) * masks.unsqueeze(-1)).pow(2).sum(0).mean()
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()
        
        # with torch.no_grad():
            # target_q = rewards + self._discount * (1-terminals) * self.critic_v_target(next_states)
        
        # if self._lambda is None:
        # q_pred = self.critic_q(states, actions, reduce=False)
        # q_loss = ((target_q - q_pred) * masks.unsqueeze(-1)).pow(2).sum(0).mean()
        # self.critic_q_optim.zero_grad()
        # q_loss.backward()
        # self.critic_q_optim.step()

        # for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
        #     o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau) 

        # with torch.no_grad():
            # target_v = rewards + self._discount * (1-terminals) * self.critic_v_target(next_states)
        v_pred = self.critic_v(states, reduce=False)
        v_loss = ((target - v_pred) * masks.unsqueeze(-1)).pow(2).sum(0).mean()
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau) 
        
        return {
            "v_loss": v_loss.item(), 
            "v_pred": v_pred.mean().item(), 
            "q_loss": q_loss.item() if self._lambda is None else 0, 
            "q_pred": q_pred.mean().item() if self._lambda is None else 0
        }

    @torch.no_grad()
    def compute_agent_adv(self, states, actions, rewards=None):
        if self._lambda is None:
            vs = self.critic_v(torch.from_numpy(states).to(self.device))
            qs = self.critic_q(torch.from_numpy(states).to(self.device), torch.from_numpy(actions).to(self.critic_v.device))
            return (qs - vs).detach().cpu().numpy()
        else:
            # assert False, "not debugged for compute agent advantages"
            assert rewards is not None
            vs = self.critic_v(torch.from_numpy(states).to(self.device)).detach().cpu().numpy()
            agent_advs, returns = compute_gae(rewards, vs[..., :-1, :], last_v=vs[..., -1:, :], gamma=self._discount, lam=self._lambda, dim=-2)
            return agent_advs
        
    @torch.no_grad()
    def select_action(self, states, actions, agent_advs, timesteps, deterministic=False, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len: ].to(self.device)
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len: ].to(self.device)
        agent_advs = torch.from_numpy(agent_advs).float().reshape(1, -1, 1)[:, -self.seq_len: ].to(self.device)
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len: ].to(self.device)
        
        B, L, *_ = states.shape
        out = self.act(
            states=states, 
            actions=actions, 
            agent_advs=agent_advs, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=None
        )
        return self.action_head.sample(out[:, 1::3], deterministic=deterministic)[0][0, L-1].cpu().squeeze().numpy()
        
    def act_loss(self, batch: Dict[str, Any]):
        states, actions, agent_advs, timesteps, masks = \
            itemgetter("observations", "actions", "agent_advs", "timesteps", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        out = self.act(
            states=states, 
            actions=actions, 
            agent_advs=agent_advs, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=key_padding_mask
        )
        
        if not isinstance(self.action_head, SquashedDeterministicActorDistribution):
            action_loss = - self.action_head.evaluate(out[:, 1::3], actions.detach())[0]
        else:
            action_loss = torch.nn.functional.mse_loss(
                self.action_head.sample(out[:, 1::3])[0], 
                actions.detach(), 
                reduction="none"
            )
        action_loss = (action_loss * masks.unsqueeze(-1)).mean()
        
        total_loss = action_loss
        return total_loss, {
            "loss/reinforce_action": action_loss.item(), 
        }
        
    def update(
        self, 
        batch: Dict[str, Any],
        clip_grad: Optional[float]=None
    ):
        metrics = dict()
        
        # compute reinforce term
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        reinforce_loss, reinforce_metrics = self.act_loss(batch)
        metrics.update(reinforce_metrics)
        
        # backward ACT losses
        final_act_loss = reinforce_loss
        self.act_optim.zero_grad()
        final_act_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), clip_grad)
        self.act_optim.step()
        
        self.act_optim_scheduler.step()
        
        metrics.update({
            "loss/final_act_loss": final_act_loss.item(), 
            "misc/act_lr": self.act_optim_scheduler.get_last_lr()[0], 
        })
        
        return metrics
        