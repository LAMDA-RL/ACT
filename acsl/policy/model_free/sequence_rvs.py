from operator import itemgetter
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor, make_target
from offlinerllib.utils.functional import expectile_regression
from offlinerllib.module.net.mlp import MLP

from acsl.module.policy_dist import SquashedDeterministicActorDistribution, ClippedGaussianActorDistribution, CategoricalActorDistribution
from acsl.module.net.attention.rvs_transformer import RvSTransformer

class SequenceRvS(BasePolicy):
    def __init__(
        self, 
        critic_q: nn.Module, 
        critic_v: nn.Module, 
        command, 
        actor_transformer: nn.Module, 
        state_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        episode_len: int, 
        action_type: str="deterministic", 
        discount: float=0.99, 
        lambda_: Optional[float]=None, 
        tau: float=5e-3, 
        v_target: float=True, 
        iql_tau: float=0.7, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__()
        self.v_target = v_target
        if self.v_target:
            self.critic_q = critic_q
            self.critic_v = critic_v
            self.critic_v_target = make_target(critic_v)
        else: 
            self.critic_q = critic_q
            self.critic_v = critic_v
            self.critic_q_target = make_target(critic_q)
        self.command = command
        self.actor_transformer = actor_transformer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        
        self._discount = discount
        self._lambda = lambda_
        self._tau = tau
        self._iql_tau = iql_tau
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
        self, critic_lr, actor_lr, actor_weight_decay, actor_betas, actor_lr_scheduler_fn, decay_embedding=False
    ) -> None:
        self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=critic_lr)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=critic_lr)

        actor_decay_param, actor_nodecay_param = self.actor_transformer.configure_params()
        if decay_embedding:
            weight_decay_embedding = actor_weight_decay
        else:
            weight_decay_embedding = 0.0
        self.actor_optim = torch.optim.AdamW([
            {"params": [*actor_decay_param, *self.action_head.parameters()], "weight_decay": actor_weight_decay}, 
            {"params": actor_nodecay_param, "weight_decay": weight_decay_embedding}
        ], lr=actor_lr, betas=actor_betas)

        self.actor_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optim, actor_lr_scheduler_fn)

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
        out = self.actor_transformer(
            states=states, 
            actions=actions, 
            timesteps=timesteps, 
            agent_advs=agent_advs, 
            key_padding_mask=None
        )
        return self.action_head.sample(out, deterministic=deterministic)[0][0, L-1].cpu().squeeze().numpy()

    def update_command(self, batch: Dict):
        loss_metrics = self.command.update(batch)
        return loss_metrics
        
    def update_critic(self, batch: Dict[str, Any], clip_grad=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        states, actions, rewards, next_states, terminals, masks = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks"
        )(batch)

        if self.v_target:
            with torch.no_grad():
                target = self.critic_v_target(next_states)
                target = rewards + self._discount * (1-terminals) * target
            q_pred = self.critic_q(states, actions, reduce=False)
            q_loss = ((target - q_pred) * masks.unsqueeze(-1)
                      ).pow(2).sum(0).mean()
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()

            v_pred = self.critic_v(states, reduce=False)
            v_loss = expectile_regression(
                v_pred, target, expectile=self._iql_tau)
            v_loss = (v_loss * masks.unsqueeze(-1)).sum(0).mean()
            self.critic_v_optim.zero_grad()
            v_loss.backward()
            self.critic_v_optim.step()
        else:
            with torch.no_grad():
                q_target = self.critic_v(next_states, reduce=False).mean(0)
                q_target = rewards + self._discount * (1-terminals) * q_target
                q_old = self.critic_q_target(states, actions)
            q_pred = self.critic_q(states, actions, reduce=False)
            q_loss = ((q_target - q_pred) *
                      masks.unsqueeze(-1)).pow(2).sum(0).mean()
            self.critic_q_optim.zero_grad()
            q_loss.backward()
            self.critic_q_optim.step()

            v_pred = self.critic_v(states, reduce=False)
            v_loss = expectile_regression(
                v_pred, q_old, expectile=self._iql_tau)
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
            
    def update_actor(self, batch: Dict[str, Any], clip_grad=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        states, actions, timesteps, agent_advs, masks = \
            itemgetter("observations", "actions", "timesteps", "agent_advs", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        out = self.actor_transformer(
            states=states, 
            actions=actions, 
            timesteps=timesteps, 
            agent_advs=agent_advs, 
            key_padding_mask=key_padding_mask
        )
        if not isinstance(self.action_head, SquashedDeterministicActorDistribution):
            action_loss = - self.action_head.evaluate(out, actions.detach(), is_onehot_action=True)[0]
        else:
            action_loss = torch.nn.functional.mse_loss(
                self.action_head.sample(out)[0], 
                actions.detach(), 
                reduction="none"
            )
        actor_loss = (action_loss * masks.unsqueeze(-1)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_([*self.actor_transformer.parameters(), *self.action_head.parameters()], clip_grad)
        self.actor_optim.step()
        self.actor_optim_scheduler.step()
        return {
            "actor_loss": actor_loss.item()
        }

    def _sync_target(self):
        if self.v_target:
            for o, n in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        else:
            for o, n in zip(self.critic_q_target.parameters(), self.critic_q.parameters()):
                o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)