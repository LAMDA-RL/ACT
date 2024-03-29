from operator import itemgetter
from typing import Any, Dict, Optional, Union

import torch

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor
from acsl.module.net.attention.decision_transformer import DecisionTransformer
from acsl.module.policy_dist import SquashedDeterministicActorDistribution, ClippedGaussianActorDistribution, CategoricalActorDistribution


class DecisionTransformerPolicy(BasePolicy):
    """
    Decision Transformer: Reinforcement Learning via Sequence Modeling <Ref: https://arxiv.org/abs/2106.01345>
    """
    def __init__(
        self, 
        dt: DecisionTransformer, 
        state_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        seq_len: int, 
        episode_len: int, 
        action_type: str="deterministic", 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len
        
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

    def configure_optimizers(self, lr, weight_decay, betas, warmup_steps):
        decay, no_decay = self.dt.configure_params()
        self.dt_optim = torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay}, 
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=lr, betas=betas)
        self.dt_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dt_optim, lambda step: min((step+1)/warmup_steps, 1))

    @torch.no_grad()
    def select_action(self, states, actions, returns_to_go, timesteps, deterministic=False, **kwargs):
        states = torch.from_numpy(states).float().reshape(1, -1, self.state_dim)[:, -self.seq_len:]
        actions = torch.from_numpy(actions).float().reshape(1, -1, self.action_dim)[:, -self.seq_len:]
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, -1, 1)[:, -self.seq_len:]
        timesteps = torch.from_numpy(timesteps).reshape(1, -1)[:, -self.seq_len:]
        
        B, L, *_ = states.shape
        if self.seq_len > L:
            states = torch.cat([states, torch.zeros(B, self.seq_len-L, self.state_dim)], dim=1)
            actions = torch.cat([actions, torch.zeros(B, self.seq_len-L, self.action_dim)], dim=1)
            returns_to_go = torch.cat([returns_to_go, torch.zeros(B, self.seq_len-L, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.zeros(B, self.seq_len-L, dtype=torch.int64)], dim=1)
            key_padding_mask = torch.cat([torch.zeros(B, L).bool(), torch.ones(B, self.seq_len-L).bool()], dim=1)
        else:
            key_padding_mask = torch.zeros(B, L).bool()
        states, actions, returns_to_go, timesteps, key_padding_mask = \
            states.to(self.device), actions.to(self.device), returns_to_go.to(self.device), timesteps.to(self.device), key_padding_mask.to(self.device)
        out = self.dt(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            attention_mask=None, 
            key_padding_mask=key_padding_mask
        )
        return self.action_head.sample(out, deterministic=deterministic)[0][0, L-1].cpu().squeeze().numpy()
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, returns_to_go, timesteps, masks = \
            itemgetter("observations", "actions", "returns", "timesteps", "masks")(batch)
        key_padding_mask = ~masks.to(torch.bool)
        
        out = self.dt(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            attention_mask=None,    # DT is causal and will handle causal masks itself
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
        action_loss = (action_loss * masks.unsqueeze(-1)).mean()
        self.dt_optim.zero_grad()
        action_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.dt.parameters(), clip_grad)
        self.dt_optim.step()
        self.dt_optim_scheduler.step()
        
        return {
            "loss/action_loss": action_loss.item(), 
            "misc/learning_rate": self.dt_optim_scheduler.get_last_lr()[0]
        }
        
        