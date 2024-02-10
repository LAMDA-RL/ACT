from operator import itemgetter
from typing import Any, Dict, Union, Tuple

import torch
import torch.nn as nn
import numpy as np

from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor
from offlinerllib.utils.functional import expectile_regression

class InSampleMaxCommand(BasePolicy):
    def __init__(
        self, 
        command_module: nn.Module, 
        is_agent: bool=True, 
        expectile: float=0.95, 
        enhance: bool=False, 
        device: Union[str, torch.device]="cpu"
    ) -> None:
        super().__init__()
        self.command = command_module
        self._expectile = expectile
        self._enhance = enhance
        self._is_agent = is_agent
        self.id = "agent_ismax" if self._is_agent else "model_ismax"
        self.to(device)
    
    @torch.no_grad()
    def select_command(self, states, *args, **kwargs):
        out = self.command(states)
        if self._enhance:
            out = out.clip(min=0)
            out += out.abs() * 0.20
        return out
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if self._is_agent:
            obss, agent_advs, masks = [convert_to_tensor(v, self.device) for v in itemgetter("observations", "agent_advs", "masks")(batch)]
            command_pred = self.command(obss, reduce=False)
            loss = expectile_regression(command_pred, agent_advs, expectile=self._expectile)
            loss = (loss * masks.unsqueeze(-1)).mean()
        else:
            obss, actions, model_advs = [convert_to_tensor(v, self.device) for v in itemgetter("observations", "actions", "model_advs")(batch)]
            command_pred = self.command(obss, actions)
            loss = expectile_regression(command_pred, model_advs, expectile=1-self._expectile).mean()
        
        self.command_agent_optim.zero_grad()
        loss.backward()
        self.command_agent_optim.step()
        return {
            f"{self.id}_ISM_loss": loss.item(), 
            f"{self.id}_ISM_value": command_pred.mean().item(), 
            f"{self.id}_ISM_suboptimality": (agent_advs - command_pred).clip(min=0).mean().item(),
            f"{self.id}_ISM_overestimate": (command_pred - agent_advs).clip(min=0).mean().item(), 
            f"{self.id}_ISM_gap": (agent_advs - command_pred).mean().item()
        }
    
    def configure_optimizers(self, lr, command_weight_decay=0.0):
        self.command_agent_optim = torch.optim.Adam(self.command.parameters(), lr=lr, weight_decay=command_weight_decay)
        
        
class ConstantCommand(BasePolicy):
    def __init__(
        self, init=0, polyak=0.995, device: Union[str, torch.device]="cpu", *args, **kwargs
    ) -> None:
        super().__init__()
        self.polyak = polyak
        self.register_buffer("constant", torch.tensor([init, ], dtype=torch.float32))
        self.to(device)
        
    @torch.no_grad()
    def select_command(self, states, *args, **kwargs):
        shape = list(states.shape)
        shape[-1] = 1
        return (torch.ones(shape).to(states.device) * self.constant)
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        agent_advs = convert_to_tensor(batch["agent_advs"], self.device)
        new_constant = agent_advs[agent_advs >= 0].mean()
        self.constant = self.polyak * self.constant + (1-self.polyak) * new_constant
        return {
            "command_value": self.constant.item(), 
        }

    def set_value(self, value):
        self.constant.data = torch.tensor(value).to(self.device)
        