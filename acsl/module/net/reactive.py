from typing import List
from torch import nn
import torch
from offlinerllib.module.net.mlp import MLP

class PureMLP(nn.Module):
    def __init__(
        self,
        state_dim:int,
        expanded_dim:int,
        action_dim:int,
        hidden_dims:List[int],
        norm_layer,
        activation:nn.Module=nn.ReLU,
        dropout:float=0.1
    ):
        super().__init__()
        self.adv_net = MLP(
            input_dim=1,
            output_dim=expanded_dim,
            hidden_dims=[],
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )
        self.net = MLP(
            input_dim=state_dim+expanded_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )

    def forward(
        self,
        states:torch.Tensor,
        agent_advs:torch.Tensor
    ):
        expended = self.adv_net(agent_advs)
        return self.net(torch.cat([states, expended], dim=-1))

class GatedMLP(nn.Module):
    def __init__(
        self,
        state_dim:int,
        state_hidden_dims:List[int],
        adv_hidden_dims:List[int],
        gate_dim:int,
        action_hidden_dims:List[int],
        action_dim:int,
        norm_layer,
        activation:nn.Module=nn.ReLU,
        dropout:float=0.1
    ):
        super().__init__()
        self.state_mlp = MLP(
            input_dim=state_dim,
            hidden_dims=state_hidden_dims,
            output_dim=gate_dim,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )
        self.adv_mlp = MLP(
            input_dim=1,
            hidden_dims=adv_hidden_dims,
            output_dim=gate_dim,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )
        self.action_mlp = MLP(
            input_dim=gate_dim,
            hidden_dims=action_hidden_dims,
            output_dim=action_dim,
            activation=activation,
            dropout=dropout,
            norm_layer=norm_layer
        )

    def forward(
        self,
        states:torch.Tensor,
        agent_advs:torch.Tensor
    ):
        state_gates = self.state_mlp(states)
        adv_gates = self.adv_mlp(agent_advs)
        gate = torch.sigmoid(adv_gates) * state_gates
        action = self.action_mlp(gate)
        return action
