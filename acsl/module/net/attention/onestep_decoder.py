import torch
import torch.nn as nn

from offlinerllib.module.net.mlp import MLP

class RLFilmDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_layers: int, 
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.adv_embed = nn.Linear(1, embed_dim)
        self.film = nn.Linear(embed_dim, 2 * embed_dim * 2)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.linear3 = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, agent_advs, enc_src, **kwargs):
        B, L, *_ = agent_advs.shape
        adv_embedding = self.adv_embed(agent_advs)
        out = enc_src[:, 0::2]
        gamma, beta = torch.split(self.film(adv_embedding).reshape(B, L, 2, -1), self.embed_dim, dim=-1)
        out = torch.relu(
            gamma[:, :, 0] * self.linear1(out) + beta[:, :, 0]
        )
        out = torch.relu(
            gamma[:, :, 1] * self.linear2(out) + beta[:, :, 1]
        )
        out = self.linear3(out)
        return out
    

class RLGatedDecoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_layers: int, 
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.adv_embed = nn.Linear(1, embed_dim)
        self.state_mlp = nn.Linear(embed_dim, embed_dim)
        self.context_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU(), 
            nn.Linear(embed_dim, embed_dim)
        )

        
    def forward(self, agent_advs, enc_src, **kwargs):
        B, L, *_ = agent_advs.shape
        adv_embedding = self.adv_embed(agent_advs)
        out = enc_src[:, 0::2]
        out = self.state_mlp(out)
        
        # start gate
        context_gate = self.context_mlp(adv_embedding)
        out = out * context_gate
        return self.final_mlp(out)
        
