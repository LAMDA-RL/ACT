from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.gpt2 import GPT2


class AdvantageConditionedTransformer(GPT2):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        episode_len: int=1000, 
        num_heads: int=4, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
        pos_encoding="embedding", 
    ) -> None:
        super().__init__(
            input_dim=embed_dim, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            # seq_len=3*seq_len, # s, A, a, W
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
            pos_encoding="none", 
        )
        # we manually handle the embedding here
        self.pos_encoding = pos_encoding
        if pos_encoding == "embedding":
            from offlinerllib.module.net.attention.positional_encoding import PositionalEmbedding
            self.pos_embed = PositionalEmbedding(embed_dim, episode_len+seq_len)
        elif pos_encoding == "sinusoid": 
            # from offlinerllib.module.net.attention.positional_encoding import SinusoidEncoding
            # self.pos_embed = SinusoidEncoding(embed_dim, episode_len+seq_len)
            self.pos_embed = None
        elif pos_encoding == "rope":
            raise NotImplementedError
        self.pos_embed = nn.Embedding(episode_len + seq_len, embed_dim)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.aadv_embed = nn.Linear(1, embed_dim)
        
        self.embed_ln = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        agent_advs: torch.Tensor, 
        timesteps: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        **kwargs
    ):
        B, L, *_ = states.shape
        if self.pos_encoding == "sinusoid":
            pos = timesteps.float().unsqueeze(-1)
            _2i = torch.arange(0, 128, step=2).float().to(pos.device)
            time_embedding = torch.stack([
                torch.sin(pos / (10000 ** (_2i / 128))), 
                torch.cos(pos / (10000 ** (_2i / 128)))
            ], dim=-1).reshape(B, L, 128)
        else:
            time_embedding = self.pos_embed(timesteps)
        state_embedding = self.obs_embed(states) + time_embedding
        action_embedding = self.act_embed(actions) + time_embedding
        aadv_embedding = self.aadv_embed(agent_advs) + time_embedding
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask for _ in range(3)], dim=2).reshape(B, 3*L)
        stacked_input = torch.stack([state_embedding, aadv_embedding, action_embedding], dim=2).reshape(B, 3*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        
        return out[:, 1::3]
    
    
class AdvantageConditionedTransformerWithDynamics(GPT2):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        episode_len: int=1000, 
        num_heads: int=4, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        embed_dropout: Optional[float]=None, 
        **kwargs
    ) -> None:
        super().__init__(
            input_dim=embed_dim, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            seq_len=4*seq_len, # s, A, a, W
            num_heads=num_heads, 
            causal=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout
        )
        # we manually do the embeddings here
        self.pos_embed = nn.Embedding(episode_len + seq_len, embed_dim)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.aadv_embed = nn.Linear(1, embed_dim)
        self.madv_embed = nn.Linear(1, embed_dim)
        
        self.embed_ln = nn.Linear(embed_dim)
        
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        agent_advs: torch.Tensor, 
        model_advs: torch.Tensor, 
        timesteps: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        **kwargs
    ):
        B, L, *_ = states.shape
        time_embedding = self.pos_embed(timesteps)
        state_embedding = self.obs_embed(states) + time_embedding
        action_embedding = self.act_embed(actions) + time_embedding
        aadv_embedding = self.aadv_embed(agent_advs) + time_embedding
        madv_embedding = self.madv_embed(model_advs) + time_embedding
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask for _ in range(4)], dim=2).reshape(B, 4*L)
        stacked_input = torch.stack([state_embedding, aadv_embedding, action_embedding, madv_embedding], dim=2).reshape(B, 4*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None, 
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        
        return out