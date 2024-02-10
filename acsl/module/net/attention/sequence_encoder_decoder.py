from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.base import BaseTransformer
from offlinerllib.module.net.attention.transformer import TransformerEncoder, TransformerDecoder
from offlinerllib.module.net.attention.positional_encoding import get_pos_encoding


class RLSequenceEncoder(TransformerEncoder):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        output_dim: int, 
        num_layers: int, 
        num_heads: int, 
        seq_len: int, 
        episode_len: int, 
        causal: bool=False, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        pos_encoding="sinusoidal"
    ) -> None:
        super().__init__(
            input_dim=embed_dim, 
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            causal=causal, 
            out_ln=True, 
            pre_norm=True, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            embed_dropout=embed_dropout, 
            pos_encoding="none", # as we handle outside
        )
        seq_len = seq_len or 1024
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, episode_len+seq_len)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        if output_dim > 0:  # if output_dim is greater than 0, then act like a normal encoder
            self.out_proj = nn.Linear(embed_dim, output_dim)
        else:
            self.out_proj = nn.Identity()
        
    def forward(
        self,
        states: torch.Tensor, 
        actions: torch.Tensor, 
        timesteps: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        **kwargs
    ):
        B, L, *_ = states.shape
        state_embedding = self.pos_encoding(self.obs_embed(states), timesteps)
        action_embedding = self.pos_encoding(self.act_embed(actions), timesteps)
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask for _ in range(2)], dim=2).reshape(B, 2*L)
        stacked_input = torch.stack([state_embedding, action_embedding], dim=2).reshape(B, 2*L, state_embedding.shape[-1])
        stacked_input = self.embed_ln(stacked_input)
        out = super().forward(
            inputs=stacked_input, 
            timesteps=None,
            attention_mask=attention_mask, 
            key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        out = self.out_proj(out)
        return out
        

class RLSequenceDecoder(TransformerDecoder):
    def __init__(
        self, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int,
        seq_len: int, 
        episode_len: int, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        pos_encoding="sinusoidal"
    ) -> None:
        super().__init__(
            input_dim=embed_dim,
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            causal=False,  # cause we don't rely on causal mask to mask out other advs
            out_ln=True, 
            pre_norm=True, 
            embed_dropout=embed_dropout, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            pos_encoding="sinusoidal"
        )
        # we manually handle the embedding here
        self.pos_encoding = get_pos_encoding(pos_encoding, embed_dim, episode_len+seq_len)
        self.adv_embed = nn.Linear(1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        agent_advs, 
        enc_src, 
        timesteps, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        B, L, E = agent_advs.shape
        adv_embedding = self.pos_encoding(self.adv_embed(agent_advs), timesteps)
        pad_embedding = torch.zeros_like(adv_embedding).to(adv_embedding.device)
        stacked_input = torch.stack([adv_embedding, pad_embedding], dim=2).reshape(B, 2*L, -1)
        stacked_input = self.embed_ln(stacked_input)
        # we additionally add pos encoding for enc src
        enc_src[:, 0::2, :] = self.pos_encoding(enc_src[:, 0::2, :], timesteps)
        enc_src[:, 1::2, :] = self.pos_encoding(enc_src[:, 1::2, :], timesteps)
        
        # we mask all other advs in RL Sequence Decoder
        tgt_mask = ~torch.eye(2*L, 2*L).bool().to(agent_advs.device)
        # we also add causal src mask in RL Sequence Encoder
        src_mask = ~torch.tril(torch.ones([2*L, 2*L])).to(torch.bool).to(agent_advs.device)
        
        if key_padding_mask is not None:
            key_padding_mask = torch.stack([key_padding_mask for _ in range(2)], dim=2).reshape(B, 2*L)
        
        out = super().forward(
            tgt=stacked_input, 
            enc_src=enc_src, 
            timesteps=None, 
            tgt_attention_mask=tgt_mask,
            src_attention_mask=src_mask,  
            # tgt_key_padding_mask=key_padding_mask, # we dont pass tgt key padding mask to avoid NaN
            src_key_padding_mask=key_padding_mask, 
            do_embedding=False
        )
        return out[:, 0::2]
        
