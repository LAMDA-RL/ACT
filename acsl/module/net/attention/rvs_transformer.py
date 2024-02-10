from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from offlinerllib.module.net.attention.transformer import TransformerEncoder, TransformerDecoder
from offlinerllib.module.net.attention.base import BaseTransformer

from acsl.module.net.attention.sequence_encoder_decoder import RLSequenceEncoder, RLSequenceDecoder
from acsl.module.net.attention.onestep_decoder import RLFilmDecoder, RLGatedDecoder

class RvSTransformer(BaseTransformer):
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        num_heads: int, 
        seq_len: int, 
        episode_len: int, 
        embed_dropout: Optional[float]=None, 
        attention_dropout: Optional[float]=None, 
        residual_dropout: Optional[float]=None, 
        pos_encoding="sinusoidal", 
        use_abs_timestep=True, 
        decoder_type="attentive", 
    ) -> None:
        super().__init__()
        self.encoder = RLSequenceEncoder(
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            embed_dim=embed_dim, 
            output_dim=0, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            seq_len=seq_len, 
            episode_len=episode_len, 
            causal=True, 
            embed_dropout=embed_dropout, 
            attention_dropout=attention_dropout, 
            residual_dropout=residual_dropout, 
            pos_encoding=pos_encoding
        )
        if decoder_type == "attentive":
            self.decoder = RLSequenceDecoder(
                embed_dim=embed_dim, 
                num_layers=num_layers, 
                num_heads=num_heads,
                seq_len=seq_len, 
                episode_len=episode_len, 
                embed_dropout=embed_dropout, 
                attention_dropout=attention_dropout, 
                residual_dropout=residual_dropout, 
                pos_encoding=pos_encoding
            )
        elif decoder_type == "film":
            self.decoder = RLFilmDecoder(
                embed_dim=embed_dim, 
                num_layers=num_layers, 
            )
        elif decoder_type == "gated":
            self.decoder = RLGatedDecoder(
                embed_dim=embed_dim, 
                num_layers=num_layers
            )
        self.use_abs_timestep = use_abs_timestep
        
    def forward(
        self, 
        states, 
        actions, 
        timesteps, 
        agent_advs, 
        key_padding_mask: Optional[torch.Tensor]=None, 
    ):
        if not self.use_abs_timestep:
            B, L, *_ = states.shape
            timesteps = torch.arange(L).repeat(B, 1).to(states.device)
        enc_src = self.encoder(
            states=states, 
            actions=actions, 
            timesteps=timesteps, 
            key_padding_mask=key_padding_mask
        )
        output = self.decoder(
            agent_advs=agent_advs, 
            enc_src=enc_src, 
            timesteps=timesteps, 
            key_padding_mask=key_padding_mask
        )
        return output
        