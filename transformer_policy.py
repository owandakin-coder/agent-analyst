"""
transformer_policy.py
=====================
Custom SB3 policy: Multi-Head Attention over the time-window dimension.

Instead of flattening the (window_size, features) observation and feeding
it through a plain MLP, this extractor treats each timestep as a token and
lets the Transformer learn temporal dependencies across the 30-day window.

Architecture
------------
obs (B, W, F)
  → Linear projection → (B, W, d_model)
  → Positional encoding
  → TransformerEncoder (num_layers × MultiHeadAttention + FFN)
  → Mean-pool over W → (B, d_model)
  → passed to SB3 actor/critic heads

Usage in TrainingPipeline
--------------------------
from transformer_policy import TransformerExtractor

policy_kwargs = dict(
    features_extractor_class=TransformerExtractor,
    features_extractor_kwargs=dict(d_model=128, nhead=4, num_layers=2),
    net_arch=[64, 64],
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal positional encoding to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
class TransformerExtractor(BaseFeaturesExtractor):
    """
    SB3 FeaturesExtractor using a Transformer encoder.

    Parameters
    ----------
    observation_space : gym.Space
        Expected shape (window_size, num_features).
    d_model : int
        Transformer hidden dimension (embedding size).
    nhead : int
        Number of attention heads. Must divide d_model evenly.
    num_layers : int
        Number of TransformerEncoder layers.
    dropout : float
        Dropout rate inside the Transformer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=d_model)

        obs_shape = observation_space.shape  # (window_size, num_features)
        if len(obs_shape) == 1:
            num_features = obs_shape[0]
        else:
            num_features = obs_shape[-1]

        # Input projection: raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # (batch, seq, d_model)
            norm_first=True,    # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # CLS token (learnable) prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, window_size, num_features)  [SB3 passes flat if 1D]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)          # treat as single timestep

        x = self.input_proj(obs)            # (B, W, d_model)
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (B, W+1, d_model)

        x = self.transformer(x)             # (B, W+1, d_model)
        return x[:, 0]                      # CLS output → (B, d_model)
