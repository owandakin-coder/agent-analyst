"""
ensemble_agent.py
=================
Ensemble of multiple PPO models — reduces variance and improves robustness.

Each model was trained with a different random seed or hyperparameter set.
At inference time, all models vote on the action; the final action is a
weighted average, clipped to [-1, 1].

Why ensemble works
------------------
A single RL model can overfit to quirks in the training period.  Three models
trained independently are unlikely to share the same failure modes, so their
average is more stable across unseen market regimes.

Usage
-----
from ensemble_agent import EnsembleAgent, load_ensemble

agent = load_ensemble()                     # loads from models/ directory
obs   = trader._build_observation(...)      # raw (W, F) numpy array
action = agent.predict(obs[np.newaxis])     # (num_stocks,) numpy array
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trading_env import TradingEnvironment

log = logging.getLogger("EnsembleAgent")

MODEL_DIR = "models"

# Ensemble member file names stored in models/
ENSEMBLE_MEMBERS = [
    ("ensemble_0.zip", "ensemble_norm_0.pkl"),
    ("ensemble_1.zip", "ensemble_norm_1.pkl"),
    ("ensemble_2.zip", "ensemble_norm_2.pkl"),
]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnsembleMember:
    model:    PPO
    vec_norm: VecNormalize
    weight:   float = 1.0


class EnsembleAgent:
    """
    Combines multiple PPO models with configurable weights.

    Parameters
    ----------
    members : list[EnsembleMember]
        Loaded models + normalisers.
    weights : list[float] | None
        Per-model weights.  If None, equal weighting is used.
        Weights are normalised to sum to 1 internally.
    """

    def __init__(
        self,
        members: list[EnsembleMember],
        weights: list[float] | None = None,
    ):
        if not members:
            raise ValueError("EnsembleAgent requires at least one member model.")

        self.members = members
        if weights is not None:
            total = sum(weights)
            self._weights = [w / total for w in weights]
        else:
            n = len(members)
            self._weights = [1.0 / n] * n

        log.info(
            f"EnsembleAgent initialised with {len(members)} models | "
            f"weights={[f'{w:.2f}' for w in self._weights]}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def predict(
        self,
        obs_raw: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Weighted average prediction across all member models.

        Parameters
        ----------
        obs_raw : np.ndarray
            Raw (un-normalised) observation, shape (1, window_size, features).
        deterministic : bool
            If True, each model uses its mean action (no sampling noise).

        Returns
        -------
        np.ndarray
            Clipped action vector in [-1, 1], shape (num_stocks,).
        """
        ensemble_action = np.zeros(self._action_dim())

        for member, weight in zip(self.members, self._weights):
            obs_norm = member.vec_norm.normalize_obs(obs_raw)
            action, _ = member.model.predict(obs_norm, deterministic=deterministic)
            ensemble_action += np.array(action).flatten() * weight

        return np.clip(ensemble_action, -1.0, 1.0)

    def _action_dim(self) -> int:
        return self.members[0].model.action_space.shape[0]

    # ──────────────────────────────────────────────────────────────────────────
    def vote_summary(self, obs_raw: np.ndarray) -> dict:
        """Returns per-model actions for debugging / dashboard display."""
        summary = {}
        for i, (member, weight) in enumerate(zip(self.members, self._weights)):
            obs_norm = member.vec_norm.normalize_obs(obs_raw)
            action, _ = member.model.predict(obs_norm, deterministic=True)
            summary[f"model_{i}"] = {
                "weight": weight,
                "action": np.array(action).flatten().tolist(),
            }
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Loader helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ensemble(
    tickers: list[str] | None = None,
    dummy_data: dict | None = None,
) -> EnsembleAgent:
    """
    Loads all ensemble members from models/.

    Falls back to final_model.zip (single model) if no ensemble files exist,
    so the live trader always has something to work with.
    """
    members = []

    for model_file, norm_file in ENSEMBLE_MEMBERS:
        model_path = os.path.join(MODEL_DIR, model_file)
        norm_path  = os.path.join(MODEL_DIR, norm_file)

        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            log.debug(f"Ensemble member not found: {model_file} — skipping.")
            continue

        try:
            model    = PPO.load(model_path)
            env      = _make_dummy_env(tickers, dummy_data)
            vec_norm = VecNormalize.load(norm_path, env)
            vec_norm.training    = False
            vec_norm.norm_reward = False
            members.append(EnsembleMember(model=model, vec_norm=vec_norm))
            log.info(f"Loaded ensemble member: {model_file}")
        except Exception as exc:
            log.warning(f"Failed to load {model_file}: {exc}")

    if not members:
        log.warning("No ensemble members found — falling back to final_model.zip.")
        members = [_load_fallback(tickers, dummy_data)]

    return EnsembleAgent(members)


def _make_dummy_env(
    tickers: list[str] | None,
    dummy_data: dict | None,
) -> DummyVecEnv:
    from data_manager import DataManager

    if dummy_data is None:
        _tickers = tickers or [
            "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META",
            "TSLA", "JPM", "V", "BAC", "JNJ", "UNH", "XOM", "WMT", "SPY",
        ]
        dm = DataManager(tickers=_tickers, start="2022-01-01", end="2024-12-31")
        dm.load_all(force_download=False)
        dummy_data = dm.get_aligned_data()

    return DummyVecEnv([lambda: TradingEnvironment(dummy_data)])


def _load_fallback(
    tickers: list[str] | None,
    dummy_data: dict | None,
) -> EnsembleMember:
    model_path = os.path.join(MODEL_DIR, "final_model.zip")
    norm_path  = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    model    = PPO.load(model_path)
    env      = _make_dummy_env(tickers, dummy_data)
    vec_norm = VecNormalize.load(norm_path, env)
    vec_norm.training    = False
    vec_norm.norm_reward = False

    return EnsembleMember(model=model, vec_norm=vec_norm, weight=1.0)
