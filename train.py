"""
train.py — Train the PPO agent on historical BTC/USDT data.

Usage:
    python train.py                          # default settings
    python train.py --timesteps 500000       # more training
    python train.py --symbol ETHUSDT         # train on ETH instead

TensorBoard:
    tensorboard --logdir logs/
"""

import argparse
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from data import fetch_ohlcv
from env import CryptoTradingEnv
from agent import build_agent, save_agent, MetricsCallback, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--initial-balance", type=float, default=10_000)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--model-name", default="ppo_crypto")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load & split data ──────────────────────────────────────────────────────
    df = fetch_ohlcv(args.symbol, args.interval, args.start, args.end)
    print(f"[Train] Total candles: {len(df)}")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"[Train] Train: {len(train_df)} | Val: {len(val_df)}")

    env_kwargs = {
        "initial_balance": args.initial_balance,
        "trading_fee": args.fee,
    }

    # ── Build vectorised + normalised environments ─────────────────────────────
    train_vec = DummyVecEnv([make_env(train_df, **env_kwargs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    val_env = CryptoTradingEnv(val_df, **env_kwargs)

    # ── Build agent ────────────────────────────────────────────────────────────
    agent = build_agent(train_vec)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        MetricsCallback(eval_env=val_env, eval_freq=10_000, verbose=1),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path="models/checkpoints/",
            name_prefix=args.model_name,
            verbose=1,
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for {args.timesteps:,} timesteps...")
    print("[Train] Run: tensorboard --logdir logs/  to monitor\n")

    agent.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=args.model_name,
        progress_bar=True,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    save_agent(agent, train_vec, args.model_name)

    # ── Quick val summary ──────────────────────────────────────────────────────
    print("\n[Train] Running final validation episode...")
    obs, _ = val_env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, _ = val_env.step(action)

    metrics = val_env.get_metrics()
    print("\n── Validation Results ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
