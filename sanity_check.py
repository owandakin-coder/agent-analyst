"""
sanity_check.py — Run 3 random episodes to verify the env works before training.
"""

import numpy as np
from data import fetch_ohlcv
from env import CryptoTradingEnv


def main():
    print("[Sanity] Fetching data...")
    df = fetch_ohlcv("BTCUSDT", "1h", "2023-01-01", "2023-06-01")
    print(f"[Sanity] {len(df)} candles, {len(df.columns)} columns")

    env = CryptoTradingEnv(df, initial_balance=10_000)

    for episode in range(3):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape}"

        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        metrics = env.get_metrics()
        print(
            f"[Episode {episode + 1}] "
            f"steps={steps} | "
            f"total_reward={total_reward:.2f} | "
            f"return={metrics['total_return_pct']:+.2f}% | "
            f"sharpe={metrics['sharpe_ratio']:.3f}"
        )

    print("\n[Sanity] ✓ Environment working correctly")
    print(f"[Sanity] Observation space: {env.observation_space.shape}")
    print(f"[Sanity] Action space: {env.action_space}")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
