"""
training_pipeline.py
====================
צינור אימון: Walk-Forward Validation + Optuna Hyperparameter Tuning.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from trading_env import TradingEnvironment

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── תקופות Walk-Forward ─────────────────────────────────────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2019-12-31"
VAL_START   = "2020-01-01"
VAL_END     = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-12-31"

MODEL_DIR   = "models"
LOG_DIR     = "logs"


class TrainingPipeline:
    """
    מנהל את כל תהליך האימון:
    1. Walk-Forward splits
    2. Optuna hyperparameter search
    3. אימון PPO סופי + שמירת מודל
    """

    def __init__(self, aligned_data: dict[str, pd.DataFrame], n_optuna_trials: int = 15):
        self.aligned_data    = aligned_data
        self.n_optuna_trials = n_optuna_trials
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        # פילוח לפי תקופה
        self.train_data = self._slice_data(TRAIN_START, TRAIN_END)
        self.val_data   = self._slice_data(VAL_START,   VAL_END)
        self.test_data  = self._slice_data(TEST_START,  TEST_END)

        self.best_params: dict = {}
        self.model: PPO | None = None
        self.vec_env = None
        self.vec_norm = None

    # ──────────────────────────────────────────────────────────────────────────
    # API ציבורי
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> PPO:
        """Runs Optuna search + full training. Returns trained model."""
        print("\n" + "=" * 60)
        print("  Phase 1: Hyperparameter Search (Optuna)")
        print("=" * 60)
        self.best_params = self._optuna_search()
        print(f"\n  Best params: {self.best_params}")

        print("\n" + "=" * 60)
        print("  Phase 2: Final Training with Best Parameters")
        print("=" * 60)
        self.model, self.vec_env, self.vec_norm = self._train_final(self.best_params)

        print("\n  Training complete. Model saved.")
        return self.model

    def get_test_env(self) -> tuple[DummyVecEnv, VecNormalize]:
        """מחזיר סביבת טסט מוכנה (עם אותה נרמליזציה כמו האימון)."""
        test_env = DummyVecEnv([lambda: TradingEnvironment(self.test_data)])
        test_norm = VecNormalize.load(
            os.path.join(MODEL_DIR, "vec_normalize.pkl"), test_env
        )
        test_norm.training = False
        test_norm.norm_reward = False
        return test_env, test_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Optuna
    # ──────────────────────────────────────────────────────────────────────────

    def _optuna_search(self) -> dict:
        """Searches for best hyperparameters on the validation set."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self._optuna_objective, n_trials=self.n_optuna_trials)
        self._cleanup_trial_models(self.n_optuna_trials)
        return study.best_params

    def _cleanup_trial_models(self, n_trials: int):
        """Deletes temporary Optuna trial model files."""
        for i in range(n_trials):
            path = os.path.join(MODEL_DIR, f"trial_{i}.zip")
            if os.path.exists(path):
                os.remove(path)
        print("[Pipeline] Cleaned up trial model files.")

    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: train briefly, evaluate on validation (avg of 3 seeds)."""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps":       trial.suggest_categorical("n_steps", [512, 1024, 2048]),
            "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma":         trial.suggest_float("gamma", 0.95, 0.999),
            "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
            "clip_range":    trial.suggest_float("clip_range", 0.1, 0.4),
            "n_epochs":      trial.suggest_int("n_epochs", 5, 20),
        }

        # Short training on train split
        model, _, vec_norm = self._train_model(
            self.train_data,
            params,
            total_timesteps=50_000,
            model_name=f"trial_{trial.number}",
        )

        # Evaluate on validation set — average 3 seeds to reduce noise
        rewards = [
            self._evaluate(model, self.val_data, vec_norm, seed=s)
            for s in [0, 42, 123]
        ]
        return float(np.mean(rewards))

    # ──────────────────────────────────────────────────────────────────────────
    # אימון
    # ──────────────────────────────────────────────────────────────────────────

    def _train_final(self, params: dict) -> tuple:
        """Full training with best params on train + validation combined."""
        # Combine train+val: more data including COVID years (2020-2021)
        combined_data = {}
        for ticker in self.train_data:
            df_train = self.train_data[ticker]
            df_val   = self.val_data[ticker]
            combined_data[ticker] = pd.concat([df_train, df_val]).sort_index()
            # Drop duplicates if date ranges overlap
            combined_data[ticker] = combined_data[ticker][
                ~combined_data[ticker].index.duplicated(keep="last")
            ]

        print(f"[Pipeline] Final training on train+val "
              f"({TRAIN_START} – {VAL_END}). "
              f"Rows per ticker: {len(next(iter(combined_data.values())))}")

        model, vec_env, vec_norm = self._train_model(
            combined_data,
            params,
            total_timesteps=500_000,
            model_name="final_model",
        )
        vec_norm.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
        return model, vec_env, vec_norm

    def _train_model(
        self,
        data: dict,
        params: dict,
        total_timesteps: int,
        model_name: str,
    ) -> tuple:
        """Creates env, trains PPO, returns (model, vec_env, vec_norm)."""
        env_fn = lambda: TradingEnvironment(data)
        vec_env = DummyVecEnv([env_fn])
        vec_norm = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

        # Deeper network architecture
        policy_kwargs = dict(net_arch=[256, 256, 128])

        model = PPO(
            "MlpPolicy",
            vec_norm,
            verbose=0,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            **params,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model.save(os.path.join(MODEL_DIR, model_name))
        return model, vec_env, vec_norm

    # ──────────────────────────────────────────────────────────────────────────
    # הערכה
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(self, model: PPO, data: dict, vec_norm: VecNormalize,
                  seed: int = 0) -> float:
        """Runs one full episode and returns cumulative reward."""
        eval_env_fn = lambda: TradingEnvironment(data)
        eval_vec    = DummyVecEnv([eval_env_fn])
        eval_norm   = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                                   clip_obs=10.0, training=False)
        # Copy running statistics from training normalizer
        eval_norm.obs_rms  = vec_norm.obs_rms
        eval_norm.ret_rms  = vec_norm.ret_rms

        np.random.seed(seed)
        obs = eval_norm.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = eval_norm.step(action)
            done = dones[0]
            total_reward += float(reward[0])

        eval_norm.close()
        return total_reward

    # ──────────────────────────────────────────────────────────────────────────
    # עזר
    # ──────────────────────────────────────────────────────────────────────────

    def _slice_data(self, start: str, end: str) -> dict[str, pd.DataFrame]:
        """Slices data to the specified date range."""
        sliced = {}
        for ticker, df in self.aligned_data.items():
            mask = (df.index >= start) & (df.index <= end)
            sliced[ticker] = df[mask].copy()
        return sliced
