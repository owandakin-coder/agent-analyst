"""
training_pipeline.py
====================
צינור אימון: Walk-Forward Validation + Optuna Hyperparameter Tuning.
⚠️ לצרכי מחקר בלבד. אין שימוש בכסף אמיתי.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import TradingEnvironment
from transformer_policy import TransformerExtractor
import benchmark

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── קבועים מ-config.yaml ────────────────────────────────────────────────────
try:
    from config_loader import CFG as _CFG
    ENSEMBLE_SEEDS  = _CFG.ensemble_seeds
    ENSEMBLE_STEPS  = _CFG.ensemble_timesteps
    TRIAL_TIMESTEPS = _CFG.trial_timesteps
    TRAIN_START     = _CFG.train_start
    TRAIN_END       = _CFG.train_end
    VAL_START       = _CFG.val_start
    VAL_END         = _CFG.val_end
    TEST_START      = _CFG.test_start
    TEST_END        = _CFG.test_end
    MODEL_DIR       = _CFG.model_dir
    LOG_DIR         = _CFG.logs_dir
    EVAL_START_OFFSETS = [int(v) for v in _CFG.get("training", "eval_start_offsets", default=[0, 21, 42])]
    EVAL_MIN_ROWS = int(_CFG.get("training", "eval_min_rows", default=126))
    VALIDATION_SCORE_WEIGHTS = {
        "annualized_return": float(_CFG.get("training", "validation_score_weights", "annualized_return", default=100.0)),
        "sharpe": float(_CFG.get("training", "validation_score_weights", "sharpe", default=5.0)),
        "max_drawdown": float(_CFG.get("training", "validation_score_weights", "max_drawdown", default=100.0)),
        "calmar": float(_CFG.get("training", "validation_score_weights", "calmar", default=2.0)),
    }
    TRANSFORMER_CONFIG = {
        "d_model": int(_CFG.get("training", "transformer", "d_model", default=128)),
        "nhead": int(_CFG.get("training", "transformer", "nhead", default=4)),
        "num_layers": int(_CFG.get("training", "transformer", "num_layers", default=2)),
        "dropout": float(_CFG.get("training", "transformer", "dropout", default=0.1)),
    }
    POLICY_HEAD_ARCH = list(_CFG.get("training", "policy_head", default=[64, 64]))
except Exception:
    ENSEMBLE_SEEDS  = [0, 42, 123]
    ENSEMBLE_STEPS  = 500_000
    TRIAL_TIMESTEPS = 50_000
    TRAIN_START     = "2015-01-01"
    TRAIN_END       = "2019-12-31"
    VAL_START       = "2020-01-01"
    VAL_END         = "2021-12-31"
    TEST_START      = "2022-01-01"
    TEST_END        = "2024-12-31"
    MODEL_DIR       = "models"
    LOG_DIR         = "logs"
    EVAL_START_OFFSETS = [0, 21, 42]
    EVAL_MIN_ROWS = 126
    VALIDATION_SCORE_WEIGHTS = {
        "annualized_return": 100.0,
        "sharpe": 5.0,
        "max_drawdown": 100.0,
        "calmar": 2.0,
    }
    TRANSFORMER_CONFIG = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }
    POLICY_HEAD_ARCH = [64, 64]


def _env_int_override(env_var: str, default: int) -> int:
    """Lets a scheduled/cloud run scope step counts down without editing config.yaml.

    config.yaml stays the full-quality source of truth for local/manual
    training. .github/workflows/retrain.yml sets these two env vars to a
    reduced budget because the unscoped run (10 trials x 50k steps, before
    even reaching the final model or ensemble) never finished inside
    GitHub's 350-minute hosted-runner ceiling — see retrain.yml for the
    real run history that led to this.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


TRIAL_TIMESTEPS = _env_int_override("ATZMA_TRIAL_TIMESTEPS", TRIAL_TIMESTEPS)
ENSEMBLE_STEPS = _env_int_override("ATZMA_ENSEMBLE_TIMESTEPS", ENSEMBLE_STEPS)


def _annualized_return_from_equity(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    total_return = float((equity[-1] - equity[0]) / equity[0])
    n_years = (len(equity) - 1) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def score_validation_metrics(metrics: dict, weights: dict | None = None) -> float:
    cfg = weights or VALIDATION_SCORE_WEIGHTS
    annualized_return = float(metrics.get("annualized_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    calmar = float(metrics.get("calmar", 0.0))
    return (
        annualized_return * cfg.get("annualized_return", 100.0)
        + sharpe * cfg.get("sharpe", 5.0)
        + calmar * cfg.get("calmar", 2.0)
        - max_drawdown * cfg.get("max_drawdown", 100.0)
    )


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
        self.window_size     = int(_CFG.window_size) if "_CFG" in globals() else 30
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

        # פילוח לפי תקופה
        self.train_data = self._slice_data(TRAIN_START, TRAIN_END)
        self.val_data   = self._slice_data(VAL_START,   VAL_END)
        self.test_data  = self._slice_data(TEST_START,  TEST_END)
        self.validation_slices = self._build_validation_slices(self.val_data)

        self.best_params: dict = {}
        self.best_validation_summary: dict = {}
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
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=max(2, min(5, self.n_optuna_trials or 2))
            ),
        )
        study.optimize(self._optuna_objective, n_trials=self.n_optuna_trials)
        self._cleanup_trial_models(self.n_optuna_trials)
        if study.best_trial:
            self.best_validation_summary = {
                "score": float(study.best_trial.value),
                "params": study.best_trial.params,
                "user_attrs": dict(study.best_trial.user_attrs),
            }
            with open(os.path.join(MODEL_DIR, "best_validation_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(self.best_validation_summary, handle, indent=2)
        return study.best_params

    def _cleanup_trial_models(self, n_trials: int = 0):
        """Deletes ALL temporary Optuna trial model files (trial_*.zip)."""
        import glob as _glob
        removed = 0
        for path in _glob.glob(os.path.join(MODEL_DIR, "trial_*.zip")):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        print(f"[Pipeline] Cleaned up {removed} trial model file(s).")

    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: train briefly, evaluate on multiple validation slices."""
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
            total_timesteps=TRIAL_TIMESTEPS,
            model_name=f"trial_{trial.number}",
        )

        slice_metrics: list[dict] = []
        for idx, (label, val_slice) in enumerate(self.validation_slices):
            metrics = self._evaluate_metrics(model, val_slice, vec_norm)
            metrics["slice"] = label
            slice_metrics.append(metrics)
            running_score = float(np.mean([m["validation_score"] for m in slice_metrics]))
            trial.report(running_score, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        avg_score = float(np.mean([m["validation_score"] for m in slice_metrics]))
        trial.set_user_attr("validation_windows", [m["slice"] for m in slice_metrics])
        trial.set_user_attr("avg_validation_score", avg_score)
        trial.set_user_attr("avg_sharpe", float(np.mean([m["sharpe"] for m in slice_metrics])))
        trial.set_user_attr("avg_max_drawdown", float(np.mean([m["max_drawdown"] for m in slice_metrics])))
        trial.set_user_attr("avg_annualized_return", float(np.mean([m["annualized_return"] for m in slice_metrics])))
        return avg_score

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
            total_timesteps=ENSEMBLE_STEPS,
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

        # Transformer feature extractor + small MLP head
        transformer_cfg = dict(TRANSFORMER_CONFIG)
        policy_kwargs = dict(
            features_extractor_class=TransformerExtractor,
            features_extractor_kwargs=transformer_cfg,
            net_arch=list(POLICY_HEAD_ARCH),
        )

        # verbose=1 for final model, 0 for Optuna trials (keep output clean)
        is_final = not model_name.startswith("trial_") and not model_name.startswith("ensemble_")
        verbosity = 1 if is_final else 0

        # Extract seed from params if present, else default 0
        model_params = dict(params)
        seed = model_params.pop("seed", 0)

        model = PPO(
            "MlpPolicy",
            vec_norm,
            verbose=verbosity,
            seed=seed,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            **model_params,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=is_final)
        model.save(os.path.join(MODEL_DIR, model_name))
        return model, vec_env, vec_norm

    def train_ensemble(self, params: dict | None = None) -> None:
        """
        Trains ENSEMBLE_SEEDS independent PPO models on train+val combined.
        Each is saved as models/ensemble_N.zip + models/ensemble_norm_N.pkl.

        Using different random seeds ensures diversity — each model will
        explore slightly different policies, reducing collective variance.
        """
        _params = params or self.best_params
        if not _params:
            raise RuntimeError(
                "Call run() first to get best_params, or pass params explicitly."
            )

        # Build combined train+val data (same as _train_final)
        combined_data = {}
        for ticker in self.train_data:
            combined_data[ticker] = pd.concat(
                [self.train_data[ticker], self.val_data[ticker]]
            ).sort_index()
            combined_data[ticker] = combined_data[ticker][
                ~combined_data[ticker].index.duplicated(keep="last")
            ]

        print(f"\n[Ensemble] Training {len(ENSEMBLE_SEEDS)} models "
              f"(seeds={ENSEMBLE_SEEDS}, steps={ENSEMBLE_STEPS:,}) ...")

        for i, seed in enumerate(ENSEMBLE_SEEDS):
            print(f"\n  [Ensemble {i}] seed={seed} ...")
            np.random.seed(seed)

            # Pass seed into params so _train_model can set it on PPO
            params_with_seed = {**_params, "seed": seed}

            model, _, vec_norm = self._train_model(
                combined_data,
                params_with_seed,
                total_timesteps=ENSEMBLE_STEPS,
                model_name=f"ensemble_{i}",
            )
            vec_norm.save(os.path.join(MODEL_DIR, f"ensemble_norm_{i}.pkl"))
            print(f"  [Ensemble {i}] Saved ensemble_{i}.zip + ensemble_norm_{i}.pkl")

        print("\n[Ensemble] All members trained and saved.")

    # ──────────────────────────────────────────────────────────────────────────
    # הערכה
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_metrics(self, model: PPO, data: dict, vec_norm: VecNormalize) -> dict:
        """Runs one full episode and returns risk-aware validation metrics."""
        eval_env_fn = lambda: TradingEnvironment(data)
        eval_vec    = DummyVecEnv([eval_env_fn])
        eval_norm   = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                                   clip_obs=10.0, training=False)
        # Copy running statistics from training normalizer
        eval_norm.obs_rms  = vec_norm.obs_rms
        eval_norm.ret_rms  = vec_norm.ret_rms

        obs = eval_norm.reset()
        done = False
        total_reward = 0.0
        equity_curve = [100_000.0]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_norm.step(action)
            done = dones[0]
            total_reward += float(reward[0])
            net_worth = float((infos or [{}])[0].get("net_worth", equity_curve[-1]))
            equity_curve.append(net_worth)

        eval_norm.close()
        equity = np.asarray(equity_curve, dtype=float)
        metrics = benchmark.compute_metrics(equity, "validation")
        metrics["annualized_return"] = _annualized_return_from_equity(equity)
        metrics["validation_reward"] = float(total_reward)
        metrics["validation_score"] = score_validation_metrics(metrics)
        return metrics

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

    def _build_validation_slices(self, data: dict[str, pd.DataFrame]) -> list[tuple[str, dict[str, pd.DataFrame]]]:
        """Builds overlapping validation slices to reduce single-window overfitting."""
        if not data:
            return []

        min_rows = max(self.window_size + 30, EVAL_MIN_ROWS)
        base_len = min(len(df) for df in data.values()) if data else 0
        slices: list[tuple[str, dict[str, pd.DataFrame]]] = []
        for offset in EVAL_START_OFFSETS:
            if base_len - offset < min_rows:
                continue
            sliced = {ticker: df.iloc[offset:].copy() for ticker, df in data.items()}
            slices.append((f"offset_{offset}", sliced))

        return slices or [("full_validation", data)]
