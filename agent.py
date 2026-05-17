"""
PPO Agent wrapper around stable-baselines3.

Why PPO here:
- Continuous action space (position sizing) → policy gradient methods fit naturally
- Clip objective prevents destructive policy updates on noisy financial data
- Better sample efficiency than DQN for this observation dimensionality
- SB3's implementation is battle-tested and export-friendly (ONNX, TorchScript)
"""

from pathlib import Path
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from env import CryptoTradingEnv


MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")


def make_env(df, **env_kwargs):
    """Factory for Monitor-wrapped env (required by SB3 VecEnv)."""
    def _init():
        env = CryptoTradingEnv(df, **env_kwargs)
        return Monitor(env)
    return _init


class MetricsCallback(BaseCallback):
    """Logs custom trading metrics to TensorBoard every N episodes."""

    def __init__(self, eval_env: CryptoTradingEnv, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.eval_env.step(action)

            metrics = self.eval_env.get_metrics()
            for k, v in metrics.items():
                self.logger.record(f"trading/{k}", v)

            if self.verbose:
                print(f"\n[Metrics @ {self.n_calls}] {metrics}")
        return True


def build_agent(
    train_env: DummyVecEnv,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,      # entropy bonus — encourages exploration
    device: str = "auto",
) -> PPO:
    """
    Build PPO agent with a custom MLP policy.

    Network: [64, 64] hidden layers — shallow intentionally.
    Financial signals don't benefit from deep nets; they overfit easily.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    policy_kwargs = {
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": torch.nn.Tanh,  # Tanh bounds activations — good for normalised financial data
    }

    agent = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=str(LOGS_DIR),
        device=device,
    )

    return agent


def save_agent(agent: PPO, vec_normalize: VecNormalize, name: str = "ppo_crypto") -> None:
    """Save model weights + normalisation stats together."""
    MODELS_DIR.mkdir(exist_ok=True)
    agent.save(str(MODELS_DIR / name))
    vec_normalize.save(str(MODELS_DIR / f"{name}_vecnorm.pkl"))
    print(f"[Agent] Saved → models/{name}")


def load_agent(
    train_env: DummyVecEnv,
    name: str = "ppo_crypto",
) -> tuple[PPO, VecNormalize]:
    """Load model + normalisation stats."""
    vec_normalize = VecNormalize.load(
        str(MODELS_DIR / f"{name}_vecnorm.pkl"), train_env
    )
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    agent = PPO.load(str(MODELS_DIR / name), env=vec_normalize)
    print(f"[Agent] Loaded ← models/{name}")
    return agent, vec_normalize


def export_to_onnx(agent: PPO, name: str = "ppo_crypto") -> None:
    """
    Export policy to ONNX for deployment to any broker API.
    Input: observation vector (float32)
    Output: action (float32, shape [1])
    """
    import torch.onnx

    obs_size = agent.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)

    # Extract the policy network
    policy = agent.policy
    policy.eval()

    # Wrapper to get only the action (not value + log_prob)
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                mean_actions = self.policy.action_net(latent_pi)
                return torch.tanh(mean_actions)

    wrapper = PolicyWrapper(policy)
    output_path = str(MODELS_DIR / f"{name}.onnx")

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"[Agent] Exported ONNX → {output_path}")
