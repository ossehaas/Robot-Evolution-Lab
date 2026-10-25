"""PPO settings shared by every candidate (RSL-RL)."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class EvoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 100
    experiment_name = "evolution"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def apply_memory(agent_cfg: EvoPPORunnerCfg, memory: str | None, hidden: int = 128) -> EvoPPORunnerCfg:
    """Give actor and critic a recurrent layer ("gru" or "lstm"): a memory of the whole episode so far instead of
    a fixed window of recent observations."""
    if not memory or memory == "none":
        return agent_cfg
    agent_cfg.actor = RslRlRNNModelCfg(
        hidden_dims=[128, 64], activation="elu", obs_normalization=True, rnn_type=memory, rnn_hidden_dim=hidden,
        rnn_num_layers=1, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    agent_cfg.critic = RslRlRNNModelCfg(
        hidden_dims=[128, 64], activation="elu", obs_normalization=True, rnn_type=memory, rnn_hidden_dim=hidden,
        rnn_num_layers=1,
    )
    return agent_cfg
