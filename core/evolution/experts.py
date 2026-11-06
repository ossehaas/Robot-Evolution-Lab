"""One model, many skills (journal step 45, next-steps section 0b): a frozen base policy plus a skill expert and a gate.

    base     the trained policy (walk, get up, balance), frozen: weights and observation normalizer never change
    expert   a small MLP that reads the observations and the base's hidden layers (what the base already knows) and
             outputs a correction to the base's action (its own strategy); its last layer starts at zero, so the model
             starts out acting exactly like the base
    gate     a small MLP on the observations, sigmoid: how much of the correction is used. Trained only on the
             situation label the environment provides during training (observation group "expert_label", e.g. "in
             the stairs plan's zone"); at run time it predicts the situation from the sensors

    action = base(obs) + gate(obs) * expert(obs, base features)

PPO trains the expert (and the action noise); the gate gets no PPO gradient. After each PPO update ExpertPPO trains
the gate on the label (binary cross-entropy, positives weighted up to balance) and keeps the expert quiet outside its
situation (its correction pulled to zero where the label is 0), so a gate that opens a little by mistake changes
little. Export: one TorchScript policy.pt (forward = the action, gate_value = the gate), used by every tool unchanged.

Enabled from core.evolution.train with --expert_base (fills CONFIG before the runner builds the algorithm).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

CONFIG: dict = {}  # lr, epochs, minibatches, quiet_weight, copy_weight, max_pos_weight
LABEL = "expert_label"
TARGET = "expert_target"  # step 45b: (target actions, mask) from the plan; the expert copies it where the mask is 1


def _base_forward(layers: nn.ModuleList, act: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The base MLP layer by layer: (output, hidden activations)."""
    feats: list[torch.Tensor] = []
    n = len(layers)
    for i, layer in enumerate(layers):
        x = layer(x)
        if i < n - 1:
            x = act(x)
            feats.append(x)
    return x, feats


class ExpertActor(MLPModel):
    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int,
                 hidden_dims=(256, 128, 64), activation: str = "elu", obs_normalization: bool = False,
                 distribution_cfg: dict | None = None, expert_dims=(128, 64), gate_dims=(64, 32)) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         distribution_cfg)
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        self.base_layers = nn.ModuleList(linears)  # the same modules as self.mlp (shared weights)
        self.base_act = next(m for m in self.mlp if not isinstance(m, nn.Linear))
        feat_dim = sum(m.out_features for m in linears[:-1])
        self.expert = MLP(self.obs_dim + feat_dim, output_dim, list(expert_dims), activation)
        self.gate = MLP(self.obs_dim, 1, list(gate_dims), activation)
        with torch.no_grad():
            self.expert[-1].weight.zero_()
            self.expert[-1].bias.zero_()
            self.gate[-1].weight.mul_(0.1)
            self.gate[-1].bias.fill_(-3.0)  # starts nearly closed
        self.freeze_base()
        self.last_gate: torch.Tensor | None = None

    def freeze_base(self) -> None:
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        if isinstance(self.obs_normalizer, nn.Module):
            for p in self.obs_normalizer.parameters():
                p.requires_grad_(False)

    def update_normalization(self, obs: TensorDict) -> None:
        pass  # the base's normalizer is part of the frozen base

    def parts(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(base action, expert correction, gate logit) for normalized observations."""
        base, feats = _base_forward(self.base_layers, self.base_act, latent)
        delta = self.expert(torch.cat([latent] + [f.detach() for f in feats], dim=-1))
        return base.detach(), delta, self.gate(latent).squeeze(-1)

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        base, delta, logit = self.parts(latent)
        gate = torch.sigmoid(logit).detach()  # the gate learns only from the label
        self.last_gate = gate
        out = base + gate.unsqueeze(-1) * delta
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(out)
                return self.distribution.sample()
            return self.distribution.deterministic_output(out)
        return out

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """A checkpoint of the plain base (no expert, no gate) loads too: the new parts keep their start values."""
        own = set(self.state_dict().keys())
        missing = own - set(state_dict.keys())
        if missing and all(k.startswith(("expert.", "gate.", "base_layers.")) for k in missing):
            sd = dict(state_dict)
            for k in [k for k in missing if k.startswith("base_layers.")]:
                i, rest = k.split(".")[1], k.split(".", 2)[2]
                layer = list(self.base_layers)[int(i)]
                idx = list(self.mlp).index(layer)
                sd[k] = state_dict[f"mlp.{idx}.{rest}"]
            return super().load_state_dict(sd, strict=False, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def as_jit(self) -> nn.Module:
        return _TorchExpertModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxExpertModel(self)


class _TorchExpertModel(nn.Module):
    """Exportable model: forward(obs) -> action; gate_value(obs) -> gate in [0, 1]."""

    def __init__(self, model: ExpertActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.base_layers = copy.deepcopy(model.base_layers)
        self.base_act = copy.deepcopy(model.base_act)
        self.expert = copy.deepcopy(model.expert)
        self.gate = copy.deepcopy(model.gate)
        self.last_layer = len(model.base_layers) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        h = x
        feats: list[torch.Tensor] = [x]
        for i, layer in enumerate(self.base_layers):  # the base MLP layer by layer (as _base_forward)
            h = layer(h)
            if i < self.last_layer:
                h = self.base_act(h)
                feats.append(h)
        delta = self.expert(torch.cat(feats, dim=-1))
        return h + torch.sigmoid(self.gate(x)) * delta

    @torch.jit.export
    def gate_value(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(self.obs_normalizer(x))).squeeze(-1)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxExpertModel(_TorchExpertModel):
    is_recurrent: bool = False

    def __init__(self, model: ExpertActor) -> None:
        super().__init__(model)
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class ExpertPPO(PPO):
    """PPO on the expert, then gate and quiet training on the same rollout (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        actor = self.actor
        params = list(actor.gate.parameters()) + list(actor.expert.parameters())
        self.side_optimizer = torch.optim.Adam(params, lr=CONFIG.get("lr", 1e-3))
        self.side_epochs, self.side_minibatches = CONFIG.get("epochs", 2), CONFIG.get("minibatches", 4)

    def update(self) -> dict[str, float]:
        obs_all = self.storage.observations
        if LABEL not in obs_all.keys():
            raise RuntimeError("ExpertPPO needs the observation group 'expert_label' (env cfg expert_label_obs)")
        obs = obs_all["policy"].reshape(-1, obs_all["policy"].shape[-1]).clone()
        label = obs_all[LABEL].reshape(-1).clone()
        copy_w = CONFIG.get("copy_weight", 0.0)
        if copy_w > 0:
            if TARGET not in obs_all.keys():
                raise RuntimeError("copy_weight needs the observation group 'expert_target' (env cfg expert_target_obs)")
            target, tmask = obs_all[TARGET].reshape(-1, obs_all[TARGET].shape[-1]).clone().chunk(2, dim=-1)
        losses = super().update()  # PPO first (it clears the storage)
        actor = self.actor
        if not any(p.requires_grad for p in actor.gate.parameters()):
            return losses  # critic warm-up
        pos = float(label.sum())
        pos_weight = torch.tensor(min(CONFIG.get("max_pos_weight", 20.0), (label.numel() - pos) / max(pos, 1.0)),
                                  device=obs.device)
        quiet_w = CONFIG.get("quiet_weight", 1.0)
        m = obs.shape[0]
        batch = m // self.side_minibatches
        bce_sum = quiet_sum = copy_sum = 0.0
        for _ in range(self.side_epochs):
            perm = torch.randperm(m, device=obs.device)
            for b in range(self.side_minibatches):
                idx = perm[b * batch : (b + 1) * batch]
                latent = actor.obs_normalizer(obs[idx])
                base_a, delta, logit = actor.parts(latent)
                y = label[idx]
                bce = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
                off = y < 0.5
                quiet = (delta[off] ** 2).mean() if bool(off.any()) else delta.sum() * 0.0
                loss = bce + quiet_w * quiet
                if copy_w > 0:
                    # copy the plan's pose: the whole action (base + gate x correction) towards the plan's joints
                    mk = tmask[idx]
                    act = base_a + torch.sigmoid(logit).detach() * delta
                    copy = ((act - target[idx]) ** 2 * mk).sum() / mk.sum().clamp(min=1.0)
                    loss = loss + copy_w * copy
                    copy_sum += float(copy)
                self.side_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.gate.parameters()) + list(actor.expert.parameters()),
                                         self.max_grad_norm)
                self.side_optimizer.step()
                bce_sum += float(bce)
                quiet_sum += float(quiet)
        k = self.side_epochs * self.side_minibatches
        with torch.no_grad():
            latent = actor.obs_normalizer(obs)
            _, delta, logit = actor.parts(latent)
            gate = torch.sigmoid(logit)
            on = label > 0.5
            losses["gate_bce"] = bce_sum / k
            losses["expert_quiet"] = quiet_sum / k
            losses["gate_label_share"] = float(on.float().mean())
            losses["gate_on_mean"] = float(gate[on].mean()) if bool(on.any()) else float("nan")
            losses["gate_off_mean"] = float(gate[~on].mean()) if bool((~on).any()) else float("nan")
            losses["gate_accuracy"] = float(((gate > 0.5) == on).float().mean())
            losses["expert_delta_on"] = float(delta[on].abs().mean()) if bool(on.any()) else float("nan")
            if copy_w > 0:
                losses["expert_copy"] = copy_sum / k
        return losses
