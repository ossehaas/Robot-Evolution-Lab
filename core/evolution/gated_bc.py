"""PPO with gated behaviour cloning (Microduck's "VelStand" recipe, journal step 39): while the policy learns balance with
RL, it keeps copying two frozen teachers, chosen by its posture, so it does not forget walking or getting up.

    samples tilted more than 35 deg  -> copy the get-up teacher (with a zero "stand still" command, as it was trained)
    samples tilted less than 25 deg  -> copy the walking teacher (its actions rescaled from its own action_scale)
    in between                       -> RL only

Each iteration: the normal PPO update, then a few epochs of copying on the same rollout's observations (a separate
optimizer on the actor, mean actions vs teacher actions). Enabled from core.evolution.train with --bc_walker and
--bc_getup, which fill CONFIG before the runner builds the algorithm.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

EXPRESSION = ("neck_j0", "head_j1", "tail_j0")
CONFIG: dict = {}  # walker / getup: TorchScript policies; walker_scale, student_scale; obs_joint_names, obs_history,
# scan_points; lr, epochs, minibatches


class TwoTeachers:
    """Teacher actions for a batch of policy observations, and which samples each teacher covers."""

    def __init__(self, walker, getup, walker_scale: float, student_scale: float, names: list[str], hist: int,
                 scan_points: int, obs_dim: int, device):
        n = len(names)
        frame = 9 + 2 * n + 2 + 3  # gravity, gyro, accel, joint angles, last actions, clock, command
        if frame * hist + scan_points != obs_dim:
            raise ValueError(f"unexpected observation layout: {frame} x {hist} + {scan_points} != {obs_dim}")
        self.walker, self.getup, self.n = walker, getup, n
        self.walker_scale, self.student_scale = walker_scale, student_scale
        last = torch.cat([torch.arange(k * frame + 9 + n, k * frame + 9 + 2 * n) for k in range(hist)])
        self.leg_pos = torch.tensor([i for i, j in enumerate(names) if j not in EXPRESSION], device=device)
        self.leg_last = last.view(hist, n)[:, self.leg_pos.cpu()].flatten().to(device)
        self.command = torch.cat([torch.arange(k * frame + frame - 3, k * frame + frame) for k in range(hist)]).to(device)
        self.gravity_z = (hist - 1) * frame + 2  # newest frame: gravity z (-1 when upright)

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(targets, use_getup, use_walker) for raw policy observations (N, obs_dim)."""
        up = -obs[:, self.gravity_z]
        use_getup, use_walker = up < 0.82, up > 0.906  # tilt > 35 deg / < 25 deg
        w_obs = obs.clone()
        # the walker sees its last leg actions in its own scale; neck, head and tail stay at rest
        w_obs[:, self.leg_last] = w_obs[:, self.leg_last] * self.student_scale / self.walker_scale
        target = torch.zeros(obs.shape[0], self.n, device=obs.device)
        target[:, self.leg_pos] = self.walker(w_obs).clamp(-1.0, 1.0) * self.walker_scale / self.student_scale
        g_obs = obs.clone()
        g_obs[:, self.command] = 0.0
        target = torch.where(use_getup.unsqueeze(-1), self.getup(g_obs).clamp(-1.0, 1.0), target)
        return target, use_getup, use_walker


class GatedBCPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = CONFIG
        self.teachers = None
        if not c:
            return
        obs_dim = self.storage.observations["policy"].shape[-1]
        self.teachers = TwoTeachers(c["walker"], c["getup"], c["walker_scale"], c["student_scale"], c["obs_joint_names"],
                                    c["obs_history"], c["scan_points"], obs_dim, self.device)
        self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.get("lr", 3e-4))
        self.bc_epochs, self.bc_minibatches = c.get("epochs", 2), c.get("minibatches", 4)

    def update(self) -> dict[str, float]:
        if self.teachers is None or not any(p.requires_grad for p in self.actor.parameters()):
            return super().update()  # also during the critic warm-up (actor frozen)
        obs = self.storage.observations["policy"].reshape(-1, self.storage.observations["policy"].shape[-1]).clone()
        losses = super().update()  # PPO first (it clears the storage), then copying on the same observations
        target, use_getup, use_walker = self.teachers(obs)
        keep = use_getup | use_walker
        obs, target = obs[keep], target[keep]
        m = obs.shape[0]
        total = 0.0
        if m >= self.bc_minibatches:
            batch = m // self.bc_minibatches
            for _ in range(self.bc_epochs):
                perm = torch.randperm(m, device=obs.device)
                for b in range(self.bc_minibatches):
                    idx = perm[b * batch : (b + 1) * batch]
                    pred = self.actor(TensorDict({"policy": obs[idx]}, batch_size=[len(idx)]))
                    loss = torch.nn.functional.mse_loss(pred, target[idx])
                    self.bc_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.bc_optimizer.step()
                    total += float(loss)
            total /= self.bc_epochs * self.bc_minibatches
        losses["bc"] = total
        losses["bc_share_getup"] = float(use_getup.float().mean())
        losses["bc_share_walker"] = float(use_walker.float().mean())
        return losses
