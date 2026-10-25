"""Adversarial motion prior (AMP) as a style reward: a discriminator learns to tell transitions (s, s') of real dog motion
(retargeted motion capture, core/mocap/retarget.py) from the robot's own transitions; the robot earns reward for
transitions the discriminator takes for real dog motion (Peng et al. 2021, Escontrela et al. 2022).

The discriminator is trained inside the environment (every few steps, on recent robot transitions and random motion
capture transitions), so the PPO trainer itself stays unchanged.

Features of a state (all available on the robot): leg joint angles and speeds, body velocity and turn rate in the body
frame, gravity direction in the body frame. No height (meaningless on stairs against flat-ground recordings).

The recordings are mirrored left-right as well (the recorded dog turns right more than left; without the mirror the
discriminator rewards drifting right).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AmpDiscriminator:
    def __init__(self, motion_file: str, joint_names: list[str], device: str, buffer_size: int = 200_000,
                 lr: float = 1e-4, grad_penalty: float = 10.0, batch: int = 4096, max_speed: float = 0.55,
                 mirror: bool = True):
        data = np.load(motion_file)
        names = list(data["joint_names"])
        cols = [names.index(n) for n in joint_names]
        dt = float(data["dt"])
        joints = data["joints"][:, cols]
        clip = data["clip"]
        jvel = np.gradient(joints, dt, axis=0)
        lin, ang, grav = data["lin_vel_b"], data["ang_vel_b"], data["gravity_b"]
        feats = np.concatenate([joints, jvel, lin, ang, grav], -1)
        if mirror:
            # left <-> right legs (the robot's joint axes are mirrored: same angle = mirrored pose), y components of
            # vectors and x/z components of turn rates change sign; the copies are separate clips
            perm = [joint_names.index(("R" if n[0] == "L" else "L") + n[1:]) for n in joint_names]
            flip_v, flip_w = np.array([1.0, -1.0, 1.0]), np.array([-1.0, 1.0, -1.0])
            mirrored = np.concatenate([joints[:, perm], jvel[:, perm], lin * flip_v, ang * flip_w, grav * flip_v], -1)
            feats = np.concatenate([feats, mirrored])
            clip = np.concatenate([clip, clip + clip.max() + 1])
            lin = np.concatenate([lin, lin * flip_v])
        # transitions within a clip, at walking speeds (the robot is commanded up to ~0.45 m/s)
        speed = np.linalg.norm(lin[:, :2], axis=1)
        ok = (clip[:-1] == clip[1:]) & (speed[:-1] < max_speed)
        expert = np.concatenate([feats[:-1][ok], feats[1:][ok]], -1).astype(np.float32)
        self.device = device
        self.expert = torch.tensor(expert, device=device)
        mean, std = feats.mean(0), feats.std(0) + 1e-3
        self.mean = torch.tensor(np.concatenate([mean, mean]), dtype=torch.float32, device=device)
        self.std = torch.tensor(np.concatenate([std, std]), dtype=torch.float32, device=device)
        self.dim = expert.shape[1]
        self.net = nn.Sequential(nn.Linear(self.dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 1)).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.grad_penalty = grad_penalty
        self.batch = batch
        self.buffer = torch.zeros(buffer_size, self.dim, device=device)
        self.filled = 0
        self.ptr = 0
        self.last_loss = {"expert": 0.0, "policy": 0.0, "grad_pen": 0.0}
        self.frozen = False  # evaluation, or scoring other policies with a trained discriminator
        print(f"[amp] {len(expert)} expert transitions from {motion_file} ({len(expert) * dt:.0f} s)")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def reward(self, transitions: torch.Tensor) -> torch.Tensor:
        """Style reward in 0..1 (least-squares GAN form): 1 = indistinguishable from the dog."""
        d = self.net(self._norm(transitions)).squeeze(-1)
        return torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)

    def add(self, transitions: torch.Tensor) -> None:
        n = transitions.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.buffer.shape[0]
        self.buffer[idx] = transitions
        self.ptr = int((self.ptr + n) % self.buffer.shape[0])
        self.filled = min(self.filled + n, self.buffer.shape[0])

    def save(self, path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, steps: int = 2) -> None:
        if self.frozen or self.filled < self.batch:
            return
        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(steps):
                pol = self.buffer[torch.randint(0, self.filled, (self.batch,), device=self.device)].clone()
                exp = self.expert[torch.randint(0, len(self.expert), (self.batch,), device=self.device)].clone()
                exp_n = self._norm(exp).requires_grad_(True)
                d_exp = self.net(exp_n)
                d_pol = self.net(self._norm(pol))
                loss_exp = torch.mean(torch.square(d_exp - 1.0))
                loss_pol = torch.mean(torch.square(d_pol + 1.0))
                grad = torch.autograd.grad(d_exp.sum(), exp_n, create_graph=True)[0]
                gp = torch.mean(torch.sum(torch.square(grad), dim=-1))
                loss = 0.5 * (loss_exp + loss_pol) + self.grad_penalty * gp
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.last_loss = {"expert": loss_exp.item(), "policy": loss_pol.item(), "grad_pen": gp.item()}
