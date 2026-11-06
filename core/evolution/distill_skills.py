"""One model, many skills (journal step 53, owner 2026-11-05): walking, stairs up and stairs down in one network, taught by
copying the dogs that have those skills. A recognizer outside the model only sends it a signal.

    lab -m core.evolution.distill_skills --up_teacher runs/exp2_dog/29_long_sole/teach_s1 \\
        --down_teacher runs/exp2_dog/31_copy_plans/copy_s1 --arm heads --out runs/exp2_dog/33_one_model/heads_s1

How it works
    * the signal is the environment's last 3 policy inputs (skill_signal_obs): one-hot walk / stairs up / stairs down, for now
      a perfect recognizer, from the simulator's truth
    * --arm none: one output, the signal not used (the control). input: one output, the signal as an input. heads: the signal
      as an input and one output section per skill (later: fine-tune one skill, lock the others)
    * no rewards: the student copies. --up_teacher for walking, stairs up and any tilted dog (it gets up), --down_teacher going
      down. Teachers see the observations without the signal
    * DAgger: early on the teacher drives part of the time, later the student drives and is corrected wherever it ends up
    * a store per kind of sample (walk, up, down, tilted), every mini-batch takes the same number from each: a rare skill is
      practised as much as a common one, and none fades while another improves

Writes policy.pt (TorchScript: actions = policy(obs)), model_final.pt (weights), result.json (the up teacher's recipe, so all
tests run unchanged, plus the signal). The tests give the numbers, there is no evaluation here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch import nn  # noqa: E402

KINDS = ("walk", "up", "down", "tilted")
SIGNAL = 3
TILTED_UP = 0.82  # gravity z of 35 deg, where gated_bc.py hands over to the get-up teacher


class SkillModel(nn.Module):
    """senses -> normalize -> (+ signal) -> shared body -> one output per skill, blended by the signal."""

    def __init__(self, n_in: int, n_act: int, arm: str, dims: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        n_senses = n_in - SIGNAL
        self.register_buffer("mean", torch.zeros(n_senses))
        self.register_buffer("var", torch.ones(n_senses))
        self.register_buffer("count", torch.zeros(()))
        self.use_signal = arm != "none"
        self.n_signal = SIGNAL  # TorchScript cannot read the global
        self.n_heads = SIGNAL if arm == "heads" else 1
        self.n_act = n_act
        layers: list[nn.Module] = []
        last = n_in if self.use_signal else n_senses
        for dim in dims:
            layers += [nn.Linear(last, dim), nn.ELU()]
            last = dim
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, self.n_heads * n_act)

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor) -> None:
        """Running mean / variance over all senses seen so far (Chan's parallel update)."""
        x = obs[:, :-SIGNAL]
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = m - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + v * n + delta**2 * self.count * n / total) / total)
        self.count.copy_(total)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        signal = obs[:, -self.n_signal:]
        x = (obs[:, : -self.n_signal] - self.mean) / torch.sqrt(self.var + 1e-6)
        if self.use_signal:
            x = torch.cat((x, signal), dim=-1)
        out = self.head(self.body(x)).view(-1, self.n_heads, self.n_act)
        if self.n_heads == 1:
            return out[:, 0]
        return (out * signal.unsqueeze(-1)).sum(dim=1)


class Store:
    """A ring buffer per kind of sample (obs, teacher's action)."""

    def __init__(self, capacity: int, n_in: int, n_act: int, device):
        self.capacity = capacity
        self.obs = {k: torch.zeros(capacity, n_in, device=device) for k in KINDS}
        self.act = {k: torch.zeros(capacity, n_act, device=device) for k in KINDS}
        self.pos = dict.fromkeys(KINDS, 0)
        self.size = dict.fromkeys(KINDS, 0)
        self.seen = dict.fromkeys(KINDS, 0)

    def add(self, kind: str, obs: torch.Tensor, act: torch.Tensor) -> None:
        n = min(obs.shape[0], self.capacity)
        if n == 0:
            return
        idx = (self.pos[kind] + torch.arange(n, device=obs.device)) % self.capacity
        self.obs[kind][idx] = obs[:n]
        self.act[kind][idx] = act[:n]
        self.pos[kind] = (self.pos[kind] + n) % self.capacity
        self.size[kind] = min(self.size[kind] + n, self.capacity)
        self.seen[kind] += obs.shape[0]

    def batch(self, kind: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(self.size[kind], (n,), device=self.obs[kind].device)
        return self.obs[kind][idx], self.act[kind][idx]


def kinds_of(signal: torch.Tensor, up_z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Which store each robot's sample goes to; a tilted dog counts as tilted whatever the signal says."""
    tilted = up_z < TILTED_UP
    return {"tilted": tilted, "walk": signal[:, 0].bool() & ~tilted, "up": signal[:, 1].bool() & ~tilted,
            "down": signal[:, 2].bool() & ~tilted}


def teacher_actions(up_teacher, down_teacher, obs: torch.Tensor, kinds: dict[str, torch.Tensor]) -> torch.Tensor:
    """The up teacher, the down teacher where the dog goes down and is not tilted."""
    senses = obs[:, :-SIGNAL]
    target = up_teacher(senses).clamp(-1.0, 1.0)
    return torch.where(kinds["down"].unsqueeze(-1), down_teacher(senses).clamp(-1.0, 1.0), target)


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Teach one model walking, stairs up and stairs down.")
    parser.add_argument("--up_teacher", type=Path, required=True, help="run directory: walking, stairs up, getting up")
    parser.add_argument("--down_teacher", type=Path, required=True, help="run directory: stairs down")
    parser.add_argument("--arm", choices=("none", "input", "heads"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=24, help="steps per rollout")
    parser.add_argument("--updates", type=int, default=16, help="mini-batches per iteration")
    parser.add_argument("--batch_per_kind", type=int, default=1024)
    parser.add_argument("--store", type=int, default=300_000, help="samples kept per kind")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--teacher_drive_until", type=float, default=0.3)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    args.out.mkdir(parents=True, exist_ok=True)

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

    teacher_res = json.loads((args.up_teacher / "result.json").read_text())
    genome_path = (args.up_teacher / "genome.json").resolve()
    genome = Genome.load(genome_path)
    genome.save(args.out / "genome.json")
    usd = build_robot_files(genome_path, args.out / "robot")
    spec = build_robot(genome)
    env_cfg = make_env_cfg(
        spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, True, task="command",
        servo_model=teacher_res["servo_model"], terrain=teacher_res["terrain_arg"], sensors=teacher_res["sensors"],
        obs_history=teacher_res["obs_history"], height_scan=teacher_res.get("height_scan", False),
        terrain_map=teacher_res.get("terrain_map", ""),
    )  # fmt: skip
    overrides = {k: v for k, v in teacher_res.get("env_overrides", {}).items() if not k.startswith("amp_")}
    overrides["skill_signal_obs"] = True
    apply_overrides(env_cfg, overrides)
    env_cfg.curriculum = False
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)

    from isaaclab.app import launch_simulation

    started = time.time()
    with launch_simulation(env_cfg, args):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from core.sim.locomotion_env import EvoLocomotionEnv

        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg), clip_actions=None)
        base = env.unwrapped
        device = base.device
        if base._sg_col is None:
            raise SystemExit(f"no stairs goal tiles in terrain {teacher_res['terrain_arg']}: nothing to tell apart")
        up_teacher = torch.jit.load(str(args.up_teacher / "policy.pt"), map_location=device).eval()
        down_teacher = torch.jit.load(str(args.down_teacher / "policy.pt"), map_location=device).eval()
        base.spread_over_terrain_levels()  # every difficulty level, curriculum frozen
        obs, _ = env.reset()
        n_in = obs["policy"].shape[1]
        model = SkillModel(n_in, env.num_actions, args.arm).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        store = Store(args.store, n_in, env.num_actions, device)
        log = []
        print(f"[skills] arm {args.arm}: {n_in} inputs, {model.n_heads} output section(s), teachers {args.up_teacher.name} "
              f"(walk, up, tilted) and {args.down_teacher.name} (down)", flush=True)
        for it in range(args.iterations):
            beta = max(0.0, 1.0 - it / max(1.0, args.teacher_drive_until * args.iterations))
            with torch.inference_mode():
                for _ in range(args.steps):
                    x = obs["policy"]
                    kinds = kinds_of(x[:, -SIGNAL:], -base.robot.data.projected_gravity_b.torch[:, 2])
                    target = teacher_actions(up_teacher, down_teacher, x, kinds)
                    for kind, mask in kinds.items():
                        store.add(kind, x[mask], target[mask])
                    model.update_normalization(x)
                    a = model(x).clamp(-1.0, 1.0)
                    drive = torch.rand(base.num_envs, 1, device=device) < beta
                    obs, _, _, _ = env.step(torch.where(drive, target, a))
            ready = [k for k in KINDS if store.size[k] > 0]
            losses = dict.fromkeys(ready, 0.0)
            for _ in range(args.updates):
                batches = {k: store.batch(k, args.batch_per_kind) for k in ready}
                errors = {k: torch.mean(torch.square(model(b[0].clone()) - b[1].clone())) for k, b in batches.items()}
                loss = sum(errors.values()) / len(errors)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for k, e in errors.items():
                    losses[k] += float(e.detach()) / args.updates
            entry = {"iteration": it, "teacher_drives": beta, "loss": losses, "seen": dict(store.seen)}
            log.append(entry)
            if it % 10 == 0 or it == args.iterations - 1:
                text = "  ".join(f"{k} {v:.4f}" for k, v in losses.items())
                print(f"[skills] it {it:4d}/{args.iterations}  loss {text}  teacher drives {beta:.2f}  seen {store.seen}", flush=True)
        train_time = time.time() - started
        model.eval()
        torch.save({"model": model.state_dict(), "n_in": n_in, "n_act": env.num_actions, "arm": args.arm}, args.out / "model_final.pt")
        torch.jit.script(model.cpu()).save(str(args.out / "policy.pt"))
        result = {
            **teacher_res,
            "method": "skill_model",
            "arm": args.arm,
            "up_teacher": str(args.up_teacher),
            "down_teacher": str(args.down_teacher),
            "env_overrides": overrides,
            "iterations": args.iterations,
            "num_envs": args.num_envs,
            "train_time_s": round(train_time, 1),
            "checkpoint": str((args.out / "model_final.pt").resolve()),
            "loss_curve": log[:: max(1, len(log) // 50)],
            "seen": store.seen,
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[skills] done in {train_time / 60:.0f} min, final loss {log[-1]['loss']}, seen {store.seen}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
