"""Learned terrain memory: a recurrent student that walks on what the chest sensor sees right now, taught by a dog
that walks on the terrain map (core/sim/terrain_map.py).

    lab -m core.evolution.distill_memory --teacher runs/exp2_dog/12_terrain_map/map_depth \\
        --out runs/exp2_dog/12_terrain_map/student_depth

How it works
    * the teacher is a trained map dog: it reads the scan grid from the map with memory (what it saw a second ago
      is still there, placed with the drifting odometry)
    * the student gets only this step's body senses, command and clock, and the scan points the LATEST sensor frame
      covers (no history, no map memory). A GRU memory must keep the rest: the step that has passed under the body,
      where it is now relative to each paw
    * it learns to (1) do what the teacher does and (2) rebuild the perfect height scan from its memory, including
      the ground under its body that no sensor sees. (2) forces the memory to hold the terrain and shows what the dog
      "thinks" the ground looks like
    * DAgger: early on the teacher drives part of the time, later the student drives and is corrected wherever it ends
      up; trained on sequences of seq_len steps (backpropagation through time), hidden state carried across them

Writes student.pt (TorchScript: actions, hidden = student(inputs, hidden)), model_final.pt (weights), result.json
(standard evaluation, and again with sensor blackouts) and a picture of rebuilt vs true terrain on the stairs.
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


class MemoryStudent(nn.Module):
    """inputs -> normalize -> encoder -> GRU cell -> actor (GRU state + encoding) -> actions; decoder (GRU state)
    -> rebuilt height scan."""

    def __init__(self, n_in: int, n_act: int, n_scan: int, hidden: int = 256):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("var", torch.ones(n_in))
        self.register_buffer("count", torch.zeros(()))
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(n_in, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU())
        self.gru = nn.GRUCell(256, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden + 256, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                   nn.Linear(128, n_act))
        self.decoder = nn.Sequential(nn.Linear(hidden, 256), nn.ELU(), nn.Linear(256, n_scan))

    @torch.no_grad()
    def update_normalization(self, x: torch.Tensor) -> None:
        """Running mean / variance over all inputs seen so far (Chan's parallel update)."""
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = m - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + v * n + delta**2 * self.count * n / total) / total)
        self.count.copy_(total)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder((x - self.mean) / torch.sqrt(self.var + 1e-6))
        h = self.gru(z, h)
        return self.actor(torch.cat((h, z), dim=-1)), self.decoder(h), h

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deployment: actions and the new hidden state."""
        a, _, h = self.step(x, h)
        return a, h


def student_inputs(base) -> torch.Tensor:
    """What the student sees this step: body senses, last action, clock, command, and the latest sensor frame on
    the scan grid (heights + covered flags)."""
    return torch.cat((base.obs_frame, base.map_view_obs), dim=-1)


class RecurrentPolicy:
    """Evaluation wrapper (core.evolution.train.evaluate): keeps the hidden state, forgets it when an episode ends."""

    def __init__(self, model: MemoryStudent, base):
        self.model, self.base = model, base
        self.h = torch.zeros(base.num_envs, model.hidden, device=base.device)

    def __call__(self, obs):
        a, self.h = self.model(student_inputs(self.base), self.h)
        return a.clamp(-1.0, 1.0)

    def reset(self, dones):
        self.h = self.h * (~dones.bool()).float().unsqueeze(-1)


def snapshot_picture(shots, cfg, q, half: float, out: Path) -> None:
    """True scan / latest frame / rebuilt from memory, for a few moments on the stairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pat, p, clip = cfg.height_scanner.pattern_cfg, cfg.height_scan_points, cfg.height_scan_clip
    nx, ny = round(pat.size[0] / pat.resolution) + 1, round(pat.size[1] / pat.resolution) + 1
    fig, axes = plt.subplots(len(shots), 3, figsize=(9, 2.2 * len(shots)), squeeze=False)
    extent = (q[:, 0].min().item(), q[:, 0].max().item(), q[:, 1].min().item(), q[:, 1].max().item())
    for row, (t, i, truth, view, rebuilt) in enumerate(shots):
        for col, (title, v, mask) in enumerate((("true ground", truth, None), ("sensor, latest frame", view[:p], view[p:]),
                                                ("rebuilt from memory", rebuilt, None))):
            h = v.clone() * clip * 100
            if mask is not None:
                h[mask < 0.5] = float("nan")
            ax = axes[row, col]
            im = ax.imshow((-h).reshape(ny, nx).flip(0).numpy(), cmap="viridis", vmin=-8, vmax=8, extent=extent)
            ax.axvline(half, color="w", lw=0.5)
            ax.axvline(-half, color="w", lw=0.5)
            ax.set_title(f"{title} (dog {i}, t={t:.1f}s)", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=axes, shrink=0.6, label="ground height [cm], grey = not seen")
    fig.savefig(out, dpi=110)


def main(argv=None) -> int:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description="Teach a recurrent student the terrain map dog's walking.")
    parser.add_argument("--teacher", type=Path, required=True, help="run directory of a terrain-map dog")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=64, help="steps per rollout = BPTT length (64 = 1.3 s)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=4, help="mini-batches per epoch (split over robots)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rebuild_weight", type=float, default=1.0, help="weight of the height-scan rebuild loss")
    parser.add_argument("--teacher_drive_until", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--dark_eval", type=float, default=0.5, help="sensor blackouts per second in the second evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_override", action="append", default=[], help="extra environment setting key=value")
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    args.out.mkdir(parents=True, exist_ok=True)

    from core.evolution.train import build_robot_files
    from core.morphology import Genome, build_robot
    from core.sim.locomotion_env_cfg import apply_overrides, make_env_cfg

    teacher_res = json.loads((args.teacher / "result.json").read_text())
    if not teacher_res.get("terrain_map"):
        raise SystemExit("the teacher must be a terrain-map dog (trained with --terrain_map)")
    genome_path = (args.teacher / "genome.json").resolve()
    genome = Genome.load(genome_path)
    genome.save(args.out / "genome.json")
    usd = build_robot_files(genome_path, args.out / "robot")
    spec = build_robot(genome)
    env_cfg = make_env_cfg(
        spec, str(usd.resolve()).replace("\\", "/"), args.num_envs, True, task="command",
        terrain=teacher_res.get("terrain_arg", "dog_rough"), sensors=teacher_res["sensors"],
        obs_history=teacher_res["obs_history"], height_scan=True, terrain_map=teacher_res["terrain_map"],
    )  # fmt: skip
    overrides = {k: v for k, v in teacher_res.get("env_overrides", {}).items() if not k.startswith("amp_")}
    import ast

    for item in args.env_override:
        key, value = item.split("=", 1)
        overrides[key] = ast.literal_eval(value)
    apply_overrides(env_cfg, overrides)
    env_cfg.curriculum = False
    env_cfg.seed = args.seed
    torch.manual_seed(args.seed)

    from isaaclab.app import launch_simulation

    started = time.time()
    with launch_simulation(env_cfg, args):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from core.evolution.train import evaluate
        from core.sim.locomotion_env import EvoLocomotionEnv

        env = RslRlVecEnvWrapper(EvoLocomotionEnv(env_cfg), clip_actions=None)
        base = env.unwrapped
        device = base.device
        teacher = torch.jit.load(str(args.teacher / "policy.pt"), map_location=device).eval()
        p = env_cfg.height_scan_points
        base.spread_over_terrain_levels()  # every difficulty level, curriculum frozen
        obs, _ = env.reset()
        n_in = student_inputs(base).shape[1]
        model = MemoryStudent(n_in, env.num_actions, p, args.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        n_env, T = base.num_envs, args.seq_len
        h = torch.zeros(n_env, args.hidden, device=device)
        start = torch.ones(n_env, dtype=torch.bool, device=device)  # the next observation begins an episode
        half = spec.genome.body.length / 2
        under = base._map_query[:, 0].abs() <= half
        log = []
        print(f"[memory] student: {n_in} inputs, GRU {args.hidden}, {T}-step sequences, teacher {args.teacher}", flush=True)
        for it in range(args.iterations):
            beta = max(0.0, 1.0 - it / max(1.0, args.teacher_drive_until * args.iterations))
            xs = torch.zeros(T, n_env, n_in, device=device)
            acts = torch.zeros(T, n_env, env.num_actions, device=device)
            truths = torch.zeros(T, n_env, p, device=device)
            starts = torch.zeros(T, n_env, dtype=torch.bool, device=device)
            h0 = h.clone()
            with torch.inference_mode():
                for t in range(T):
                    x = student_inputs(base)
                    target = teacher(obs["policy"]).clamp(-1.0, 1.0)
                    h = h * (~start).float().unsqueeze(-1)
                    a, _, h = model.step(x, h)
                    xs[t], acts[t], truths[t], starts[t] = x, target, base.scan_truth[:, :p], start
                    model.update_normalization(x)
                    drive = torch.rand(n_env, 1, device=device) < beta
                    obs, _, dones, _ = env.step(torch.where(drive, target, a.clamp(-1.0, 1.0)))
                    start = dones.bool()
            h = h.clone()  # out of inference mode for the next rollout
            xs, acts, truths, starts, h0 = xs.clone(), acts.clone(), truths.clone(), starts.clone(), h0.clone()
            act_losses, rebuild_losses, under_err = [], [], []
            chunk = n_env // args.chunks
            for _ in range(args.epochs):
                perm = torch.randperm(n_env, device=device)
                for c in range(args.chunks):
                    idx = perm[c * chunk : (c + 1) * chunk]
                    hc = h0[idx]
                    la = lr_ = 0.0
                    for t in range(T):
                        hc = hc * (~starts[t, idx]).float().unsqueeze(-1)
                        a, rebuilt, hc = model.step(xs[t, idx], hc)
                        la = la + torch.mean(torch.square(a - acts[t, idx]))
                        lr_ = lr_ + torch.mean(torch.square(rebuilt - truths[t, idx]))
                        if t == T - 1:
                            err = (rebuilt.detach() - truths[t, idx]).abs()[:, under].mean()
                            under_err.append(float(err) * env_cfg.height_scan_clip * 100)
                    loss = (la + args.rebuild_weight * lr_) / T
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    act_losses.append(float(la) / T)
                    rebuild_losses.append(float(lr_) / T)
            entry = {"iteration": it, "action_loss": sum(act_losses) / len(act_losses),
                     "rebuild_loss": sum(rebuild_losses) / len(rebuild_losses),
                     "rebuild_err_under_body_cm": sum(under_err) / len(under_err), "teacher_drives": beta}
            log.append(entry)
            if it % 10 == 0 or it == args.iterations - 1:
                print(f"[memory] it {it:4d}/{args.iterations}  action loss {entry['action_loss']:.4f}  rebuild loss "
                      f"{entry['rebuild_loss']:.4f} (under the body {entry['rebuild_err_under_body_cm']:.1f} cm)  "
                      f"teacher drives {beta:.2f}", flush=True)
        train_time = time.time() - started
        model.eval()
        torch.save({"model": model.state_dict(), "n_in": n_in, "n_act": env.num_actions, "n_scan": p,
                    "hidden": args.hidden}, args.out / "model_final.pt")
        scripted = torch.jit.script(model.cpu())
        scripted.save(str(args.out / "student.pt"))
        model.to(device)

        # pictures: rebuilt vs true terrain for dogs on the stairs (before the evaluation resets everything)
        gen = env_cfg.terrain.terrain_generator
        names = list(gen.sub_terrains)
        cum = torch.cumsum(torch.tensor([gen.sub_terrains[n].proportion for n in names]), 0)
        cum = cum / cum[-1]
        col_type = [names[int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0])] for c in range(gen.num_cols)]
        shots = []
        policy = RecurrentPolicy(model, base)
        with torch.inference_mode():
            for step in range(300):
                x = student_inputs(base)
                a, rebuilt, policy.h = model.step(x, policy.h)
                if step in (150, 250):
                    stair = [i for i in range(n_env) if col_type[int(base.terrain.terrain_types[i])].startswith("stairs")
                             and int(base.terrain.terrain_levels[i]) >= 5][:2] or [0]
                    for i in stair:
                        shots.append((step * base.step_dt, i, base.scan_truth[i, :p].cpu(), base.map_view_obs[i].cpu(),
                                      rebuilt[i].cpu()))
                obs, _, dones, _ = env.step(a.clamp(-1.0, 1.0))
                policy.reset(dones)
        snapshot_picture(shots, env_cfg, base._map_query.cpu(), half, args.out / "rebuilt_terrain.png")

        stats = evaluate(env, RecurrentPolicy(model, base), int(base.max_episode_length))
        base.cfg.map_dropout_rate = args.dark_eval
        dark = evaluate(env, RecurrentPolicy(model, base), int(base.max_episode_length))
        result = {
            "name": genome.name,
            "genome": str(genome_path),
            "method": "memory_student",
            "teacher": str(args.teacher),
            "terrain_arg": teacher_res.get("terrain_arg", "dog_rough"),
            "sensors": teacher_res["sensors"],
            "obs_history": teacher_res["obs_history"],
            "height_scan": True,
            "terrain_map": teacher_res["terrain_map"],
            "env_overrides": overrides,
            "iterations": args.iterations,
            "seq_len": T,
            "rebuild_weight": args.rebuild_weight,
            "hidden": args.hidden,
            "num_envs": args.num_envs,
            "train_time_s": round(train_time, 1),
            "loss_curve": log[:: max(1, len(log) // 50)],
            **stats,
            "dark": {k: dark[k] for k in ("fall_rate", "walk_score", "lin_vel_error_m_s", "type_fall_rates") if k in dark},
        }
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[memory] student: walk score {stats['walk_score']:.3f}, falls {stats['fall_rate']:.3f}; with "
              f"blackouts falls {dark['fall_rate']:.3f}")
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
