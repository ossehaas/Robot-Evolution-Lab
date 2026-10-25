# Environment setup: Isaac Lab 3.0 (EA) on Windows 11 + RTX 5080 Laptop (Blackwell, sm_120)

Last rebuilt: 2026-09-30. This replaces the previous guide (conda env "isaaclab" with Isaac Sim 6.0.0.0, a
numpy<2 downgrade, a PyTorch nightly and hand-patched site-packages). None of those workarounds are needed.

## What is installed

| Component | Version | Notes |
|---|---|---|
| Isaac Lab | tag `v3.0.0-EA` (local branch `evolution-v3.0.0-EA`) | `IsaacLab/` checkout |
| Isaac Sim | 6.1.0.0 (pip) | pinned by Isaac Lab's `uv.lock` |
| Python | 3.12 (managed by uv) | env lives in `IsaacLab/.venv` |
| PyTorch | 2.11.0+cu128 | stable wheel, includes sm_120 kernels for the RTX 5080 |
| numpy / warp | 2.x / 1.16.0 | as Isaac Lab requires |
| RL libraries | rsl-rl-lib 5.x, skrl 2.x | via the `rsl-rl` and `skrl` extras |
| Video | moviepy + imageio-ffmpeg | via the `video` extra |

Everything comes from Isaac Lab's lockfile, so the install is reproducible and nothing in site-packages is
patched. `uv pip check` reports a few version mismatches (numpy, newton, pywin32); these are overrides NVIDIA
deliberately pins in the lockfile and are expected.

## Prerequisites

- A recent NVIDIA driver (592.01 is installed and works)
- Windows long paths enabled (already on). If ever needed, in an elevated PowerShell:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
- uv (installed via winget): `winget install astral-sh.uv`
- ~40 GB free disk space (Isaac Sim is large; uv hardlinks from its cache, so do not use UV_LINK_MODE=copy)

## Install / rebuild

```powershell
cd <repository folder>\IsaacLab
git fetch --tags
git checkout -B evolution-v3.0.0-EA v3.0.0-EA
$env:UV_HTTP_TIMEOUT = "900"      # the NVIDIA index sometimes stalls on the large wheels
uv sync --extra isaacsim --extra rsl-rl --extra skrl --extra video
```

The first `uv sync` downloads ~15 GB (about 15-30 minutes). If a download times out, run the same command
again: finished downloads are cached.

Note: `uv sync` makes the environment match the lockfile exactly. Anything installed by hand with
`uv pip install` is removed by the next `uv sync`; add packages as extras instead.

## Verify

```bat
cd <repository folder>
lab test_sim.py
```

Expected: `PASS: cube rests at 0.0250 m`. For a full RL check, train the official cartpole (~15 s):

```bat
IsaacLab\.venv\Scripts\isaaclab.exe train --rl_library rsl_rl --task Isaac-Cartpole-Direct --max_iterations 30 physics=isaacsim_physx
```

## Things that are easy to trip over

- There is no `--headless` flag any more: runs are headless unless you pass `--viz kit`.
- The first launch with rendering compiles shaders and can take ~4 minutes; later launches take seconds.
- Do not import `pxr` (standalone USD) in a process before Isaac Sim starts: Isaac Sim then fails to load its
  own USD libraries ("Could not load the dynamic library ... usd_tf.dll"). Configuration classes are safe to
  import before launch; environment and asset classes (`DirectRLEnv`, `Articulation`) are not. Our code keeps
  them in separate modules (`core/sim/locomotion_env_cfg.py` vs `locomotion_env.py`) and builds USD files in
  a subprocess.
- Articulation data are warp-backed `ProxyArray`s: use `.torch` (e.g. `robot.data.joint_pos.torch`).
- Joint position targets: `robot.actuators.target_command.set_position_index(value=...)`
  (`set_joint_position_target*` is deprecated in 3.0).
- `sim.reset()` does not apply the configured initial state; write root and joint state explicitly.

## Old environment

The previous conda environment `isaaclab` (%USERPROFILE%\anaconda3\envs\isaaclab, Isaac Sim 6.0.0.0 with
patched files) is no longer used and can be removed to free ~20 GB:

```powershell
conda env remove -n isaaclab
```
