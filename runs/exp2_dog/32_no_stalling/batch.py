"""Stalling costs on a descent (journal step 52, owner 2026-11-03: going down the dog waits, like a grid world with no cost of
time; waiting should cost, so that it goes down and balances meanwhile).

Measured: on 4 / 15 down almost no dog arrives and none falls; step 42 found +7.3/s at the edge against +8.4/s on the stairs.
A flat cost of time would not do: goals do not end the episode, a hard fall does. From teach_s1, seeds 1 and 2, step 49's recipe:
  stairs_stall_weight 6/s   after 2 s without coming 2 cm closer to the goal, only on stairs with a descent plan (plans_down_m15)
  stairs_margin_weight 2/s  step 42's balance reward: the centre of gravity inside the paws on the ground
  mid-climb starts          half of the descents, from the same plans
Crouching on the stairs is already allowed (stairs_goal_height_scale 0.3). The first thing to read: more steps down, without
more falls? Tests as step 50 + descent_terms. Videos teach_s1 / nostall_s1 / nostall_s2. Smoke first. ~6.5 h.

    lab runs/exp2_dog/32_no_stalling/batch.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS = HERE / "plans_down_m15"
_spec = importlib.util.spec_from_file_location("level_body_batch", HERE.parent / "30_level_body" / "batch.py")
s50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s50)
s47, s49, O, LAB, run, has = s50.s47, s50.s49, s50.O, s50.LAB, s50.run, s50.has
s50.HERE = s47.HERE = HERE
s50.RUNS = [("nostall_s1", 1), ("nostall_s2", 2)]
s50.LABEL, s50.DIRECTION = "no stalling", -1
s50.VIDEOS = [("down_r3_t15", "goal_stairs_t15", 2, "down 3 cm steps, 15 cm treads"), ("down_r3_t9", "goal_stairs_t9", 2, "down 3 cm steps, 9 cm treads"),
              ("down_r4_t12", "goal_stairs_t12", 3, "down 4 cm steps, 12 cm treads (no plan, waiting is free)"),
              ("down_r5_t20", "goal_stairs_t20", 4, "down 5 cm steps, 20 cm treads (no plan, waiting is free)")]
base_test = s47.test


def train_cmd(out: Path, seed: int, num_envs: int = 2048, iterations: int = s47.ITERATIONS) -> list[str]:
    cmd = s49.train_cmd(out, "teach", seed, num_envs, iterations)
    cmd[cmd.index("--init_checkpoint") + 1] = str(s50.START / "model_final.pt")
    return cmd + [O + f"stairs_rsi_plans={PLANS.as_posix()}", O + "stairs_rsi_prob=0.5", O + "stairs_stall_weight=6.0", O + "stairs_margin_weight=2.0"]


def test(name: str) -> None:
    base_test(name)
    if not has(HERE / "descent_terms.jsonl", name):
        run([LAB, "tools/dog/descent_terms.py", str(HERE / name), "--out", str(HERE / "descent_terms.jsonl")], HERE / "tests.log")


def smoke() -> bool:
    out = HERE / "smoke"
    out.mkdir(exist_ok=True)
    log = out / "smoke.log"
    code = run(train_cmd(out / "train", 1, 64, 4), log)
    if code != 0 or not (out / "train" / "result.json").exists():
        return s50.fail(f"smoke (training): exit code {code}, see smoke/smoke.log")
    text = log.read_text(encoding="utf-8", errors="ignore")
    for key in ("stalling costs on", "sg_stall", "sg_margin", "mid-climb starts"):
        if key not in text:
            return s50.fail(f"smoke: {key!r} never logged, see smoke/smoke.log")
    res = json.loads((out / "train" / "result.json").read_text())
    if res.get("env_overrides", {}).get("stairs_stall_weight") != 6.0:
        return s50.fail("smoke: stairs_stall_weight not recorded in result.json")
    (out / "ok").write_text("ok")
    return True


s47.test, s50.train_cmd, s50.smoke = test, train_cmd, smoke

if __name__ == "__main__":
    if not any(json.loads(f.read_text()).get("feasible") for f in PLANS.glob("down_r*_t*.json")):
        s50.fail(f"no descent plans in {PLANS}")
        sys.exit(1)
    sys.exit(s50.main())
