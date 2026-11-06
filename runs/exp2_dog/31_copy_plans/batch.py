"""The standing legs copy the plans (journal step 51, owner 2026-11-03: with a paw going up, the leg on the lower step must
stretch to level the body; on 4 / 9 it does not and the dog flips onto its back).

Step 50's rewards did not teach it (plan_track 0.08 against ~20 per episode of the others). As step 50 from teach_s1, seeds 1
and 2, plus --bc_plan_copy 1.0: in the plan zone the policy copies the plan's joint pose for the legs whose paw is down, in
epochs of its own. Tests as step 50; the lunge test now also has the body's height over the rear paws against the plan.
Videos teach_s1 / copy_s1 / copy_s2. Smoke first. ~6.5 h.

    lab runs/exp2_dog/31_copy_plans/batch.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("level_body_batch", HERE.parent / "30_level_body" / "batch.py")
s50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s50)
s50.HERE = s50.s47.HERE = HERE
s50.RUNS = [("copy_s1", 1), ("copy_s2", 2)]
s50.LABEL = "copies the plan"
level_cmd, level_smoke = s50.train_cmd, s50.smoke


def train_cmd(out: Path, seed: int, num_envs: int = 2048, iterations: int = s50.s47.ITERATIONS) -> list[str]:
    return level_cmd(out, seed, num_envs, iterations) + ["--bc_plan_copy", "1.0"]


def smoke() -> bool:
    if not level_smoke():
        return False
    res = json.loads((HERE / "smoke" / "train" / "result.json").read_text())
    text = (HERE / "smoke" / "smoke.log").read_text(encoding="utf-8", errors="ignore")
    if res.get("bc_plan_copy") != 1.0 or "bc_plan" not in text or not res.get("env_overrides", {}).get("expert_target_obs"):
        (HERE / "smoke" / "ok").unlink()
        return s50.fail("smoke: the plan copying is not on or not logged, see smoke/smoke.log")
    return True


s50.train_cmd, s50.smoke = train_cmd, smoke

if __name__ == "__main__":
    HERE.mkdir(exist_ok=True)
    sys.exit(s50.main())
