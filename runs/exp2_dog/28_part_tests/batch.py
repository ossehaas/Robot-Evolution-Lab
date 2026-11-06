"""Step 48: the paw designs on the test stand (tools/parts/paw_stand.py), one Kit process each. Smoke = the current paw
only, with its table printed; the full run does all designs and writes stand.md.

    lab runs/exp2_dog/28_part_tests/batch.py smoke
    lab runs/exp2_dog/28_part_tests/batch.py full
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
BASE = json.loads((HERE.parent / "26_ring_base" / "genome_ring.json").read_text(encoding="utf-8"))
DESIGNS = {"old ball": dict(paw_housing="tpu"), "now": dict(), "P 3": dict(paw_protrusion=3.0), "P 4.5": dict(paw_protrusion=4.5),
           "P 3 + bevel": dict(paw_protrusion=3.0, paw_bevel=True), "P 4.5 + bevel": dict(paw_protrusion=4.5, paw_bevel=True)}


def run(name: str, out: Path) -> None:
    g = json.loads(json.dumps(BASE))
    g["legs"].update(DESIGNS[name])
    path = HERE / "genomes" / f"{name.replace(' ', '_').replace('+', 'plus').replace('.', '_')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(g, indent=1), encoding="utf-8")
    cmd = [sys.executable, "tools/parts/paw_stand.py", "--genome", str(path), "--name", name, "--out", str(out)]
    for attempt in range(3):  # the CAD build crashes at random now and then (access violation, keep_main): try again
        code = subprocess.run(cmd, cwd=REPO).returncode
        if code == 0:
            return
        print(f"[batch] {name} failed (exit code {code}), attempt {attempt + 1} of 3", flush=True)
    raise SystemExit(code)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        out = HERE / "smoke" / "stand.jsonl"
        out.parent.mkdir(exist_ok=True)
        out.unlink(missing_ok=True)
        run("now", out)
    else:
        out = HERE / "stand.jsonl"
        out.unlink(missing_ok=True)
        for name in DESIGNS:
            run(name, out)
    subprocess.run([sys.executable, str(HERE / "summary.py"), str(out)], cwd=REPO, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
