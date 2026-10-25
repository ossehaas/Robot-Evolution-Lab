"""The four candidate dogs side by side on stairs and on rubble (level 6, heading hold 2)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
RUNS = ["now: D_print_s2=runs/exp2_dog/10_printable_dog/D_print_s2",
        "imitation (style 6)=runs/exp2_dog/11_imitation/amp_w6m_s2",
        "imitation + slow walks, seed 1=runs/exp2_dog/11_imitation/amp_w6m_slow_s1",
        "imitation + slow walks, seed 2=runs/exp2_dog/11_imitation/amp_w6m_slow_s2"]
for terrain in ("stairs_climb", "stairs_descend", "blocks"):
    cmd = [sys.executable, "-m", "core.evolution.compare_video", "--out",
           f"runs/exp2_dog/11_imitation/videos/terrain_{terrain}_l6.mp4", "--terrain_type", terrain, "--level", "6",
           "--heading_hold", "2.0", "--eye", "0.9", "-1.1", "0.45", "--subtitle", f"{terrain}, level 6, heading hold"]
    for r in RUNS:
        cmd += ["--run", r]
    subprocess.run(cmd, cwd=ROOT)
