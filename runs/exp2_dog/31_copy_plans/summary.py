"""Tables of step 51 -> results.md (step 50's tables on this folder)."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], str(HERE), "The standing legs copy the plans (step 51)"]
runpy.run_path(str(HERE.parent / "30_level_body" / "summary.py"), run_name="__main__")
