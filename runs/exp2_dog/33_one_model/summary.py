"""Tables of step 53 -> results.md (step 50's tables on this folder, with the down teacher copy_s1 for reference)."""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], str(HERE), "One model, many skills (step 53)", str(HERE.parent / "31_copy_plans" / "copy_s1")]
runpy.run_path(str(HERE.parent / "30_level_body" / "summary.py"), run_name="__main__")
