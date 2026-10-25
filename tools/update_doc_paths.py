"""One-off after tools/reorganize_runs.py: rewrite runs/... references in the documentation to the new layout."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reorganize_runs import GROUPS, ROOT, RUNS  # noqa: E402

SCRIPTS = {"dog_walk/debug_fall.py", "dog_walk/measure_knees.py", "dog_walk/measure_power.py", "dog_walk/mirror_test.py",
           "dog_stairs/scan_check.py", "dog_jump/jump_check.py", "dog_capability/leg_reach.py", "dog_capability/jump_test.py"}
DOCS = [ROOT / "README.md", ROOT / ".documentation/project-journal.md", ROOT / ".documentation/roadmap.md",
        ROOT / ".agents/project-info.md", ROOT / "prototypes/puppy/README.md", ROOT / "core/morphology/dog_check.py",
        ROOT / "core/morphology/dog_views.py"]
PATTERN = re.compile(r"runs([/\\])([A-Za-z0-9_.\-]+)((?:[/\\][A-Za-z0-9_.\-@]+)*)")


def new_path(top: str, rest: list[str]) -> str | None:
    rel = "/".join([top, *rest])
    if rel in SCRIPTS:
        return "tools/dog/" + rest[-1]
    if top not in GROUPS:
        return None
    candidate = "runs/" + GROUPS[top] + ("/" + "/".join(rest) if rest else "")
    if (ROOT / candidate).exists() or len(rest) < 3 or rest[1] != "videos":
        return candidate
    salvaged = f"runs/{GROUPS[top].split('/')[0]}/{GROUPS[top].split('/')[1]}/videos/{rest[0]}__{rest[-1]}"
    return salvaged if (ROOT / salvaged).exists() else candidate


missing = set()
for doc in DOCS:
    if not doc.exists():
        continue
    text = doc.read_text(encoding="utf-8")

    def repl(m):
        sep, top, tail = m.group(1), m.group(2), m.group(3)
        rest = [p for p in re.split(r"[/\\]", tail) if p]
        out = new_path(top, rest)
        if out is None:
            return m.group(0)
        if not (ROOT / out.split("@")[0]).exists():
            missing.add(out)
        return out.replace("/", sep) if sep == "\\" else out

    new = PATTERN.sub(repl, text)
    if new != text:
        doc.write_text(new, encoding="utf-8")
        print(f"updated {doc.relative_to(ROOT) if doc.is_relative_to(ROOT) else doc.name}")
print(f"\nreferences to paths that no longer exist ({len(missing)}; removed runs, see runs/INDEX.md):")
for m in sorted(missing):
    print("  ", m)
