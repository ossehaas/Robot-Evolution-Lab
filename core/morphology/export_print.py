"""RobotSpec -> print-ready STL files (laid flat on the bed), a bill of materials and a servo map for firmware.

Generators can attach optional extras to the spec (plain attributes, not part of the RobotSpec interface):
    part_mass    {part name: kg} printed mass (perimeters + infill) instead of volume x effective density
    part_notes   {part name: str} print / assembly hint per part
    servo_ids    {joint name: bus id} for serial bus servos (the servo map then lists ids instead of PWM channels)
    servo_roles  {joint name: str} human-readable servo position
    hardware     list of objects with .item, .qty, .use (screws, inserts, bearings...) for a detailed BOM table
    print_notes  list of str replacing the generic notes
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from . import components as C
from .builder import RobotSpec


def _lay_flat(part) -> trimesh.Trimesh:
    """Move a part into its print frame (plate normal = +z), then drop it onto z = 0 at the bed origin."""
    mesh = part.mesh.copy()
    if part.print_frame is not None:
        mesh.apply_transform(np.linalg.inv(part.print_frame))
    else:
        # a part without a plate (the chassis, feet) prints in its assembled orientation
        pass
    lo = mesh.bounds[0]
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -center[1], -lo[2]])
    return mesh


def write_print_files(spec: RobotSpec, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    stl_dir = out_dir / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)
    part_mass = getattr(spec, "part_mass", None) or {}
    part_notes = getattr(spec, "part_notes", None) or {}

    rows = []
    for part in spec.parts:
        mesh = _lay_flat(part)
        fname = f"{part.name}.stl"
        # STL is conventionally in millimetres
        mm = mesh.copy()
        mm.apply_scale(1000.0)
        mm.export(stl_dir / fname)
        material = C.MATERIALS[part.material]
        if part.name in part_mass:
            grams = 1000.0 * part_mass[part.name]
        else:
            volume = abs(float(mesh.volume)) if mesh.is_volume else float(mesh.convex_hull.volume)
            grams = 1000.0 * volume * material.effective_density
        rows.append(
            {
                "part": part.name,
                "file": f"stl/{fname}",
                "material": material.name,
                "grams": round(grams, 1),
                "size_mm": [round(1000.0 * v, 1) for v in mesh.extents],
                "note": part_notes.get(part.name, ""),
            }
        )

    servo_ids = getattr(spec, "servo_ids", None)
    if servo_ids:
        roles = getattr(spec, "servo_roles", None) or {}
        servo_type = C.SERVOS[spec.servo]
        servo_map = [
            {
                "id": servo_ids[j.name],
                "joint": j.name,
                "role": roles.get(j.name, ""),
                "direction": j.servo_direction,
                "center_ticks": 2048,
                "ticks_per_rad": round(4096 / (2 * math.pi), 2),
                "limit_deg": [round(math.degrees(j.lower), 1), round(math.degrees(j.upper), 1)],
                "max_torque_Nm": round(servo_type.stall_torque, 2),
            }
            for j in spec.joints
            if j.name in servo_ids  # passive spring hinges (flexible back) have no servo
        ]
    else:
        servo_map = [
            {
                "channel": i,
                "joint": j.name,
                "direction": j.servo_direction,
                "center_us": 1500,
                "us_per_rad": round(1000.0 / (C.SERVOS[spec.servo].travel / 2), 1),
                "limit_deg": [round(math.degrees(j.lower), 1), round(math.degrees(j.upper), 1)],
            }
            for i, j in enumerate(spec.joints)
        ]
    (out_dir / "servo_map.json").write_text(json.dumps(servo_map, indent=2))

    printed_mass = sum(r["grams"] for r in rows)
    lines = [
        f"# Bill of materials: {spec.name}",
        "",
        f"Total mass (simulated): **{1000 * spec.total_mass:.0f} g**, of which printed parts ~{printed_mass:.0f} g.",
        f"Servos: {len(servo_map)} x {C.SERVOS[spec.servo].name}; rest height of the chassis centre: {1000 * spec.spawn_height:.0f} mm.",
        "",
        "## Printed parts",
        "",
    ]
    if part_notes:
        lines += ["| Part | Material | Mass (g) | Size on the bed (mm) | File | Notes |", "|---|---|---|---|---|---|"]
    else:
        lines += ["| Part | Material | Mass (g) | Size (mm) | File |", "|---|---|---|---|---|"]
    for r in rows:
        size = " x ".join(f"{v:.0f}" for v in r["size_mm"])
        row = f"| {r['part']} | {r['material']} | {r['grams']} | {size} | `{r['file']}` |"
        lines.append(row + (f" {r['note']} |" if part_notes else ""))
    by_material = defaultdict(float)
    for r in rows:
        by_material[r["material"]] += r["grams"]
    if part_notes:
        lines += ["", "Filament: " + ", ".join(f"{m} ~{g:.0f} g" for m, g in by_material.items()) + " (plus ~15 % for purge and brims)."]
    lines += ["", "## Purchased parts", "", "| Item | Qty |", "|---|---|"]
    for item, qty in spec.extra_bom.items():
        lines.append(f"| {item} | {qty} |")
    hardware = getattr(spec, "hardware", None)
    if hardware:
        uses = defaultdict(list)
        for h in hardware:
            uses[h.item].append(f"{h.qty}x {h.use}")
        lines += ["", "### Where the screws, inserts and bearings go", "", "| Item | Used for |", "|---|---|"]
        for item in sorted(uses):
            lines.append(f"| {item} | {'; '.join(uses[item])} |")
    notes = getattr(spec, "print_notes", None) or [
        "Plates print flat without supports; struts that join two plates at an angle may need supports.",
        "Servo pockets have 0.3 mm clearance per side; tab holes are 1.8 mm pilot holes for M2 self-tapping screws.",
        "Feet print in TPU (95A) for grip; PLA feet are much more slippery, and the simulation models that.",
        "`servo_map.json` lists the PWM channel, direction and limits of every joint for the firmware.",
        "The generator produces draft geometry: check the parts in a slicer before printing.",
    ]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    (out_dir / "BOM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"parts": rows, "servo_map": servo_map}
