"""RobotSpec -> URDF (with STL visual meshes), for ROS tooling, MuJoCo/PyBullet or other simulators.

URDF has no capsule primitive, so capsules are written as a cylinder plus two end spheres.
"""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from .builder import RobotSpec


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> URDF roll/pitch/yaw (fixed XYZ)."""
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return float(roll), float(pitch), float(yaw)


def _origin(parent: ET.Element, xyz, R=None) -> None:
    attrs = {"xyz": " ".join(f"{v:.6f}" for v in xyz)}
    if R is not None:
        attrs["rpy"] = " ".join(f"{v:.6f}" for v in _rpy(R))
    ET.SubElement(parent, "origin", attrs)


def write_urdf(spec: RobotSpec, path: str | Path) -> Path:
    path = Path(path)
    mesh_dir = path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", {"name": spec.name})

    colors = {}
    for link in spec.links:
        for vis in link.visuals:
            colors.setdefault(tuple(round(c, 3) for c in vis.color), f"color{len(colors)}")
    for rgb, name in colors.items():
        mat = ET.SubElement(robot, "material", {"name": name})
        ET.SubElement(mat, "color", {"rgba": f"{rgb[0]} {rgb[1]} {rgb[2]} 1"})

    for link in spec.links:
        el = ET.SubElement(robot, "link", {"name": link.name})
        inertial = ET.SubElement(el, "inertial")
        _origin(inertial, link.com)
        ET.SubElement(inertial, "mass", {"value": f"{link.mass:.6f}"})
        I = link.inertia
        ET.SubElement(
            inertial,
            "inertia",
            {k: f"{v:.9f}" for k, v in zip(("ixx", "ixy", "ixz", "iyy", "iyz", "izz"), (I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]))},
        )
        # one STL per colour keeps the file count low
        by_color: dict[tuple, list[trimesh.Trimesh]] = {}
        for vis in link.visuals:
            by_color.setdefault(tuple(round(c, 3) for c in vis.color), []).append(vis.mesh)
        for rgb, meshes in by_color.items():
            fname = f"{link.name}_{colors[rgb]}.stl"
            trimesh.util.concatenate(meshes).export(mesh_dir / fname)
            visual = ET.SubElement(el, "visual")
            _origin(visual, (0, 0, 0))
            geom = ET.SubElement(visual, "geometry")
            ET.SubElement(geom, "mesh", {"filename": f"meshes/{fname}"})
            ET.SubElement(visual, "material", {"name": colors[rgb]})
        for col in link.collisions:
            R, t = col.transform[:3, :3], col.transform[:3, 3]
            if col.kind == "capsule":
                r, length = col.size
                pieces = [("cylinder", t, {"radius": f"{r:.6f}", "length": f"{length:.6f}"})]
                for s in (-1, 1):
                    pieces.append(("sphere", t + s * R[:, 2] * length / 2, {"radius": f"{r:.6f}"}))
            elif col.kind == "box":
                pieces = [("box", t, {"size": " ".join(f"{v:.6f}" for v in col.size)})]
            else:
                pieces = [("sphere", t, {"radius": f"{col.size[0]:.6f}"})]
            for kind, pos, attrs in pieces:
                c = ET.SubElement(el, "collision")
                _origin(c, pos, R)
                ET.SubElement(ET.SubElement(c, "geometry"), kind, attrs)

    links = spec.link_map
    for joint in spec.joints:
        el = ET.SubElement(robot, "joint", {"name": joint.name, "type": "revolute"})
        _origin(el, joint.origin - links[joint.parent].origin)
        ET.SubElement(el, "parent", {"link": joint.parent})
        ET.SubElement(el, "child", {"link": joint.child})
        ET.SubElement(el, "axis", {"xyz": " ".join(f"{v:.6f}" for v in joint.axis)})
        ET.SubElement(
            el,
            "limit",
            {"lower": f"{joint.lower:.4f}", "upper": f"{joint.upper:.4f}", "effort": f"{joint.effort:.4f}", "velocity": f"{joint.velocity:.4f}"},
        )

    xml = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    path.write_text(xml)
    return path
