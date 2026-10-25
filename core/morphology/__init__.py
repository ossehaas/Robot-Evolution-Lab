"""Genome -> robot: simulation USD, URDF, printable STL parts, bill of materials and preview renders."""

from .builder import RobotSpec, build_robot
from .genome import Genome, crossover, mutate, random_genome


def export_all(genome: Genome, out_dir, usd: bool = True, urdf: bool = True, print_files: bool = True, preview: bool = True):
    """Build a genome and write every artefact into ``out_dir``. Returns the RobotSpec."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = build_robot(genome)
    genome.save(out / "genome.json")
    if usd:
        from .export_usd import write_usd

        write_usd(spec, out / f"{spec.name}.usda")
    if urdf:
        from .export_urdf import write_urdf

        write_urdf(spec, out / "urdf" / f"{spec.name}.urdf")
    if print_files:
        from .export_print import write_print_files

        write_print_files(spec, out / "print")
    if preview:
        from .preview import render_preview

        render_preview(spec, out / "preview.png")
    return spec


__all__ = ["Genome", "RobotSpec", "build_robot", "crossover", "export_all", "mutate", "random_genome"]
