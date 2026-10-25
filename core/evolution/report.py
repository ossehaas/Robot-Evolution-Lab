"""Human-readable output of an evolution run: leaderboard.md, fitness.png and gallery.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def write_report(run: Path, individuals: list) -> None:
    done = sorted((i for i in individuals if i.status == "done"), key=lambda i: -i.fitness)
    if not done:
        return

    lines = [
        f"# Evolution run `{run.name}`",
        "",
        f"{len(done)} robots evaluated, {sum(i.status == 'failed' for i in individuals)} failed.",
        "",
        "| Rank | Robot | Gen | Fitness | Speed +x (cm/s) | Falls | Servos | Mass (g) | Topology | Parents | Warm start |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, ind in enumerate(done[:25], 1):
        r = ind.result
        lines.append(
            f"| {rank} | [{ind.name}]({ind.name}/robot/preview.png) | {ind.generation} | {ind.fitness:.3f} | {100 * r['speed_m_s']:.2f} | "
            f"{r['fall_rate']:.2f} | {r['num_servos']} | {1000 * r['mass_kg']:.0f} | `{ind.topology}` | "
            f"{', '.join(ind.parents)} | {'yes' if ind.warm_start_from else 'no'} |"
        )
    lines += ["", "Play the best robot:", "", "```", f"python -m core.evolution.play --run_dir {run / done[0].name}", "```"]
    (run / "leaderboard.md").write_text("\n".join(lines) + "\n")

    # fitness over generations
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    gens = sorted({i.generation for i in done})
    ax.scatter([i.generation for i in done], [i.fitness for i in done], s=18, color="#8a94a6", label="candidate")
    best_so_far, running = [], float("-inf")
    for g in gens:
        running = max(running, max(i.fitness for i in done if i.generation == g))
        best_so_far.append(running)
    ax.plot(gens, best_so_far, color="#e0701a", lw=2.2, label="best so far")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run / "fitness.png")
    plt.close(fig)

    # gallery of the best bodies
    top = [i for i in done[:8] if (run / i.name / "robot" / "preview.png").exists()]
    if top:
        cols = 2
        rows = (len(top) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 3.2), dpi=90, squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, ind in zip(axes.flat, top):
            ax.imshow(mpimg.imread(run / ind.name / "robot" / "preview.png"))
            ax.set_title(f"{ind.name}: fitness {ind.fitness:.3f}", fontsize=11)
        fig.tight_layout()
        fig.savefig(run / "gallery.png")
        plt.close(fig)
