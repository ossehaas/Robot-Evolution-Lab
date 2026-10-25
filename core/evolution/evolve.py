"""Co-evolution of bodies and brains: a (mu + lambda) genetic algorithm over robot genomes.

Each candidate body is trained from scratch (or warm-started from its parent when the joint structure is
identical, i.e. Lamarckian inheritance of the learned gait) and scored by how far its policy walks.

    python -m core.evolution.evolve --run runs/evo_001 --population 12 --generations 10 --iterations 200
    python -m core.evolution.evolve --run runs/evo_001 --resume          # continue after a stop / crash

The run directory holds everything: state.json (every individual ever evaluated), one folder per
candidate (genome, robot files, checkpoints, result.json), leaderboard.md, fitness.png and gallery.png.
This process never imports the simulator; every candidate is trained in its own subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.morphology.genome import Genome, crossover, mutate, random_genome  # noqa: E402


@dataclass
class Individual:
    name: str
    generation: int
    parents: list[str]
    genome_file: str
    topology: str
    fitness: float | None = None
    result: dict = field(default_factory=dict)
    warm_start_from: str | None = None
    status: str = "pending"  # pending | done | failed


def genome_hash(genome: Genome) -> str:
    data = genome.to_dict()
    data.pop("name", None)
    data.pop("lineage", None)
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]


def score(result: dict, servo_cost: float) -> float:
    """Fitness, minus a small optional penalty per servo (cheaper robots win ties).

    forward task: mean speed along +x [m/s]; command task: walk score in [0, 1] (standing still scores 0;
    results from before the walk score existed fall back to the tracking score).
    """
    if result.get("task") == "goal":
        base = result["goals_per_minute"] / 10.0 - result["collision_fraction"] - 0.2 * result["fall_rate"]
    elif result.get("task") == "command":
        base = result.get("walk_score", result["tracking_score"]) - 0.2 * result["fall_rate"]
    else:
        base = result["speed_m_s"] - 0.02 * result["fall_rate"]
    return base - servo_cost * result["num_servos"]


class Evolution:
    def __init__(self, args):
        self.args = args
        self.run = args.run
        self.run.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(args.seed)
        self.pop: dict[str, Individual] = {}
        self.hashes: dict[str, str] = {}
        self.generation = 0
        if args.resume and (self.run / "state.json").exists():
            self._load()

    # ------------------------------------------------------------------ persistence

    def _save(self) -> None:
        state = {
            "generation": self.generation,
            "args": {k: str(v) for k, v in vars(self.args).items()},
            "individuals": [asdict(i) for i in self.pop.values()],
        }
        tmp = self.run / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.run / "state.json")

    def _load(self) -> None:
        state = json.loads((self.run / "state.json").read_text())
        self.generation = state["generation"]
        for data in state["individuals"]:
            ind = Individual(**data)
            self.pop[ind.name] = ind
            self.hashes[genome_hash(Genome.load(ind.genome_file))] = ind.name
        self.rng.seed(f"{self.args.seed}-{self.generation}")
        print(f"resumed at generation {self.generation} with {len(self.pop)} individuals")

    # ------------------------------------------------------------------ population management

    def _add(self, genome: Genome, parents: list[str], warm: str | None = None) -> Individual | None:
        h = genome_hash(genome)
        if h in self.hashes:
            return None  # identical body already evaluated
        genome.name = f"g{self.generation:02d}_{len(self.pop):03d}"
        cdir = self.run / genome.name
        cdir.mkdir(parents=True, exist_ok=True)
        genome.save(cdir / "genome.json")
        ind = Individual(genome.name, self.generation, parents, str(cdir / "genome.json"), genome.topology_key(), warm_start_from=warm)
        self.pop[ind.name] = ind
        self.hashes[h] = ind.name
        return ind

    def evaluated(self) -> list[Individual]:
        return sorted((i for i in self.pop.values() if i.status == "done"), key=lambda i: -i.fitness)

    def seed_population(self) -> None:
        for path in self.args.seed_genomes:
            self._add(Genome.load(path), parents=["seed"])
        while sum(1 for i in self.pop.values() if i.generation == 0) < self.args.population:
            self._add(random_genome(self.rng), parents=["random"])

    def select_elite(self) -> list[Individual]:
        """Parents for the next generation: the best of each body type first, then the best of the rest.

        Without this the population collapses onto near-copies of one design within a generation or two.
        """
        size = self.args.population // 2
        ranked = self.evaluated()
        elite, seen = [], set()
        for ind in ranked:
            if ind.topology not in seen:
                elite.append(ind)
                seen.add(ind.topology)
        elite = sorted(elite, key=lambda i: -i.fitness)[: max(1, size // 2 + 1)]
        for ind in ranked:
            if len(elite) >= size:
                break
            if ind not in elite:
                elite.append(ind)
        return elite

    def breed(self) -> None:
        elite = self.select_elite()
        if not elite:
            raise RuntimeError("no successfully evaluated individuals to breed from")
        num_children = self.args.population - len(elite)
        attempts = 0
        made = 0
        while made < num_children and attempts < 50 * num_children:
            attempts += 1
            a = self._tournament(elite)
            if self.rng.random() < self.args.crossover_rate and len(elite) > 1:
                b = self._tournament([e for e in elite if e is not a])
                child = crossover(Genome.load(a.genome_file), Genome.load(b.genome_file), self.rng)
                child = mutate(child, self.rng)
                parents = [a.name, b.name]
            else:
                child = mutate(Genome.load(a.genome_file), self.rng)
                parents = [a.name]
            # Lamarckian warm start when the joint structure matches a parent that has a checkpoint
            warm = None
            for pname in parents:
                p = self.pop[pname]
                if p.topology == child.topology_key() and p.result.get("checkpoint") and not self.args.no_warm_start:
                    warm = p.result["checkpoint"]
                    break
            if self._add(child, parents, warm) is not None:
                made += 1

    def _tournament(self, candidates: list[Individual], k: int = 3) -> Individual:
        picks = self.rng.sample(candidates, min(k, len(candidates)))
        return max(picks, key=lambda i: i.fitness)

    # ------------------------------------------------------------------ evaluation

    def _train_command(self, ind: Individual) -> list[str]:
        iterations = self.args.iterations
        if ind.warm_start_from:
            iterations = max(int(iterations * self.args.warm_fraction), 20)
        cmd = [
            sys.executable, "-m", "core.evolution.train",
            "--genome", ind.genome_file,
            "--out", str(Path(ind.genome_file).parent),
            "--iterations", str(iterations),
            "--num_envs", str(self.args.num_envs),
            "--physics", self.args.physics,
            "--task", self.args.task,
            "--sensors", self.args.sensors,
            "--obs_history", str(self.args.obs_history),
            "--terrain", self.args.terrain,
            "--seed", str(self.args.seed),
            "--pretrain_flat", str(self.args.pretrain_flat),
        ]  # fmt: skip
        if ind.warm_start_from:
            cmd += ["--init_checkpoint", ind.warm_start_from]
        return cmd

    def _collect(self, ind: Individual, returncode: int, started: float) -> None:
        cdir = Path(ind.genome_file).parent
        result_file = cdir / "result.json"
        if returncode == 0 and result_file.exists():
            ind.result = json.loads(result_file.read_text())
            ind.fitness = score(ind.result, self.args.servo_cost)
            ind.status = "done"
            r = ind.result
            if "walk_score" in r:
                detail = (f"walk {r['walk_score']:.3f}, progress {r['progress_ratio']:.2f}, "
                          f"velocity error {100 * r['lin_vel_error_m_s']:.1f} cm/s")
            elif "tracking_score" in r:
                detail = f"tracking {r['tracking_score']:.3f}, velocity error {100 * r['lin_vel_error_m_s']:.1f} cm/s"
            else:
                detail = f"{100 * r['speed_m_s']:.2f} cm/s along +x"
            print(f"    -> {ind.name}: {detail}, falls {r['fall_rate']:.2f}, fitness {ind.fitness:.4f} "
                  f"({time.time() - started:.0f} s)", flush=True)
        else:
            ind.status = "failed"
            ind.fitness = float("-inf")
            print(f"    -> {ind.name}: FAILED (exit {returncode}), see {cdir / 'train.log'}", flush=True)
        self._save()

    def evaluate_pending(self) -> None:
        """Train all pending candidates, up to --parallel at a time (each in its own simulator process)."""
        pending = [i for i in self.pop.values() if i.status == "pending"]
        running: list[tuple[Individual, subprocess.Popen, object, float]] = []
        for n, ind in enumerate(pending):
            while len(running) >= max(1, self.args.parallel):
                running = self._reap(running, block=True)
            cdir = Path(ind.genome_file).parent
            cmd = self._train_command(ind)
            print(f"[gen {self.generation}] ({n + 1}/{len(pending)}) training {ind.name} [{ind.topology}] "
                  f"{'warm' if ind.warm_start_from else 'cold'} {cmd[cmd.index('--iterations') + 1]} it", flush=True)
            log = open(cdir / "train.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            running.append((ind, proc, log, time.time()))
            if self.args.parallel > 1:
                time.sleep(self.args.stagger_s)  # Isaac Sim start-up is the memory peak: do not start all at once
        while running:
            running = self._reap(running, block=True)

    def _reap(self, running, block: bool):
        """Collect finished trainings; with block=True wait until at least one has finished."""
        while True:
            still = []
            for ind, proc, log, started in running:
                if proc.poll() is None:
                    still.append((ind, proc, log, started))
                else:
                    log.close()
                    self._collect(ind, proc.returncode, started)
            if len(still) < len(running) or not block or not still:
                return still
            time.sleep(2.0)

    def record_video(self, ind: Individual) -> None:
        """Captioned tour of the generation's best body (on stairs for rough terrain) in <run>/videos/."""
        name = f"gen{self.generation:02d}_{ind.name}.mp4"
        if (self.run / "videos" / name).exists():
            return
        cdir = Path(ind.genome_file).parent
        cmd = [sys.executable, "-m", "core.evolution.play", "--run_dir", str(cdir), "--video", "--num_envs", "16",
               "--out_name", name]
        if self.args.terrain == "rough":
            cmd += ["--level", "4", "--terrain_type", "stairs_climb", "--num_envs", "1"]
        with open(cdir / "video.log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        made = cdir / "videos" / name
        if made.exists():
            (self.run / "videos").mkdir(exist_ok=True)
            made.replace(self.run / "videos" / name)
            print(f"    video: {self.run / 'videos' / name}", flush=True)

    # ------------------------------------------------------------------ reporting

    def report(self) -> None:
        from core.evolution.report import write_report

        write_report(self.run, list(self.pop.values()))

    def loop(self) -> None:
        if not self.pop:
            self.seed_population()
            self._save()
        while True:
            self.evaluate_pending()
            self.report()
            best = self.evaluated()[0] if self.evaluated() else None
            if best:
                print(f"== generation {self.generation}: best {best.name}, fitness {best.fitness:.4f} ==", flush=True)
                if self.args.video:
                    self.record_video(best)
            if self.generation + 1 >= self.args.generations:
                break
            self.generation += 1
            self.breed()
            self._save()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evolve robot bodies and train their gaits.")
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200, help="PPO iterations per new candidate")
    parser.add_argument("--warm_fraction", type=float, default=0.5, help="iterations for warm-started children")
    parser.add_argument("--pretrain_flat", type=int, default=0, help="rough terrain: flat-ground iterations first (cold)")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--physics", choices=("physx", "newton"), default="physx")
    parser.add_argument("--task", choices=("forward", "command", "goal"), default="forward", help="what fitness measures")
    parser.add_argument("--terrain", choices=("flat", "rough", "obstacles", "obstacles_rough"), default="flat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_genomes", type=Path, nargs="*", default=[], help="genomes to include in generation 0")
    parser.add_argument("--crossover_rate", type=float, default=0.3)
    parser.add_argument("--servo_cost", type=float, default=0.0, help="fitness penalty per servo [m/s]")
    parser.add_argument("--no_warm_start", action="store_true")
    parser.add_argument("--sensors", choices=("ideal", "imu", "imu_flow"), default="ideal")
    parser.add_argument("--obs_history", type=int, default=1)
    parser.add_argument("--video", action="store_true", help="record the best body on stairs after each generation")
    parser.add_argument("--parallel", type=int, default=1, help="candidates trained at the same time (needs RAM/VRAM)")
    parser.add_argument("--stagger_s", type=float, default=90.0, help="delay between parallel start-ups [s]")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.run = args.run.resolve()
    Evolution(args).loop()


if __name__ == "__main__":
    main()
