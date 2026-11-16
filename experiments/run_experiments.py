"""Run the paired, fixed-budget study and write the data and figures for Sections 4–5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ga_models import Config, NUTRIENTS, Person, UL
from meal_ga import MealGA, load_foods

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, *range(101, 110))
SALMON_ID = "3053"
TIER = {"checked_no_excess": 0, "not_fully_assessed": 1, "exceeded": 2}
STYLES = {
    "default_ea": ("Default EA", "#006d77", "-"),
    "random_search": ("Random search", "#546579", "--"),
    "no_crossover": ("No crossover", "#974b24", "-."),
    "no_portion_mutation": ("No portion mutation", "#6553a3", ":"),
    "no_salmon_ea": ("EA without salmon", "#006d77", "-"),
    "no_salmon_random": ("Random without salmon", "#546579", "--"),
}
TOL = 1e-9


def run_random_search(ga):
    """Generate independent feasible plans in equally sized evaluation batches."""
    incumbent = None
    history = []
    size = ga.config.population_size
    for generation in range(ga.config.generations + 1):
        plans = [ga.initialize() for _ in range(size)]
        evaluations = ga.evaluate_population(plans)
        i = min(range(size), key=lambda j: evaluations[j].key)
        if incumbent is None or evaluations[i].key < incumbent[1].key:
            incumbent = plans[i], evaluations[i]
        history.append({
            "generation": generation,
            "best_score": evaluations[i].base_score,
            "best_fitness": evaluations[i].fitness,
            "best_ul_status": evaluations[i].ul_status,
            "best_excess": evaluations[i].excess,
            "max_score": max(e.base_score for e in evaluations),
            "mean_score": mean(e.base_score for e in evaluations),
            "exceeded_fraction": mean(e.ul_status == "exceeded" for e in evaluations),
            "unassessed_fraction": mean(e.ul_status == "not_fully_assessed" for e in evaluations),
            "diversity": len({ga.signature(ind) for ind in plans}) / size,
            "evaluations": (generation + 1) * size,
            "incumbent_fitness": incumbent[1].fitness,
            "incumbent_base_score": incumbent[1].base_score,
            "incumbent_ul_status": incumbent[1].ul_status,
        })
    return incumbent[0], incumbent[1], history


def attach_ea_incumbent(history, population_size):
    incumbent_key = None
    for row in history:
        key = (TIER[row["best_ul_status"]], row["best_excess"], -row["best_fitness"])
        if incumbent_key is None or key < incumbent_key:
            incumbent_key, incumbent = key, row
        row.update({
            "evaluations": (row["generation"] + 1) * population_size,
            "incumbent_fitness": incumbent["best_fitness"],
            "incumbent_base_score": incumbent["best_score"],
            "incumbent_ul_status": incumbent["best_ul_status"],
        })


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(runs):
    result = []
    for condition in STYLES:
        group = [r for r in runs if r["condition"] == condition]
        finals = [r["final_fitness"] for r in group]
        result.append({
            "condition": condition,
            "runs": len(group),
            "mean_initial_fitness": mean(r["initial_best_fitness"] for r in group),
            "mean_final_fitness": mean(finals),
            "sd_final_fitness": stdev(finals) if len(finals) > 1 else 0,
            "min_final_fitness": min(finals),
            "max_final_fitness": max(finals),
            "mean_fitness_gain": mean(r["fitness_gain"] for r in group),
            "all_targets_met": sum(r["all_targets_met"] for r in group),
            "median_first_all_targets_generation_success_only": (
                median([r["first_all_targets_generation"] for r in group if r["all_targets_met"]])
                if any(r["all_targets_met"] for r in group) else None
            ),
            "mean_final_diversity": mean(r["final_diversity"] for r in group),
            "mean_final_population_score": mean(r["final_population_mean_score"] for r in group),
            "mean_vitamin_d_adequacy_pct": mean(r["mean_vitamin_d_adequacy_pct"] for r in group),
            "mean_vitamin_d_deficit_ug": mean(r["mean_vitamin_d_deficit_ug"] for r in group),
            "winning_plans_with_known_ul_excess": sum(r["final_ul_status"] == "exceeded" for r in group),
            "winning_plans_with_incomplete_ul_checks": sum(
                r["incomplete_ul_checks"] > 0 for r in group
            ),
            "median_runtime_s": median(r["runtime_s"] for r in group),
        })
    return result


def make_figures(output, histories, runs, seeds):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#25344a",
        "text.color": "#17263b", "xtick.color": "#40516a", "ytick.color": "#40516a",
        "savefig.facecolor": "white",
    })
    by_condition = {
        condition: [r for r in histories if r["condition"] == condition]
        for condition in STYLES
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
    for ax, conditions, title in (
        (axes[0], tuple(STYLES)[:4], "A   Original 30-food pool"),
        (axes[1], tuple(STYLES)[4:], "B   Candidate-pool stress test"),
    ):
        for condition in conditions:
            rows = by_condition[condition]
            x = [r["evaluations"] for r in rows if r["seed"] == seeds[0]]
            values = np.array([r["incumbent_fitness"] for r in rows]).reshape(len(seeds), len(x))
            label, color, line = STYLES[condition]
            lo, med, hi = np.percentile(values, (25, 50, 75), axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
            ax.plot(x, med, label=label, color=color, linestyle=line, linewidth=2)
        ax.axhline(100, color="#a0a9b3", linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Cumulative fitness evaluations")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    axes[0].set_ylabel("Best-so-far fitness (points)")
    fig.savefig(output / "convergence.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained", sharex=True)
    for ax, conditions, title in (
        (axes[0], tuple(STYLES)[:4], "A   Original 30-food pool"),
        (axes[1], tuple(STYLES)[4:], "B   Candidate-pool stress test"),
    ):
        for i, condition in enumerate(conditions):
            group = [r for r in runs if r["condition"] == condition]
            finals = [r["final_fitness"] for r in group]
            _, color, _ = STYLES[condition]
            jitter = np.linspace(-0.18, 0.18, len(finals)) if len(finals) > 1 else [0]
            ax.scatter(finals, i + np.array(jitter), color=color, s=20, alpha=0.62)
            ax.plot([mean(finals)] * 2, [i - 0.27, i + 0.27], color=color, linewidth=2)
        ax.set_yticks(range(len(conditions)), [STYLES[c][0] for c in conditions])
        ax.set_ylim(len(conditions) - 0.5, -0.5)
        ax.axvline(100, color="#a0a9b3", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Final best-so-far fitness (points)")
        ax.grid(axis="x", alpha=0.18)
    fig.savefig(output / "final_comparison.png", dpi=210)
    plt.close(fig)


def run_suite(output, seeds=SEEDS, population=100, generations=50, backend="numpy"):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply at least one seed and do not repeat seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    foods, meta = load_foods(ROOT / "dataset")
    without_salmon = [food for food in foods if food.id != SALMON_ID]
    if len(without_salmon) != len(foods) - 1:
        raise ValueError("The stress test requires CNF salmon food code 3053")
    people_path = ROOT / "data" / "people.json"
    people = [Person(**row) for row in json.loads(people_path.read_text(encoding="utf-8"))]
    cfg = Config(population_size=population, generations=generations)
    conditions = {
        "default_ea": (foods, cfg, False),
        "random_search": (foods, cfg, True),
        "no_crossover": (foods, replace(cfg, crossover_probability=0.0), False),
        "no_portion_mutation": (foods, replace(cfg, portion_mutation_probability=0.0), False),
        "no_salmon_ea": (without_salmon, cfg, False),
        "no_salmon_random": (without_salmon, cfg, True),
    }
    runs, histories, nutrients = [], [], []
    for condition, (pool, config, random_only) in conditions.items():
        for seed in seeds:
            ga = MealGA(pool, people, config, seed, backend=backend)
            start = perf_counter()
            if random_only:
                individual, ev, history = run_random_search(ga)
            else:
                winners, history = ga.run()
                individual, ev = winners[0]
                attach_ea_incumbent(history, population)
            elapsed = perf_counter() - start
            ga.validate(individual)
            last = history[-1]
            assert math.isclose(ev.fitness, last["incumbent_fitness"], abs_tol=TOL)
            assert ev.ul_status == last["incumbent_ul_status"]
            first_full = next((r["generation"] for r in history
                               if r["incumbent_base_score"] >= 100 - TOL), None)
            d_adequacy = [100 * min(1.0, p["totals"]["vitamin_d_ug"] / p["targets"]["vitamin_d_ug"])
                          for p in ev.people]
            salmon_g = sum(
                grams for pi in range(len(people)) for _, _, food_id, grams in ga.portions(individual, pi)
                if food_id == SALMON_ID
            )
            row = {
                "condition": condition,
                "seed": seed,
                "candidate_count": len(pool),
                "evaluations": last["evaluations"],
                "initial_best_fitness": history[0]["incumbent_fitness"],
                "initial_population_mean_score": history[0]["mean_score"],
                "final_fitness": ev.fitness,
                "final_base_score": ev.base_score,
                "fitness_gain": ev.fitness - history[0]["incumbent_fitness"],
                "final_ul_status": ev.ul_status,
                "final_excess": ev.excess,
                "all_targets_met": int(first_full is not None),
                "first_all_targets_generation": first_full,
                "final_population_mean_score": last["mean_score"],
                "final_diversity": last["diversity"],
                "min_member_score": min(p["score"] for p in ev.people),
                "member_score_spread": max(p["score"] for p in ev.people) - min(p["score"] for p in ev.people),
                "mean_vitamin_d_adequacy_pct": mean(d_adequacy),
                "mean_vitamin_d_deficit_ug": mean(p["deficits"]["vitamin_d_ug"] for p in ev.people),
                "mean_energy_pct_of_reference": mean(100 * p["totals"]["energy_kcal"] / person.energy_kcal
                                                    for p, person in zip(ev.people, people)),
                "salmon_g_total": float(salmon_g),
                "incomplete_ul_checks": sum(not check["fully_assessed"] for p in ev.people
                                            for check in p["ul_checks"].values()),
                "runtime_s": elapsed,
            }
            runs.append(row)
            histories.extend({"condition": condition, "seed": seed, **r} for r in history)
            for index, person_result in enumerate(ev.people, 1):
                for name in NUTRIENTS:
                    nutrients.append({
                        "condition": condition,
                        "seed": seed,
                        "member": f"Adult {index}",
                        "nutrient": name,
                        "actual": person_result["totals"][name],
                        "target": person_result["targets"][name],
                        "deficit": person_result["deficits"][name],
                        "score": person_result["scores"][name],
                    })
            if condition == "default_ea" and seed == seeds[0]:
                write_json(output / f"example_plan_seed{seed}.json", {
                    "condition": condition, "seed": seed, "backend": ga.backend_info(),
                    "config": asdict(config), "members": [asdict(p) for p in people],
                    "plan": ga.report(individual, ev),
                })
        print(f"{condition}: {len(seeds)} runs completed", flush=True)
    summary = summarize(runs)
    paired = {(r["condition"], r["seed"]): r for r in runs}
    comparisons = (
        ("default_ea", "random_search"), ("default_ea", "no_crossover"),
        ("default_ea", "no_portion_mutation"), ("no_salmon_ea", "no_salmon_random"),
        ("default_ea", "no_salmon_ea"),
    )
    differences = []
    for a, b in comparisons:
        deltas = [paired[a, s]["final_fitness"] - paired[b, s]["final_fitness"] for s in seeds]
        differences.append({
            "first": a, "second": b, "mean_final_fitness_difference": mean(deltas),
            "first_wins": sum(d > TOL for d in deltas),
            "ties": sum(abs(d) <= TOL for d in deltas),
            "second_wins": sum(d < -TOL for d in deltas),
        })
    write_csv(output / "runs.csv", runs)
    write_csv(output / "history.csv", histories)
    write_csv(output / "nutrients.csv", nutrients)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", differences)
    files = {Path(path).name: digest for path, digest in meta["input_sha256"].items()}
    write_json(output / "run_manifest.json", {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "backend": ga.backend_info(),
        "seeds": list(seeds),
        "generation_zero_counts": True,
        "evaluations_per_run": population * (generations + 1),
        "number_of_runs": len(runs),
        "total_evaluations": sum(r["evaluations"] for r in runs),
        "counting_rule": "All scored individuals count, including rescored elites and duplicate plans.",
        "conditions": {
            key: {"method": "independent random search" if spec[2] else "EA",
                  "config": asdict(spec[1]), "food_ids": [f.id for f in spec[0]]}
            for key, spec in conditions.items()
        },
        "members": [asdict(p) for p in people],
        "dataset": {"source": meta["source"], "source_url": meta["source_url"],
                    "input_sha256": files, "people_sha256": sha256(people_path.read_bytes()).hexdigest()},
        "winning_plan_ul_status_counts": dict(Counter(r["final_ul_status"] for r in runs)),
        "ul_fields": list(UL),
    })
    make_figures(output, histories, runs, seeds)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--quick", action="store_true", help="Run a smoke study in a separate folder by default.")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    args = parser.parse_args()
    output = args.output
    if args.quick and output == ROOT / "experiments" / "results":
        output = ROOT / "experiments" / "quick_results"
    if args.quick:
        run_suite(output, seeds=(42, 101), population=12, generations=3, backend=args.backend)
    else:
        run_suite(output, backend=args.backend)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
