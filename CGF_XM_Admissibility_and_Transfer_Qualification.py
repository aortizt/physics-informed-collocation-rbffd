"""Qualification campaign for the earlier X/M allocation signals.

The script supplies exact manufactured states for an oblique crossing
interface (X) and an off-centre material inclusion (M).  A finite simplex
library is evaluated under one PHS RBF-FD engine.  Candidate selection uses
only the named training configuration; the geometrically and physically
changed holdout is then evaluated without retuning.

Run ``screen`` first in Spyder.  Change only PROFILE to ``production`` after
the screen report has been inspected.  Screen output is never paper evidence.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from CGF_Replay_Common_PHS_RBFFD import (
    DEFAULT_STENCIL,
    OrderedLog,
    atomic_csv,
    atomic_json,
    case_stem,
    circular_inclusion_problem,
    coverage_metrics,
    environment_record,
    global_halton,
    load_json,
    median,
    oblique_interface_problem,
    pilot_gradient_from_result,
    protected_weighted_allocation,
    protocol_hash,
    scientific_manifest,
    sha256_file,
    solve_monolithic,
    utc_now,
)


PROFILE = "production"  # "screen" or "production"
FORCE = False
OUTPUT_DIRECTORY = None

IMPLEMENTATION_VERSION = "CGF-XM-QUALIFICATION-1.2"
SEEDS = (11, 23, 37, 53, 71)
BUDGETS = (120, 180, 260, 360)
PROTECTED_SKELETON_FRACTION = 0.45
GREEN_LENGTH = 0.11
CANDIDATE_MULTIPLIER = 18
STENCIL_SIZE = DEFAULT_STENCIL
EVALUATION_COLLAR_WIDTH = 0.075

ALGEBRAIC_TOLERANCE = 1.0e-8
INTERFACE_TRACE_TOLERANCE = 1.0e-8
INTERFACE_FLUX_TOLERANCE = 1.0e-6
CONSERVATION_TOLERANCE = 0.10
OVERSHOOT_TOLERANCE = 0.05
CONDITION_LIMIT = 1.0e10
FULL_FIELD_RATIO_LIMIT = 1.25

METHOD_WEIGHTS: dict[str, tuple[float, float, float] | None] = {
    "HALTON": None,
    "GHG_EQUAL": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "C1_COVERAGE_GRADIENT_EQUAL": (0.0, 0.50, 0.50),
    "C3_COVERAGE_HEAVY": (0.0, 0.75, 0.25),
    "GREEN_COVERAGE_EQUAL": (0.50, 0.50, 0.0),
    "GREEN_HEAVY": (0.60, 0.25, 0.15),
}


def script_directory() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def make_problems() -> tuple[Any, ...]:
    return (
        oblique_interface_problem("X_TRAIN_ANGLE25_K1_50", 25.0, (0.50, 0.50), 1.0, 50.0),
        oblique_interface_problem("X_HOLDOUT_ANGLE40_K1_20", 40.0, (0.52, 0.47), 1.0, 20.0),
        circular_inclusion_problem("M_TRAIN_CIRCLE_R024_K30_1", (0.50, 0.50), 0.24, 30.0, 1.0),
        circular_inclusion_problem("M_HOLDOUT_CIRCLE_R020_K15_1", (0.56, 0.44), 0.20, 15.0, 1.0),
    )


def profile_contract() -> dict[str, Any]:
    if PROFILE == "screen":
        seeds = (SEEDS[0],)
        budgets = (120,)
        evaluation_resolution = 37
    elif PROFILE == "production":
        seeds = SEEDS
        budgets = BUDGETS
        evaluation_resolution = 81
    else:
        raise ValueError("PROFILE must be 'screen' or 'production'")
    problems = make_problems()
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "profile": PROFILE,
        "evidence_authorization": PROFILE == "production",
        "seeds": list(seeds),
        "budgets": list(budgets),
        "problems": [
            {"name": problem.name, "k_minus": problem.k_minus, "k_plus": problem.k_plus,
             "metadata": problem.metadata}
            for problem in problems
        ],
        "method_weights_alpha_beta_gamma": {
            method: list(weights) if weights is not None else None
            for method, weights in METHOD_WEIGHTS.items()
        },
        "protected_skeleton_fraction": PROTECTED_SKELETON_FRACTION,
        "green_length": GREEN_LENGTH,
        "gradient_indicator": "local least-squares magnitude from matched Halton pilot state",
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "phs_power": 3,
        "polynomial_degree": 2,
        "stencil_size": STENCIL_SIZE,
        "evaluation_resolution": evaluation_resolution,
        "gates": {
            "algebraic_residual_relative": ALGEBRAIC_TOLERANCE,
            "interface_trace_rms": INTERFACE_TRACE_TOLERANCE,
            "interface_flux_rms": INTERFACE_FLUX_TOLERANCE,
            "conservation_defect": CONSERVATION_TOLERANCE,
            "relative_overshoot": OVERSHOOT_TOLERANCE,
            "condition_proxy_equilibrated_1norm": CONDITION_LIMIT,
        },
        "selection_rule": (
            "Within each regime, retain only training candidates with at least 18/20 admissible rows; "
            "select the smallest training median collar RMSE, breaking ties by full RMSE and condition."
        ),
        "holdout_rule": (
            "The frozen selected candidate must have at least 18/20 admissible holdout rows, median "
            "collar ratio < 1 and one-sided paired Wilcoxon p < 0.05 versus Halton, and median full "
            f"ratio <= {FULL_FIELD_RATIO_LIMIT}."
        ),
    }


def make_allocation(method: str, problem: Any, budget: int, seed: int, pilot: Any) -> tuple[np.ndarray, int]:
    weights = METHOD_WEIGHTS[method]
    if weights is None:
        return global_halton(problem, budget, seed), budget
    return protected_weighted_allocation(
        problem,
        budget,
        seed,
        weights,
        pilot,
        skeleton_fraction=PROTECTED_SKELETON_FRACTION,
        green_length=GREEN_LENGTH,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
    )


def gates(metrics: dict[str, Any]) -> dict[str, bool]:
    finite = all(np.isfinite(float(metrics[key])) for key in (
        "full_rmse", "collar_rmse", "relative_overshoot", "conservation_defect",
        "algebraic_residual_relative", "interface_trace_rms", "interface_flux_rms",
        "condition_proxy_equilibrated_1norm",
    ))
    result = {
        "gate_finite": finite,
        "gate_algebraic": finite and metrics["algebraic_residual_relative"] <= ALGEBRAIC_TOLERANCE,
        "gate_trace": finite and metrics["interface_trace_rms"] <= INTERFACE_TRACE_TOLERANCE,
        "gate_flux": finite and metrics["interface_flux_rms"] <= INTERFACE_FLUX_TOLERANCE,
        "gate_conservation": finite and metrics["conservation_defect"] <= CONSERVATION_TOLERANCE,
        "gate_overshoot": finite and metrics["relative_overshoot"] <= OVERSHOOT_TOLERANCE,
        "gate_condition": finite and metrics["condition_proxy_equilibrated_1norm"] <= CONDITION_LIMIT,
    }
    result["gate_admissible"] = all(result.values())
    return result


def p_less(challenger: list[float], primary: list[float]) -> float:
    difference = np.asarray(challenger) - np.asarray(primary)
    if difference.size == 0 or np.all(np.abs(difference) <= 1.0e-15):
        return 1.0
    try:
        return float(wilcoxon(difference, alternative="less", zero_method="pratt").pvalue)
    except ValueError:
        return 1.0


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["problem"], row["method"], int(row["budget"]), int(row["seed"])): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for problem in sorted({row["problem"] for row in rows}):
        keys = sorted({(int(row["budget"]), int(row["seed"])) for row in rows if row["problem"] == problem})
        primary = [indexed[(problem, "HALTON", budget, seed)] for budget, seed in keys]
        for method in METHOD_WEIGHTS:
            if method == "HALTON":
                continue
            candidate = [indexed[(problem, method, budget, seed)] for budget, seed in keys]
            for metric in ("full_rmse", "collar_rmse", "condition_proxy_equilibrated_1norm",
                           "coverage_fill_max", "conservation_defect", "relative_overshoot"):
                a = [float(row[metric]) for row in candidate]
                b = [float(row[metric]) for row in primary]
                ratios = [left / max(right, 1.0e-300) for left, right in zip(a, b)]
                result.append({
                    "problem": problem,
                    "method": method,
                    "reference": "HALTON",
                    "metric": metric,
                    "pairs": len(ratios),
                    "median_ratio": median(ratios),
                    "wins": int(sum(value < 1.0 for value in ratios)),
                    "one_sided_p_candidate_less": p_less(a, b),
                    "candidate_admissible_rows": int(sum(bool(row["gate_admissible"]) for row in candidate)),
                    "halton_admissible_rows": int(sum(bool(row["gate_admissible"]) for row in primary)),
                })
    return result


def select_and_adjudicate(rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    expected = len(SEEDS) * len(BUDGETS) if PROFILE == "production" else 1
    required = 18 if PROFILE == "production" else 1
    for regime in ("X", "M"):
        training_name = next(name for name in {row["problem"] for row in rows} if name.startswith(f"{regime}_TRAIN"))
        holdout_name = next(name for name in {row["problem"] for row in rows} if name.startswith(f"{regime}_HOLDOUT"))
        candidates: list[tuple[float, float, float, str, int]] = []
        for method in METHOD_WEIGHTS:
            if method == "HALTON":
                continue
            subset = [row for row in rows if row["problem"] == training_name and row["method"] == method]
            admissible = sum(bool(row["gate_admissible"]) for row in subset)
            candidates.append((
                median(float(row["collar_rmse"]) for row in subset),
                median(float(row["full_rmse"]) for row in subset),
                median(float(row["condition_proxy_equilibrated_1norm"]) for row in subset),
                method,
                admissible,
            ))
        eligible = [item for item in candidates if item[4] >= required]
        selected = min(eligible) if eligible else None
        if selected is None:
            decisions[regime] = {
                "training_problem": training_name,
                "holdout_problem": holdout_name,
                "selected_method": None,
                "reason": "No candidate met the frozen training admissibility requirement.",
                "holdout_pass": False if PROFILE == "production" else None,
            }
            continue
        method = selected[3]
        lookup = {
            (row["problem"], row["method"], row["metric"]): row
            for row in paired
        }
        collar = lookup[(holdout_name, method, "collar_rmse")]
        full = lookup[(holdout_name, method, "full_rmse")]
        holdout_pass = bool(
            collar["candidate_admissible_rows"] >= required
            and collar["median_ratio"] < 1.0
            and collar["one_sided_p_candidate_less"] < 0.05
            and full["median_ratio"] <= FULL_FIELD_RATIO_LIMIT
        ) if PROFILE == "production" else None
        decisions[regime] = {
            "training_problem": training_name,
            "holdout_problem": holdout_name,
            "selected_method": method,
            "training_median_collar_rmse": selected[0],
            "training_median_full_rmse": selected[1],
            "training_median_condition": selected[2],
            "training_admissible_rows": selected[4],
            "holdout_admissible_rows": collar["candidate_admissible_rows"],
            "holdout_median_collar_ratio": collar["median_ratio"],
            "holdout_collar_one_sided_p": collar["one_sided_p_candidate_less"],
            "holdout_median_full_ratio": full["median_ratio"],
            "holdout_pass": holdout_pass,
        }
    return {
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "rows_expected_per_problem_method": expected,
        "required_admissible_rows": required,
        "regime_decisions": decisions,
        "interpretation": (
            "A regime may enter the paper as a qualified allocation result only if holdout_pass is true."
            if PROFILE == "production"
            else "Implementation screen only; no scientific decision is authorized."
        ),
    }


def save_figures(rows: list[dict[str, Any]], paired: list[dict[str, Any]], output: Path) -> None:
    problems = sorted({row["problem"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), squeeze=False)
    for axis, problem in zip(axes.ravel(), problems):
        for method in METHOD_WEIGHTS:
            subset = [row for row in rows if row["problem"] == problem and row["method"] == method]
            budgets = sorted({int(row["budget"]) for row in subset})
            values = [median(float(row["collar_rmse"]) for row in subset if int(row["budget"]) == budget) for budget in budgets]
            axis.plot(budgets, values, marker="o", label=method.replace("_", " ").title())
        axis.set_yscale("log")
        axis.set_title(problem)
        axis.set_xlabel("Interior budget")
        axis.set_ylabel("Interface/active-layer RMSE")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "xm_accuracy_refinement.png", dpi=220)
    plt.close(fig)

    display = [row for row in paired if row["metric"] == "collar_rmse"]
    labels = [f"{row['problem']}\n{row['method']}" for row in display]
    ratios = [float(row["median_ratio"]) for row in display]
    admissible = [float(row["candidate_admissible_rows"]) / max(float(row["pairs"]), 1.0) for row in display]
    positions = np.arange(len(display))
    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.48 * len(display)), 8.0), sharex=True)
    axes[0].bar(positions, ratios, color=["#287271" if value < 1.0 else "#c65f2f" for value in ratios])
    axes[0].axhline(1.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("Median collar-error ratio to Halton")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(positions, admissible, color="#4c78a8")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Admissible fraction")
    axes[1].set_xticks(positions, labels, rotation=75, ha="right", fontsize=6)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "xm_gain_and_admissibility.png", dpi=220)
    plt.close(fig)


def main() -> None:
    contract = profile_contract()
    base = script_directory()
    output = Path(OUTPUT_DIRECTORY).resolve() if OUTPUT_DIRECTORY else base / f"results_xm_{PROFILE}_v1p2"
    output.mkdir(parents=True, exist_ok=True)
    cases = output / "cases"
    cases.mkdir(exist_ok=True)
    log = OrderedLog(output / "run.log")
    contract["source_sha256"] = {
        "campaign": sha256_file(Path(__file__).resolve()) if "__file__" in globals() else None,
        "common_core": sha256_file(base / "CGF_Replay_Common_PHS_RBFFD.py"),
    }
    contract["protocol_hash"] = protocol_hash(contract)
    atomic_json(output / "protocol.json", contract)
    atomic_json(output / "environment.json", environment_record())
    if "__file__" in globals():
        source = output / "source"
        source.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), source / Path(__file__).name)
        common = base / "CGF_Replay_Common_PHS_RBFFD.py"
        shutil.copy2(common, source / common.name)
    log(f"START {IMPLEMENTATION_VERSION}; profile={PROFILE}; protocol={contract['protocol_hash']}")

    rows: list[dict[str, Any]] = []
    for problem in make_problems():
        for budget in contract["budgets"]:
            budget = int(budget)
            boundary_per_edge = max(8, int(math.ceil(math.sqrt(budget) * 0.72)))
            interface_count = max(12, int(math.ceil(math.sqrt(budget) * 0.90)))
            for seed_value in contract["seeds"]:
                seed = int(seed_value)
                pilot_points = global_halton(problem, budget, seed)
                pilot_result = solve_monolithic(
                    problem, pilot_points, boundary_per_edge, interface_count,
                    int(contract["evaluation_resolution"]), EVALUATION_COLLAR_WIDTH,
                    STENCIL_SIZE, compute_condition=True,
                )
                pilot = pilot_gradient_from_result(pilot_result)
                for method in METHOD_WEIGHTS:
                    stem = case_stem(problem.name, method, budget, seed)
                    path = cases / f"{stem}.json"
                    if path.exists() and not FORCE:
                        row = load_json(path)
                        if row.get("protocol_hash") != contract["protocol_hash"]:
                            raise RuntimeError(f"Protocol mismatch in {path}")
                        rows.append(row)
                        log(f"RESUME {stem}")
                        continue
                    started = time.perf_counter()
                    points, skeleton_count = make_allocation(method, problem, budget, seed, pilot)
                    result = pilot_result if method == "HALTON" else solve_monolithic(
                        problem, points, boundary_per_edge, interface_count,
                        int(contract["evaluation_resolution"]), EVALUATION_COLLAR_WIDTH,
                        STENCIL_SIZE, compute_condition=True,
                    )
                    row = {
                        "implementation_version": IMPLEMENTATION_VERSION,
                        "profile": PROFILE,
                        "protocol_hash": contract["protocol_hash"],
                        "problem": problem.name,
                        "regime": problem.name[0],
                        "configuration": "TRAIN" if "_TRAIN_" in problem.name else "HOLDOUT",
                        "method": method,
                        "alpha": METHOD_WEIGHTS[method][0] if METHOD_WEIGHTS[method] else 0.0,
                        "beta": METHOD_WEIGHTS[method][1] if METHOD_WEIGHTS[method] else 1.0,
                        "gamma": METHOD_WEIGHTS[method][2] if METHOD_WEIGHTS[method] else 0.0,
                        "budget": budget,
                        "seed": seed,
                        "skeleton_count": int(skeleton_count),
                        "charged_seconds": float(time.perf_counter() - started),
                        **coverage_metrics(points),
                        **result.metrics,
                    }
                    row.update(gates(row))
                    atomic_json(path, row)
                    rows.append(row)
                    log(
                        f"DONE {stem}; full={row['full_rmse']:.4e}; collar={row['collar_rmse']:.4e}; "
                        f"conservation={row['conservation_defect']:.3e}; admissible={row['gate_admissible']}"
                    )

    rows.sort(key=lambda row: (row["problem"], int(row["budget"]), int(row["seed"]), row["method"]))
    paired = paired_summary(rows)
    decision = select_and_adjudicate(rows, paired)
    atomic_csv(output / "campaign_metrics.csv", rows)
    atomic_csv(output / "paired_statistics.csv", paired)
    atomic_json(output / "selection_and_holdout_decision.json", decision)
    save_figures(rows, paired, output)
    completion = "production_complete.json" if PROFILE == "production" else "screen_complete.json"
    atomic_json(output / completion, {
        "completed_utc": utc_now(),
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "protocol_hash": contract["protocol_hash"],
        "case_count": len(rows),
        "expected_case_count": len(make_problems()) * len(contract["budgets"]) * len(contract["seeds"]) * len(METHOD_WEIGHTS),
    })
    log(f"COMPLETE profile={PROFILE}; cases={len(rows)}; evidence_authorized={PROFILE == 'production'}")
    atomic_csv(output / "scientific_manifest_sha256.csv", scientific_manifest(output))


if __name__ == "__main__":
    main()
