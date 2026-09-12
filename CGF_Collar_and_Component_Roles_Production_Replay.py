"""Clean replay of the finite-collar and GHG component-role claims.

Spyder use
----------
1. Run once with PROFILE = "screen" (the default).
2. Inspect ``screen_complete.json`` and the ordered log.
3. Change PROFILE to "production" without changing any scientific constant.
4. Run again. Existing completed case files are resumed automatically.

The campaign replaces the heterogeneous early tables by matched comparisons
under one interface-aware PHS RBF-FD engine. It does not reuse the historical
numerical values and does not assume that a positive result must occur.
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

import json
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
    coverage_metrics,
    environment_record,
    global_halton,
    global_random,
    load_json,
    median,
    pilot_gradient_from_result,
    protected_band_allocation,
    protected_weighted_allocation,
    protocol_hash,
    scientific_manifest,
    sha256_file,
    solve_monolithic,
    utc_now,
    vertical_resistance_problem,
)


# =============================================================================
# USER SWITCH: run screen first, then change only this line to "production".
# =============================================================================

PROFILE = "production"  # "screen" or "production"
FORCE = False
OUTPUT_DIRECTORY = None  # None -> directory beside this script


IMPLEMENTATION_VERSION = "CGF-COLLAR-ROLES-REPLAY-1.2"
SEEDS = (11, 23, 37, 53, 71)
BUDGETS = (120, 180, 260, 360)
PROTECTED_SKELETON_FRACTION = 0.45
COLLAR_MULTIPLIER = 2.5
COLLAR_WIDTH_CAP = 0.20
SHEET_MULTIPLIER = 0.15
EVALUATION_COLLAR_WIDTH = 0.08
CANDIDATE_MULTIPLIER = 18
STENCIL_SIZE = DEFAULT_STENCIL

ALGEBRAIC_TOLERANCE = 1.0e-8
INTERFACE_TRACE_TOLERANCE = 1.0e-8
INTERFACE_FLUX_TOLERANCE = 1.0e-6
CONSERVATION_TOLERANCE = 0.10
OVERSHOOT_TOLERANCE = 0.05
CONDITION_LIMIT = 1.0e10

METHODS = (
    "HALTON",
    "GLOBAL_RANDOM",
    "PROTECTED_NEAR_INTERFACE_SHEET",
    "PROTECTED_FINITE_COLLAR_HALTON",
    "PROTECTED_FINITE_COLLAR_RANDOM",
    "PROTECTED_GHG_EQUAL",
    "PROTECTED_GHG_NO_GREEN",
    "PROTECTED_GHG_NO_GRADIENT",
    "UNPROTECTED_GHG_EQUAL",
)


def script_directory() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def profile_contract() -> dict[str, Any]:
    if PROFILE == "screen":
        seeds = (SEEDS[0],)
        budgets = (80,)
        evaluation_resolution = 35
    elif PROFILE == "production":
        seeds = SEEDS
        budgets = BUDGETS
        evaluation_resolution = 81
    else:
        raise ValueError("PROFILE must be 'screen' or 'production'")
    problems = (
        vertical_resistance_problem("S_TRAIN_XI050_K1_100", 0.50, 1.0, 100.0),
        vertical_resistance_problem("S_HOLDOUT_XI037_K1_30", 0.37, 1.0, 30.0),
    )
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "profile": PROFILE,
        "evidence_authorization": PROFILE == "production",
        "seeds": list(seeds),
        "budgets": list(budgets),
        "problems": [
            {
                "name": problem.name,
                "k_minus": problem.k_minus,
                "k_plus": problem.k_plus,
                "metadata": problem.metadata,
            }
            for problem in problems
        ],
        "methods": list(METHODS),
        "protected_skeleton_fraction": PROTECTED_SKELETON_FRACTION,
        "collar_width_rule": "min(0.20, 2.5/sqrt(interior_budget))",
        "near_interface_sheet_rule": "0.15/sqrt(interior_budget)",
        "evaluation_collar_width": EVALUATION_COLLAR_WIDTH,
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
        "decision_rules": {
            "finite_collar_principle": (
                "For each configuration, the finite-collar Halton enrichment must have "
                "median collar RMSE ratio < 1 against the near-interface sheet, one-sided "
                "paired Wilcoxon p < 0.05, at least 18/20 admissible rows, and median "
                "full-field ratio <= 1.25. Both configurations must pass."
            ),
            "low_discrepancy_within_collar": (
                "Compare protected finite-collar Halton and protected finite-collar random "
                "under identical collar geometry; report without a mandatory superiority claim."
            ),
            "component_roles": (
                "Report paired changes caused by Green removal, Gradient removal, and floor "
                "removal separately on full, collar, coverage, condition, and admissibility metrics."
            ),
        },
    }


def make_allocation(method: str, problem: Any, budget: int, seed: int, pilot: Any) -> tuple[np.ndarray, int]:
    collar_width = min(COLLAR_WIDTH_CAP, COLLAR_MULTIPLIER / math.sqrt(budget))
    sheet_width = SHEET_MULTIPLIER / math.sqrt(budget)
    if method == "HALTON":
        return global_halton(problem, budget, seed), budget
    if method == "GLOBAL_RANDOM":
        return global_random(problem, budget, seed), 0
    if method == "PROTECTED_NEAR_INTERFACE_SHEET":
        return protected_band_allocation(
            problem, budget, seed, PROTECTED_SKELETON_FRACTION, sheet_width, False
        )
    if method == "PROTECTED_FINITE_COLLAR_HALTON":
        return protected_band_allocation(
            problem, budget, seed, PROTECTED_SKELETON_FRACTION, collar_width, False
        )
    if method == "PROTECTED_FINITE_COLLAR_RANDOM":
        return protected_band_allocation(
            problem, budget, seed, PROTECTED_SKELETON_FRACTION, collar_width, True
        )
    if method == "PROTECTED_GHG_EQUAL":
        weights, floor = (1.0 / 3.0,) * 3, PROTECTED_SKELETON_FRACTION
    elif method == "PROTECTED_GHG_NO_GREEN":
        weights, floor = (0.0, 0.5, 0.5), PROTECTED_SKELETON_FRACTION
    elif method == "PROTECTED_GHG_NO_GRADIENT":
        weights, floor = (0.5, 0.5, 0.0), PROTECTED_SKELETON_FRACTION
    elif method == "UNPROTECTED_GHG_EQUAL":
        weights, floor = (1.0 / 3.0,) * 3, 0.0
    else:
        raise ValueError(method)
    return protected_weighted_allocation(
        problem,
        budget,
        seed,
        weights,
        pilot,
        skeleton_fraction=floor,
        green_length=collar_width,
        candidate_multiplier=CANDIDATE_MULTIPLIER,
    )


def gates(metrics: dict[str, Any]) -> dict[str, bool]:
    finite = all(np.isfinite(float(metrics[key])) for key in (
        "full_rmse", "collar_rmse", "relative_overshoot", "conservation_defect",
        "algebraic_residual_relative", "interface_trace_rms", "interface_flux_rms",
    ))
    condition = float(metrics["condition_proxy_equilibrated_1norm"])
    gate_condition = bool(np.isfinite(condition) and condition <= CONDITION_LIMIT)
    result = {
        "gate_finite": finite,
        "gate_algebraic": finite and metrics["algebraic_residual_relative"] <= ALGEBRAIC_TOLERANCE,
        "gate_trace": finite and metrics["interface_trace_rms"] <= INTERFACE_TRACE_TOLERANCE,
        "gate_flux": finite and metrics["interface_flux_rms"] <= INTERFACE_FLUX_TOLERANCE,
        "gate_conservation": finite and metrics["conservation_defect"] <= CONSERVATION_TOLERANCE,
        "gate_overshoot": finite and metrics["relative_overshoot"] <= OVERSHOOT_TOLERANCE,
        "gate_condition": gate_condition,
    }
    result["gate_admissible"] = all(result.values())
    return result


def paired_p_less(challenger: list[float], reference: list[float]) -> float:
    differences = np.asarray(challenger) - np.asarray(reference)
    if differences.size == 0 or np.all(np.abs(differences) <= 1.0e-15):
        return 1.0
    try:
        return float(wilcoxon(differences, alternative="less", zero_method="pratt").pvalue)
    except ValueError:
        return 1.0


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired_rows: list[dict[str, Any]] = []
    indexed = {
        (row["problem"], int(row["budget"]), int(row["seed"]), row["method"]): row
        for row in rows
    }
    references = {
        "PROTECTED_NEAR_INTERFACE_SHEET": "PROTECTED_FINITE_COLLAR_HALTON",
        "PROTECTED_FINITE_COLLAR_RANDOM": "PROTECTED_FINITE_COLLAR_HALTON",
        "PROTECTED_GHG_EQUAL": "PROTECTED_GHG_NO_GREEN",
        "PROTECTED_GHG_EQUAL__NO_GRADIENT": "PROTECTED_GHG_NO_GRADIENT",
        "PROTECTED_GHG_EQUAL__NO_FLOOR": "UNPROTECTED_GHG_EQUAL",
    }
    for problem in sorted({row["problem"] for row in rows}):
        keys = sorted({(int(row["budget"]), int(row["seed"])) for row in rows if row["problem"] == problem})
        for comparison, method in references.items():
            if comparison.startswith("PROTECTED_GHG_EQUAL__"):
                reference_method = "PROTECTED_GHG_EQUAL"
            else:
                reference_method = comparison
            challenger_rows = [indexed[(problem, budget, seed, method)] for budget, seed in keys]
            reference_rows = [indexed[(problem, budget, seed, reference_method)] for budget, seed in keys]
            for metric in (
                "full_rmse", "collar_rmse", "coverage_fill_max",
                "condition_proxy_equilibrated_1norm", "conservation_defect",
            ):
                challenger = [float(row[metric]) for row in challenger_rows]
                reference = [float(row[metric]) for row in reference_rows]
                ratios = [a / max(b, 1.0e-300) for a, b in zip(challenger, reference)]
                paired_rows.append({
                    "problem": problem,
                    "comparison": f"{method}_OVER_{reference_method}",
                    "metric": metric,
                    "pairs": len(ratios),
                    "median_ratio": median(ratios),
                    "wins_ratio_below_one": int(sum(value < 1.0 for value in ratios)),
                    "one_sided_p_challenger_less": paired_p_less(challenger, reference),
                    "challenger_admissible": int(sum(bool(row["gate_admissible"]) for row in challenger_rows)),
                    "reference_admissible": int(sum(bool(row["gate_admissible"]) for row in reference_rows)),
                })

    production_pairs = len(SEEDS) * len(BUDGETS)
    collar_decisions: dict[str, Any] = {}
    for problem in sorted({row["problem"] for row in rows}):
        lookup = {
            (row["comparison"], row["metric"]): row
            for row in paired_rows if row["problem"] == problem
        }
        collar = lookup[(
            "PROTECTED_FINITE_COLLAR_HALTON_OVER_PROTECTED_NEAR_INTERFACE_SHEET", "collar_rmse"
        )]
        full = lookup[(
            "PROTECTED_FINITE_COLLAR_HALTON_OVER_PROTECTED_NEAR_INTERFACE_SHEET", "full_rmse"
        )]
        required_admissible = 18 if PROFILE == "production" else 1
        collar_decisions[problem] = {
            "median_collar_ratio": collar["median_ratio"],
            "one_sided_p": collar["one_sided_p_challenger_less"],
            "median_full_ratio": full["median_ratio"],
            "admissible_rows": collar["challenger_admissible"],
            "required_rows": required_admissible,
            "pass": bool(
                collar["median_ratio"] < 1.0
                and collar["one_sided_p_challenger_less"] < 0.05
                and full["median_ratio"] <= 1.25
                and collar["challenger_admissible"] >= required_admissible
            ) if PROFILE == "production" else None,
        }
    decision = {
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "expected_pairs_per_problem": production_pairs if PROFILE == "production" else 1,
        "finite_collar_by_problem": collar_decisions,
        "finite_collar_principle_confirmed": (
            all(item["pass"] for item in collar_decisions.values()) if PROFILE == "production" else None
        ),
        "interpretation": (
            "Production decisions are mechanical consequences of the frozen rule."
            if PROFILE == "production"
            else "Implementation screen only; no manuscript claim is authorized."
        ),
    }
    return paired_rows, decision


def save_figures(rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], output: Path) -> None:
    problems = sorted({row["problem"] for row in rows})
    fig, axes = plt.subplots(1, len(problems), figsize=(6.2 * len(problems), 4.8), squeeze=False)
    for axis, problem in zip(axes[0], problems):
        for method in (
            "HALTON", "PROTECTED_NEAR_INTERFACE_SHEET",
            "PROTECTED_FINITE_COLLAR_HALTON", "PROTECTED_FINITE_COLLAR_RANDOM",
        ):
            subset = [row for row in rows if row["problem"] == problem and row["method"] == method]
            budgets = sorted({int(row["budget"]) for row in subset})
            values = [median(float(row["collar_rmse"]) for row in subset if int(row["budget"]) == budget) for budget in budgets]
            axis.plot(budgets, values, marker="o", label=method.replace("PROTECTED_", "").replace("_", " ").title())
        axis.set_yscale("log")
        axis.set_xlabel("Interior budget")
        axis.set_ylabel("Collar RMSE")
        axis.set_title(problem)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "finite_collar_matched_controls.png", dpi=220)
    plt.close(fig)

    component = [
        row for row in paired_rows
        if row["metric"] in ("full_rmse", "collar_rmse", "coverage_fill_max")
        and "PROTECTED_GHG" in row["comparison"]
    ]
    labels = [f"{row['problem']}\n{row['comparison'].replace('PROTECTED_GHG_', '')}\n{row['metric']}" for row in component]
    values = [float(row["median_ratio"]) for row in component]
    fig, axis = plt.subplots(figsize=(max(9.0, 0.42 * len(values)), 5.2))
    positions = np.arange(len(values))
    axis.bar(positions, values, color=["#2d6a9f" if value <= 1.0 else "#ba5a31" for value in values])
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xticks(positions, labels, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Median ratio (listed method / protected full GHG)")
    axis.set_title("Component-removal and floor-removal effects")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "component_role_paired_ratios.png", dpi=220)
    plt.close(fig)


def main() -> None:
    contract = profile_contract()
    base = script_directory()
    output = Path(OUTPUT_DIRECTORY).resolve() if OUTPUT_DIRECTORY else base / f"results_collar_roles_{PROFILE}_v1p2"
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

    seeds = tuple(contract["seeds"])
    budgets = tuple(contract["budgets"])
    problems = (
        vertical_resistance_problem("S_TRAIN_XI050_K1_100", 0.50, 1.0, 100.0),
        vertical_resistance_problem("S_HOLDOUT_XI037_K1_30", 0.37, 1.0, 30.0),
    )
    rows: list[dict[str, Any]] = []
    for problem in problems:
        for budget in budgets:
            boundary_per_edge = max(8, int(math.ceil(math.sqrt(budget) * 0.70)))
            interface_count = max(10, int(math.ceil(math.sqrt(budget) * 0.85)))
            for seed in seeds:
                # The Halton pilot is deliberately recomputed on resume because its
                # numerical state defines the deployable Gradient indicator.
                pilot_points, _ = make_allocation("HALTON", problem, budget, seed, None)
                pilot_result = solve_monolithic(
                    problem, pilot_points, boundary_per_edge, interface_count,
                    int(contract["evaluation_resolution"]), EVALUATION_COLLAR_WIDTH,
                    STENCIL_SIZE, compute_condition=True,
                )
                pilot = pilot_gradient_from_result(pilot_result)
                for method in METHODS:
                    stem = case_stem(problem.name, method, budget, seed)
                    case_path = cases / f"{stem}.json"
                    if case_path.exists() and not FORCE:
                        row = load_json(case_path)
                        if row.get("protocol_hash") != contract["protocol_hash"]:
                            raise RuntimeError(f"Protocol mismatch in existing case: {case_path}")
                        rows.append(row)
                        log(f"RESUME {stem}")
                        continue
                    start = time.perf_counter()
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
                        "method": method,
                        "budget": int(budget),
                        "seed": int(seed),
                        "skeleton_count": int(skeleton_count),
                        "allocation_collar_half_width": min(COLLAR_WIDTH_CAP, COLLAR_MULTIPLIER / math.sqrt(budget)),
                        "near_interface_sheet_half_width": SHEET_MULTIPLIER / math.sqrt(budget),
                        "charged_seconds": float(time.perf_counter() - start),
                        **coverage_metrics(points),
                        **result.metrics,
                    }
                    row.update(gates(row))
                    atomic_json(case_path, row)
                    rows.append(row)
                    log(
                        f"DONE {stem}; full={row['full_rmse']:.4e}; collar={row['collar_rmse']:.4e}; "
                        f"condition={row['condition_proxy_equilibrated_1norm']:.3e}; admissible={row['gate_admissible']}"
                    )

    rows.sort(key=lambda row: (row["problem"], int(row["budget"]), int(row["seed"]), row["method"]))
    paired_rows, decision = aggregate(rows)
    atomic_csv(output / "campaign_metrics.csv", rows)
    atomic_csv(output / "paired_statistics.csv", paired_rows)
    atomic_json(output / "decision.json", decision)
    save_figures(rows, paired_rows, output)
    completion_name = "production_complete.json" if PROFILE == "production" else "screen_complete.json"
    atomic_json(output / completion_name, {
        "completed_utc": utc_now(),
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "protocol_hash": contract["protocol_hash"],
        "case_count": len(rows),
        "expected_case_count": len(problems) * len(budgets) * len(seeds) * len(METHODS),
        "decision_file": "decision.json",
    })
    log(f"COMPLETE profile={PROFILE}; cases={len(rows)}; evidence_authorized={PROFILE == 'production'}")
    atomic_csv(output / "scientific_manifest_sha256.csv", scientific_manifest(output))


if __name__ == "__main__":
    main()
