"""Two-way physical-coupling qualification for the vug (V) mechanism.

This campaign distinguishes three statements that the old development table
mixed together: local approximation quality, contraction of trace/flux
mismatch, and conservative global assembly.  It uses exact circular-inclusion
states, two changing local solves at every iteration, and a training/holdout
split.  One branch makes the GHG state enter the Robin impedance explicitly;
it is therefore a genuine GHG-conditioned transmission test rather than GHG
node allocation followed by an unrelated scalar Robin law.

Run the default ``screen`` profile first in Spyder. Change only PROFILE to
``production`` after inspecting the screen output. Screen output is not
scientific evidence.
"""

from __future__ import annotations

import csv
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
    normalized_indicator,
    pilot_gradient_from_result,
    protected_weighted_allocation,
    protocol_hash,
    run_two_way_robin_schwarz,
    scientific_manifest,
    sha256_file,
    solve_monolithic,
    utc_now,
)


PROFILE = "production"  # "screen" or "production"
FORCE = False
OUTPUT_DIRECTORY = None

IMPLEMENTATION_VERSION = "CGF-VUG-COUPLING-QUALIFICATION-1.4"
SEEDS = (11, 23, 37, 53, 71)
BUDGETS = (120, 180, 260, 360)
ALLOCATIONS = ("HALTON", "PROTECTED_GHG_EQUAL")
TRANSMISSIONS = ("SCALAR_ROBIN_8", "DTN_GEOMETRIC", "GHG_CONDITIONED_DTN")
PROTECTED_SKELETON_FRACTION = 0.45
GREEN_LENGTH = 0.10
CANDIDATE_MULTIPLIER = 18
STENCIL_SIZE = DEFAULT_STENCIL
RELAXATION = 0.70
MAXIMUM_ITERATIONS = 20
EVALUATION_COLLAR_WIDTH = 0.075

Q_NET_LIMIT = 0.95
Q_TAIL_LIMIT = 0.95
TERMINAL_MISMATCH_LIMIT = 5.0e-3
CONSERVATION_TOLERANCE = 0.05
OVERSHOOT_TOLERANCE = 0.05
CONDITION_LIMIT = 1.0e10
ASSEMBLED_ERROR_FACTOR = 1.25


def script_directory() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def make_problems() -> tuple[Any, ...]:
    return (
        circular_inclusion_problem("V_TRAIN_CIRCLE_R024_K50_1", (0.50, 0.50), 0.24, 50.0, 1.0),
        circular_inclusion_problem("V_HOLDOUT_CIRCLE_R020_K20_1", (0.56, 0.44), 0.20, 20.0, 1.0),
    )


def profile_contract() -> dict[str, Any]:
    if PROFILE == "screen":
        seeds = (SEEDS[0],)
        budgets = (120,)
        evaluation_resolution = 37
        maximum_iterations = 8
    elif PROFILE == "production":
        seeds = SEEDS
        budgets = BUDGETS
        evaluation_resolution = 81
        maximum_iterations = MAXIMUM_ITERATIONS
    else:
        raise ValueError("PROFILE must be 'screen' or 'production'")
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "profile": PROFILE,
        "evidence_authorization": PROFILE == "production",
        "seeds": list(seeds),
        "budgets": list(budgets),
        "problems": [
            {"name": problem.name, "k_minus": problem.k_minus, "k_plus": problem.k_plus,
             "metadata": problem.metadata}
            for problem in make_problems()
        ],
        "allocations": list(ALLOCATIONS),
        "transmissions": list(TRANSMISSIONS),
        "protected_skeleton_fraction": PROTECTED_SKELETON_FRACTION,
        "protected_weights_alpha_beta_gamma": [1.0 / 3.0] * 3,
        "gradient_indicator": "local least-squares magnitude from matched Halton pilot",
        "ghg_conditioned_impedance": (
            "lambda_minus(s)=(K_plus/L_plus)*(0.65+0.70*normalized_pilot_gradient(s)); "
            "lambda_plus(s)=(K_minus/L_minus)*(0.65+0.70*normalized_pilot_gradient(s)); "
            "L_minus=r_vug, L_plus=0.5-r_vug"
        ),
        "dtn_geometric_impedance": (
            "cross-side receiving impedances: lambda_minus=K_plus/(0.5-r_vug); "
            "lambda_plus=K_minus/r_vug"
        ),
        "scalar_impedance": 8.0,
        "relaxation": RELAXATION,
        "maximum_iterations": maximum_iterations,
        "phs_power": 3,
        "polynomial_degree": 2,
        "stencil_size": STENCIL_SIZE,
        "evaluation_resolution": evaluation_resolution,
        "gates": {
            "q_net_less_than": Q_NET_LIMIT,
            "q_tail_less_than": Q_TAIL_LIMIT,
            "terminal_physical_mismatch_at_most": TERMINAL_MISMATCH_LIMIT,
            "conservation_defect_at_most": CONSERVATION_TOLERANCE,
            "relative_overshoot_at_most": OVERSHOOT_TOLERANCE,
            "condition_proxy_at_most": CONDITION_LIMIT,
            "full_and_collar_error_at_most_matching_monolithic_factor": ASSEMBLED_ERROR_FACTOR,
        },
        "qualification_rule": (
            "A training branch is eligible only with at least 18/20 simultaneous local, physical, "
            "and assembly passages. The best eligible training branch is frozen by passage count, "
            "terminal mismatch, assembled error, and condition, in that order. It qualifies only "
            "if the unchanged holdout also passes at least 18/20 rows."
        ),
    }


def allocation(method: str, problem: Any, budget: int, seed: int, pilot: Any) -> tuple[np.ndarray, int]:
    if method == "HALTON":
        return global_halton(problem, budget, seed), budget
    if method == "PROTECTED_GHG_EQUAL":
        return protected_weighted_allocation(
            problem,
            budget,
            seed,
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            pilot,
            skeleton_fraction=PROTECTED_SKELETON_FRACTION,
            green_length=GREEN_LENGTH,
            candidate_multiplier=CANDIDATE_MULTIPLIER,
        )
    raise ValueError(method)


def impedances(transmission: str, problem: Any, interface_points: np.ndarray, pilot: Any) -> tuple[np.ndarray, np.ndarray]:
    count = interface_points.shape[0]
    if transmission == "SCALAR_ROBIN_8":
        return np.full(count, 8.0), np.full(count, 8.0)
    radius = float(problem.metadata["radius"])
    # The receiving-side DtN scale is used, matching the declared cross-side
    # transmission convention: the minus solve receives the plus impedance
    # and the plus solve receives the minus impedance.
    base_minus = problem.k_plus / max(0.5 - radius, 0.08)
    base_plus = problem.k_minus / radius
    if transmission == "DTN_GEOMETRIC":
        return np.full(count, base_minus), np.full(count, base_plus)
    if transmission == "GHG_CONDITIONED_DTN":
        state = normalized_indicator(pilot(interface_points))
        modulation = 0.65 + 0.70 * state
        return base_minus * modulation, base_plus * modulation
    raise ValueError(transmission)


def apply_gates(metrics: dict[str, Any], monolithic: dict[str, Any]) -> dict[str, bool]:
    finite = all(np.isfinite(float(metrics[key])) for key in (
        "full_rmse", "collar_rmse", "relative_overshoot", "conservation_defect",
        "q_net", "q_tail", "physical_mismatch_terminal", "condition_proxy_max",
    ))
    result = {
        "gate_local": bool(
            finite
            and metrics["condition_proxy_max"] <= CONDITION_LIMIT
            and metrics["relative_overshoot"] <= OVERSHOOT_TOLERANCE
        ),
        "gate_physical": bool(
            finite
            and metrics["q_net"] < Q_NET_LIMIT
            and metrics["q_tail"] < Q_TAIL_LIMIT
            and metrics["physical_mismatch_terminal"] <= TERMINAL_MISMATCH_LIMIT
        ),
        "gate_assembly": bool(
            finite
            and metrics["conservation_defect"] <= CONSERVATION_TOLERANCE
            and metrics["full_rmse"] <= ASSEMBLED_ERROR_FACTOR * monolithic["full_rmse"]
            and metrics["collar_rmse"] <= ASSEMBLED_ERROR_FACTOR * monolithic["collar_rmse"]
        ),
    }
    result["gate_all"] = all(result.values())
    return result


def _p_less(challenger: list[float], reference: list[float]) -> float:
    difference = np.asarray(challenger) - np.asarray(reference)
    if difference.size == 0 or np.all(np.abs(difference) <= 1.0e-15):
        return 1.0
    try:
        return float(wilcoxon(difference, alternative="less", zero_method="pratt").pvalue)
    except ValueError:
        return 1.0


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for problem in sorted({row["problem"] for row in rows}):
        for allocation_name in ALLOCATIONS:
            for transmission in TRANSMISSIONS:
                subset = [
                    row for row in rows
                    if row["problem"] == problem
                    and row["allocation"] == allocation_name
                    and row["transmission"] == transmission
                ]
                summary.append({
                    "problem": problem,
                    "allocation": allocation_name,
                    "transmission": transmission,
                    "rows": len(subset),
                    "local_passes": sum(bool(row["gate_local"]) for row in subset),
                    "physical_passes": sum(bool(row["gate_physical"]) for row in subset),
                    "assembly_passes": sum(bool(row["gate_assembly"]) for row in subset),
                    "all_gate_passes": sum(bool(row["gate_all"]) for row in subset),
                    "local_pass_physical_fail": sum(
                        bool(row["gate_local"]) and not bool(row["gate_physical"]) for row in subset
                    ),
                    "median_q_net": median(float(row["q_net"]) for row in subset),
                    "median_q_tail": median(float(row["q_tail"]) for row in subset),
                    "median_terminal_mismatch": median(float(row["physical_mismatch_terminal"]) for row in subset),
                    "median_full_rmse": median(float(row["full_rmse"]) for row in subset),
                    "median_collar_rmse": median(float(row["collar_rmse"]) for row in subset),
                    "median_conservation_defect": median(float(row["conservation_defect"]) for row in subset),
                    "median_condition": median(float(row["condition_proxy_max"]) for row in subset),
                })
    required = 18 if PROFILE == "production" else 1
    training = [row for row in summary if "_TRAIN_" in row["problem"]]
    eligible = [row for row in training if int(row["all_gate_passes"]) >= required]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                -int(row["all_gate_passes"]),
                float(row["median_terminal_mismatch"]),
                float(row["median_full_rmse"]),
                float(row["median_condition"]),
                row["allocation"],
                row["transmission"],
            ),
        )
    if selected is None:
        holdout = None
        overall_qualified = False if PROFILE == "production" else None
    else:
        holdout = next(
            row for row in summary
            if "_HOLDOUT_" in row["problem"]
            and row["allocation"] == selected["allocation"]
            and row["transmission"] == selected["transmission"]
        )
        overall_qualified = bool(int(holdout["all_gate_passes"]) >= required) if PROFILE == "production" else None

    ghg_training_rows = [
        row for row in training
        if row["transmission"] == "GHG_CONDITIONED_DTN" and int(row["all_gate_passes"]) >= required
    ]
    best_ghg_training = min(
        ghg_training_rows,
        key=lambda row: (
            -int(row["all_gate_passes"]),
            float(row["median_terminal_mismatch"]),
            float(row["median_full_rmse"]),
            float(row["median_condition"]),
            row["allocation"],
        ),
    ) if ghg_training_rows else None
    if best_ghg_training is None:
        best_ghg_holdout = None
        ghg_qualified = False if PROFILE == "production" else None
    else:
        best_ghg_holdout = next(
            row for row in summary
            if "_HOLDOUT_" in row["problem"]
            and row["allocation"] == best_ghg_training["allocation"]
            and row["transmission"] == "GHG_CONDITIONED_DTN"
        )
        ghg_qualified = bool(
            int(best_ghg_holdout["all_gate_passes"]) >= required
        ) if PROFILE == "production" else None

    integrated_training = next(
        row for row in summary
        if "_TRAIN_" in row["problem"]
        and row["allocation"] == "PROTECTED_GHG_EQUAL"
        and row["transmission"] == "GHG_CONDITIONED_DTN"
    )
    integrated_holdout = next(
        row for row in summary
        if "_HOLDOUT_" in row["problem"]
        and row["allocation"] == "PROTECTED_GHG_EQUAL"
        and row["transmission"] == "GHG_CONDITIONED_DTN"
    )
    integrated_qualified = bool(
        int(integrated_training["all_gate_passes"]) >= required
        and int(integrated_holdout["all_gate_passes"]) >= required
    ) if PROFILE == "production" else None

    paired: list[dict[str, Any]] = []
    indexed = {
        (row["problem"], row["allocation"], row["transmission"], int(row["budget"]), int(row["seed"])): row
        for row in rows
    }
    for problem in sorted({row["problem"] for row in rows}):
        keys = sorted({(int(row["budget"]), int(row["seed"])) for row in rows if row["problem"] == problem})
        for allocation_name in ALLOCATIONS:
            challenger = [
                indexed[(problem, allocation_name, "GHG_CONDITIONED_DTN", budget, seed)]
                for budget, seed in keys
            ]
            reference = [
                indexed[(problem, allocation_name, "DTN_GEOMETRIC", budget, seed)]
                for budget, seed in keys
            ]
            for metric in (
                "q_net", "q_tail", "physical_mismatch_terminal", "full_rmse",
                "collar_rmse", "conservation_defect", "condition_proxy_max",
            ):
                a = [float(row[metric]) for row in challenger]
                b = [float(row[metric]) for row in reference]
                ratio = [left / max(right, 1.0e-300) for left, right in zip(a, b)]
                paired.append({
                    "problem": problem,
                    "allocation": allocation_name,
                    "challenger": "GHG_CONDITIONED_DTN",
                    "reference": "DTN_GEOMETRIC",
                    "metric": metric,
                    "pairs": len(ratio),
                    "median_ratio": median(ratio),
                    "wins": int(sum(value < 1.0 for value in ratio)),
                    "one_sided_p_challenger_less": _p_less(a, b),
                })
    negative_count = sum(int(row["local_pass_physical_fail"]) for row in summary)
    decision = {
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "required_passes_out_of_20": required,
        "best_overall_training_branch": selected,
        "best_overall_holdout_branch": holdout,
        "best_overall_branch_qualified": overall_qualified,
        "best_ghg_conditioned_training_branch": best_ghg_training,
        "best_ghg_conditioned_holdout_branch": best_ghg_holdout,
        "ghg_conditioned_transmission_qualified": ghg_qualified,
        "integrated_protected_ghg_conditioned_training_branch": integrated_training,
        "integrated_protected_ghg_conditioned_holdout_branch": integrated_holdout,
        "integrated_protected_ghg_conditioned_qualified": integrated_qualified,
        "local_pass_physical_fail_count_all_branches": negative_count,
        "negative_mechanism_observed": negative_count > 0 if PROFILE == "production" else None,
        "interpretation": (
            "Qualification requires simultaneous local, physical, and assembly passage; local accuracy alone is not success."
            if PROFILE == "production"
            else "Implementation screen only; no manuscript claim is authorized."
        ),
    }
    return summary, paired, decision


def load_histories(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def save_figures(summary: list[dict[str, Any]], history: list[dict[str, Any]], output: Path) -> None:
    labels = [f"{row['problem']}\n{row['allocation']}\n{row['transmission']}" for row in summary]
    positions = np.arange(len(summary))
    rows_count = np.asarray([max(int(row["rows"]), 1) for row in summary], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(max(11.0, 0.55 * len(summary)), 10.0), sharex=True)
    for axis, key, title, color in (
        (axes[0], "local_passes", "Local admissibility", "#4c78a8"),
        (axes[1], "physical_passes", "Physical contraction", "#f28e2b"),
        (axes[2], "assembly_passes", "Conservative assembly and accuracy", "#59a14f"),
    ):
        fractions = np.asarray([int(row[key]) for row in summary]) / rows_count
        axis.bar(positions, fractions, color=color)
        axis.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(positions, labels, rotation=75, ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "vug_coupling_gate_profiles.png", dpi=220)
    plt.close(fig)

    # A fixed representative rule: lowest seed and budget for each branch/problem.
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in history:
        key = (row["problem"], row["allocation"], row["transmission"])
        grouped.setdefault(key, []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), squeeze=False)
    for axis, configuration in zip(axes[0], ("TRAIN", "HOLDOUT")):
        for key, subset in sorted(grouped.items()):
            if f"_{configuration}_" not in key[0]:
                continue
            minimum_budget = min(int(float(row["budget"])) for row in subset)
            minimum_seed = min(int(float(row["seed"])) for row in subset if int(float(row["budget"])) == minimum_budget)
            selected = sorted(
                [row for row in subset if int(float(row["budget"])) == minimum_budget and int(float(row["seed"])) == minimum_seed],
                key=lambda row: int(float(row["iteration"])),
            )
            axis.semilogy(
                [int(float(row["iteration"])) for row in selected],
                [max(float(row["physical_mismatch"]), 1.0e-14) for row in selected],
                marker="o", markersize=2.5,
                label=f"{key[1]} / {key[2]}",
            )
        axis.set_title(configuration.title())
        axis.set_xlabel("Two-way Jacobi iteration")
        axis.set_ylabel("Trace/flux physical mismatch")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "vug_physical_mismatch_histories.png", dpi=220)
    plt.close(fig)


def main() -> None:
    contract = profile_contract()
    base = script_directory()
    output = Path(OUTPUT_DIRECTORY).resolve() if OUTPUT_DIRECTORY else base / f"results_vug_{PROFILE}_v1p4"
    output.mkdir(parents=True, exist_ok=True)
    cases = output / "cases"
    histories = output / "histories"
    cases.mkdir(exist_ok=True)
    histories.mkdir(exist_ok=True)
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
        for budget_value in contract["budgets"]:
            budget = int(budget_value)
            boundary_per_edge = max(8, int(math.ceil(math.sqrt(budget) * 0.72)))
            interface_count = max(12, int(math.ceil(math.sqrt(budget) * 0.90)))
            parameter = (np.arange(interface_count) + 0.5) / interface_count
            interface_points, _ = problem.sample_interface(parameter)
            for seed_value in contract["seeds"]:
                seed = int(seed_value)
                pilot_points = global_halton(problem, budget, seed)
                pilot_result = solve_monolithic(
                    problem, pilot_points, boundary_per_edge, interface_count,
                    int(contract["evaluation_resolution"]), EVALUATION_COLLAR_WIDTH,
                    STENCIL_SIZE, compute_condition=True,
                )
                pilot = pilot_gradient_from_result(pilot_result)
                for allocation_name in ALLOCATIONS:
                    points, skeleton_count = allocation(allocation_name, problem, budget, seed, pilot)
                    monolithic_result = pilot_result if allocation_name == "HALTON" else solve_monolithic(
                        problem, points, boundary_per_edge, interface_count,
                        int(contract["evaluation_resolution"]), EVALUATION_COLLAR_WIDTH,
                        STENCIL_SIZE, compute_condition=True,
                    )
                    for transmission in TRANSMISSIONS:
                        stem = case_stem(problem.name, f"{allocation_name}_{transmission}", budget, seed)
                        case_path = cases / f"{stem}.json"
                        history_path = histories / f"{stem}.csv"
                        if case_path.exists() and history_path.exists() and not FORCE:
                            row = load_json(case_path)
                            if row.get("protocol_hash") != contract["protocol_hash"]:
                                raise RuntimeError(f"Protocol mismatch in {case_path}")
                            rows.append(row)
                            log(f"RESUME {stem}")
                            continue
                        started = time.perf_counter()
                        lambda_minus, lambda_plus = impedances(
                            transmission, problem, interface_points, pilot
                        )
                        result = run_two_way_robin_schwarz(
                            problem,
                            points,
                            boundary_per_edge,
                            interface_count,
                            lambda_minus,
                            lambda_plus,
                            RELAXATION,
                            int(contract["maximum_iterations"]),
                            int(contract["evaluation_resolution"]),
                            EVALUATION_COLLAR_WIDTH,
                            STENCIL_SIZE,
                            compute_condition=True,
                        )
                        row = {
                            "implementation_version": IMPLEMENTATION_VERSION,
                            "profile": PROFILE,
                            "protocol_hash": contract["protocol_hash"],
                            "problem": problem.name,
                            "configuration": "TRAIN" if "_TRAIN_" in problem.name else "HOLDOUT",
                            "allocation": allocation_name,
                            "transmission": transmission,
                            "budget": budget,
                            "seed": seed,
                            "skeleton_count": int(skeleton_count),
                            "lambda_minus_min": float(np.min(lambda_minus)),
                            "lambda_minus_max": float(np.max(lambda_minus)),
                            "lambda_plus_min": float(np.min(lambda_plus)),
                            "lambda_plus_max": float(np.max(lambda_plus)),
                            "monolithic_full_rmse": float(monolithic_result.metrics["full_rmse"]),
                            "monolithic_collar_rmse": float(monolithic_result.metrics["collar_rmse"]),
                            "charged_seconds": float(time.perf_counter() - started),
                            **coverage_metrics(points),
                            **result.metrics,
                        }
                        row.update(apply_gates(row, monolithic_result.metrics))
                        atomic_json(case_path, row)
                        history_rows = [
                            {
                                "problem": problem.name,
                                "allocation": allocation_name,
                                "transmission": transmission,
                                "budget": budget,
                                "seed": seed,
                                **history_row,
                            }
                            for history_row in result.history
                        ]
                        atomic_csv(history_path, history_rows)
                        rows.append(row)
                        log(
                            f"DONE {stem}; q_net={row['q_net']:.4f}; q_tail={row['q_tail']:.4f}; "
                            f"terminal={row['physical_mismatch_terminal']:.3e}; all_gates={row['gate_all']}"
                        )

    rows.sort(key=lambda row: (
        row["problem"], int(row["budget"]), int(row["seed"]), row["allocation"], row["transmission"]
    ))
    summary, paired, decision = aggregate(rows)
    history_rows = load_histories(histories)
    atomic_csv(output / "campaign_metrics.csv", rows)
    atomic_csv(output / "branch_summary.csv", summary)
    atomic_csv(output / "ghg_conditioned_vs_geometric_dtn.csv", paired)
    atomic_csv(output / "campaign_history.csv", history_rows)
    atomic_json(output / "qualification_decision.json", decision)
    save_figures(summary, history_rows, output)
    completion = "production_complete.json" if PROFILE == "production" else "screen_complete.json"
    atomic_json(output / completion, {
        "completed_utc": utc_now(),
        "profile": PROFILE,
        "paper_evidence_authorized": PROFILE == "production",
        "protocol_hash": contract["protocol_hash"],
        "case_count": len(rows),
        "expected_case_count": len(make_problems()) * len(contract["budgets"]) * len(contract["seeds"]) * len(ALLOCATIONS) * len(TRANSMISSIONS),
    })
    log(f"COMPLETE profile={PROFILE}; cases={len(rows)}; evidence_authorized={PROFILE == 'production'}")
    atomic_csv(output / "scientific_manifest_sha256.csv", scientific_manifest(output))


if __name__ == "__main__":
    main()
