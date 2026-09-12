#!/usr/bin/env python3
"""
CGF_ESF_Selector_Freeze_and_SPE10_Preflight.py
================================================

Purpose
-------
This is the required adjudication step after the completed F campaign and
before any SPE10 pressure solve.  It does not retune F and it does not invent a
new transmission law.  It performs four bounded tasks:

1. independently audit the frozen localized-front (F) archive;
2. reproduce the paper-facing F statistics and gate counts;
3. freeze the conservative E/S/F selector after the prospective F failure;
4. optionally hash and fingerprint a user-supplied SPE10 Model 2, Layer 68
   permeability file so that the later P solver can be written against an
   immutable data contract.

Scientific decision encoded here
--------------------------------
The F fingerprint correctly recognized a localized layer, but the prescribed
RESIDUAL_ADAPTIVE allocation passed only 4/20 final selector rows.  Therefore
"localized front implies residual concentration" is rejected for Paper I.
The frozen Paper-I policy uses a global Halton branch as the conservative
primary choice outside the already-validated smooth equal-area case.  A
protected interface-aware GHG construction may remain a matched challenger in
P, but it is not retrospectively promoted to the selected branch.  DDM remains
off in P.

This script is deliberately a preflight, not the P solver.  SPE10 has a rough,
cellwise coefficient field.  A conservative pressure assembly, boundary/source
contract, and reference discretization must be frozen explicitly before a
collocation comparison is scientifically admissible.

Spyder use
----------
Edit only the USER CONFIGURATION block, then Run File.  Command-line arguments
are also provided for smoke tests and non-Spyder use.

Outputs
-------
Plain CSV, JSON, TXT, optional PNG/NPZ, an ordered log, a scientific SHA-256
manifest, and a completion marker.  Existing completed output is reused unless
FORCE is True.

Dependencies
------------
numpy, pandas, scipy, matplotlib (matplotlib is needed only when SPE10 data are
provided).  The script forces common numerical libraries to one thread before
importing NumPy.
"""

from __future__ import annotations

import os

# Single-thread defaults for the user's Dakini workstation and ordered logs.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import wilcoxon


# =============================================================================
# USER CONFIGURATION (safe to edit in Spyder)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The production F archive.  If the script is copied to the project root, this
# relative default normally resolves without further editing.
F_CAMPAIGN_DIR = SCRIPT_DIR / "localized_front_activation_campaign"

# Output is kept separate from the simulation archives.
OUTPUT_DIR = SCRIPT_DIR / "esf_selector_freeze_and_spe10_preflight"

# Optional SPE10 file.  Leave as None for the first adjudication run.  Supported
# inputs: 2-D .csv/.txt/.npy Layer-68 arrays; full .npy/.npz/.mat arrays; or a
# flat text file with an explicitly declared array order below.
SPE10_FILE: Optional[Path] = None

# Required only for ambiguous flat/full arrays.  Accepted values:
#   "auto"          : accept only unambiguous 2-D or labeled 3-D arrays
#   "k_j_i_c"       : reshape/interpret as (85, 220, 60), C order
#   "i_j_k_fortran" : reshape as (60, 220, 85), Fortran order, then transpose
#                     to canonical (k, j, i)
SPE10_ARRAY_ORDER = "auto"
SPE10_LAYER_ONE_BASED = 68

# The code never downloads a data file silently.  The supplied byte stream and
# extracted layer receive SHA-256 hashes in the frozen P preflight contract.
FORCE = False


# =============================================================================
# FROZEN PRIOR E/S EVIDENCE FROM MANUSCRIPT v2.5
# =============================================================================

# These facts are not recomputed here because their archives are not inputs to
# this bounded F adjudicator.  They are recorded verbatim with their provenance
# label so the later public release can replace the label by archive hashes.
PRIOR_ES_EVIDENCE: Dict[str, Any] = {
    "source": "CGF_First_Paper_Draft_v2p5_Smooth_Nonactivation_Update.pdf",
    "E": {
        "problem_class": "SMOOTH_HOMOGENEOUS",
        "selected_architecture": "EQUAL_AREA_STRATIFIED",
        "interface_enrichment": False,
        "domain_decomposition": False,
        "selector_final_passages": 20,
        "selector_rows": 20,
        "status": "CONFIRMED_NONACTIVATION",
    },
    "S": {
        "problem_class": "STATIC_DISCONTINUOUS_COEFFICIENT_INTERFACE",
        "training_protected_ghg_collar_wins": 13,
        "training_rows": 20,
        "training_median_collar_ratio_vs_halton": 0.314,
        "training_wilcoxon_p_two_sided": 0.0348,
        "holdout_protected_ghg_collar_wins": 11,
        "holdout_rows": 20,
        "holdout_median_collar_ratio_vs_halton": 0.896,
        "holdout_wilcoxon_p_two_sided": 0.3643,
        "general_accuracy_transfer": False,
        "coverage_floor_structural_transfer": True,
        "gradient_positive_role_supported": False,
        "ghg_conditioned_ddm_status": "FAIL_DEFER",
    },
}


EXPECTED_SEEDS = [11, 23, 37, 53, 71]
EXPECTED_BUDGETS = [120, 180, 260, 360]
EXPECTED_FAMILIES = [
    "HALTON",
    "EQUAL_AREA_STRATIFIED",
    "RESIDUAL_ADAPTIVE",
    "GHG_PROTECTED_CHALLENGER",
]


@dataclass(frozen=True)
class RunConfig:
    f_campaign_dir: str
    output_dir: str
    spe10_file: Optional[str]
    spe10_array_order: str
    spe10_layer_one_based: int
    force: bool


class OrderedLogger:
    """Small logger that writes the screen and file in the same order."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{stamp} UTC] {message}"
        print(line, flush=True)
        self._stream.write(line + "\n")

    def close(self) -> None:
        self._stream.close()


def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_bytes(canonical_json_bytes(obj))


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        bad = sorted(set(series[mapped.isna()].astype(str)))
        raise ValueError(f"Unrecognized Boolean values: {bad}")
    return mapped.astype(bool)


def verify_declared_manifest(manifest: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """Verify size and SHA-256 for every row relative to the manifest folder."""
    problems: List[Dict[str, Any]] = []
    declared = 0
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"file", "bytes", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest {manifest} lacks columns {sorted(required)}")
        for row in reader:
            declared += 1
            target = manifest.parent / row["file"]
            if not target.is_file():
                problems.append({"manifest": str(manifest), "file": row["file"], "issue": "MISSING"})
                continue
            actual_bytes = target.stat().st_size
            expected_bytes = int(row["bytes"])
            if actual_bytes != expected_bytes:
                problems.append(
                    {
                        "manifest": str(manifest),
                        "file": row["file"],
                        "issue": "SIZE",
                        "expected": expected_bytes,
                        "actual": actual_bytes,
                    }
                )
            actual_hash = sha256_file(target)
            if actual_hash != row["sha256"]:
                problems.append(
                    {
                        "manifest": str(manifest),
                        "file": row["file"],
                        "issue": "SHA256",
                        "expected": row["sha256"],
                        "actual": actual_hash,
                    }
                )
    return declared, problems


def audit_f_archive(root: Path) -> Dict[str, Any]:
    required = [
        "activation_decision_summary.txt",
        "campaign_allocation_gate_matrix.csv",
        "campaign_allocation_metrics.csv",
        "campaign_config.json",
        "campaign_manifest_sha256.csv",
        "campaign_node_metrics.csv",
        "campaign_run_index.csv",
        "campaign_selector_gate_matrix.csv",
        "frozen_experiment_contract.json",
        "physical_fingerprint.json",
        "production_complete.json",
        "seed_refinement_summary.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"F archive is incomplete; missing {missing}")

    production = json.loads((root / "production_complete.json").read_text(encoding="utf-8"))
    if production.get("status") != "COMPLETED" or int(production.get("completed_seed_budget_rows", -1)) != 20:
        raise ValueError(f"F production marker is not a completed 20-row rectangle: {production}")

    manifests = [root / "campaign_manifest_sha256.csv"] + sorted(root.glob("seed_*/run_manifest_sha256.csv"))
    if len(manifests) != 21:
        raise ValueError(f"Expected 1 campaign + 20 run manifests, found {len(manifests)}")
    declared = 0
    manifest_problems: List[Dict[str, Any]] = []
    for manifest in manifests:
        count, problems = verify_declared_manifest(manifest)
        declared += count
        manifest_problems.extend(problems)
    if manifest_problems:
        raise ValueError(f"F manifest audit failed; first problems: {manifest_problems[:5]}")

    npz_files = sorted(root.glob("seed_*/state_arrays.npz"))
    if len(npz_files) != 20:
        raise ValueError(f"Expected 20 F state arrays, found {len(npz_files)}")
    schema: Optional[Tuple[str, ...]] = None
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = tuple(data.files)
            if schema is None:
                schema = keys
            elif keys != schema:
                raise ValueError(f"NPZ schema mismatch in {npz_path}")
            for key in data.files:
                array = np.asarray(data[key])
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise ValueError(f"Non-finite state data in {npz_path}:{key}")

    return {
        "status": "PASS",
        "campaign_status": production.get("status"),
        "completed_seed_budget_rows": 20,
        "manifest_files": len(manifests),
        "declared_manifest_entries": declared,
        "manifest_issues": 0,
        "state_npz_files": len(npz_files),
        "state_npz_schema": list(schema or ()),
        "state_nonfinite_issues": 0,
        "frozen_experiment_sha256": production.get("frozen_experiment_sha256"),
    }


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Paired Wilcoxon arrays have different lengths")
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)


def summarize_f(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    metrics = pd.read_csv(root / "campaign_allocation_metrics.csv")
    allocation_gates = pd.read_csv(root / "campaign_allocation_gate_matrix.csv")
    selector_gates = pd.read_csv(root / "campaign_selector_gate_matrix.csv")
    allocation_gates["passed"] = bool_series(allocation_gates["passed"])
    selector_gates["passed"] = bool_series(selector_gates["passed"])

    if len(metrics) != 80:
        raise ValueError(f"Expected 80 allocation rows, found {len(metrics)}")
    if sorted(metrics["seed"].unique().tolist()) != EXPECTED_SEEDS:
        raise ValueError("Unexpected F seed set")
    if sorted(metrics["budget"].unique().tolist()) != EXPECTED_BUDGETS:
        raise ValueError("Unexpected F budget set")
    if set(metrics["allocation_family"].unique()) != set(EXPECTED_FAMILIES):
        raise ValueError("Unexpected F allocation families")
    if metrics.groupby(["seed", "budget", "allocation_family"]).size().ne(1).any():
        raise ValueError("F metrics do not form one matched row per seed-budget-family")

    all_gate = allocation_gates[allocation_gates["gate"] == "ALL_ADMISSIBILITY_GATES"]
    admissibility = all_gate.groupby("allocation_family")["passed"].agg(["sum", "count"])

    rows: List[Dict[str, Any]] = []
    for family in EXPECTED_FAMILIES:
        part = metrics[metrics["allocation_family"] == family]
        rows.append(
            {
                "allocation_family": family,
                "rows": len(part),
                "full_rmse_median": float(part["full_rmse"].median()),
                "front_collar_rmse_median": float(part["front_collar_rmse"].median()),
                "h1_rmse_median": float(part["h1_seminorm_rmse"].median()),
                "independent_residual_relative_median": float(part["independent_residual_relative"].median()),
                "condition_proxy_max": float(part["condition_proxy"].max()),
                "charged_time_median_s": float(part["charged_total_time_s"].median()),
                "admissibility_passes": int(admissibility.loc[family, "sum"]),
                "admissibility_rows": int(admissibility.loc[family, "count"]),
            }
        )
    family_summary = pd.DataFrame(rows)

    refinement = (
        metrics.groupby(["allocation_family", "budget"], sort=False)
        .agg(
            seeds=("seed", "count"),
            full_rmse_median=("full_rmse", "median"),
            front_collar_rmse_median=("front_collar_rmse", "median"),
            h1_rmse_median=("h1_seminorm_rmse", "median"),
            charged_time_median_s=("charged_total_time_s", "median"),
        )
        .reset_index()
    )

    selected = metrics[metrics["allocation_family"] == "RESIDUAL_ADAPTIVE"].sort_values(["seed", "budget"])
    comparisons: List[Dict[str, Any]] = []
    for comparator in ["HALTON", "EQUAL_AREA_STRATIFIED", "GHG_PROTECTED_CHALLENGER"]:
        reference = metrics[metrics["allocation_family"] == comparator].sort_values(["seed", "budget"])
        if not np.array_equal(selected[["seed", "budget"]].to_numpy(), reference[["seed", "budget"]].to_numpy()):
            raise ValueError(f"Pairing mismatch for {comparator}")
        for metric in ["full_rmse", "front_collar_rmse", "h1_seminorm_rmse", "charged_total_time_s"]:
            a = selected[metric].to_numpy(float)
            b = reference[metric].to_numpy(float)
            ratio = a / b
            comparisons.append(
                {
                    "selected_family": "RESIDUAL_ADAPTIVE",
                    "comparator": comparator,
                    "metric": metric,
                    "selected_wins": int(np.sum(a < b)),
                    "paired_rows": len(a),
                    "median_selected_to_comparator_ratio": float(np.median(ratio)),
                    "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratio)))),
                    "wilcoxon_p_two_sided": safe_wilcoxon(a, b),
                }
            )
    pairwise = pd.DataFrame(comparisons)

    final = selector_gates[selector_gates["gate"] == "SELECTOR_FINAL_GATE"].copy()
    gate_counts = selector_gates.groupby("gate")["passed"].agg(["sum", "count"]).reset_index()
    final_passes = int(final["passed"].sum())

    def endpoint_ratio(family: str, metric: str) -> float:
        sub = refinement[refinement["allocation_family"] == family].set_index("budget")
        return float(sub.loc[360, metric] / sub.loc[120, metric])

    f_decision = {
        "problem_class": "FROZEN_LOCALIZED_FRONT",
        "preselected_architecture": "RESIDUAL_ADAPTIVE",
        "fingerprint_activation_rows": int(
            selector_gates[selector_gates["gate"] == "S0_FROZEN_LOCALIZATION_ACTIVATION"]["passed"].sum()
        ),
        "selector_final_passages": final_passes,
        "selector_rows": len(final),
        "status": "FAIL_DEFER" if final_passes < len(final) else "PASS",
        "dominant_failed_gate": "S2B_FRONT_COLLAR_COMPETITIVE",
        "front_collar_competitive_passes": int(
            selector_gates[selector_gates["gate"] == "S2B_FRONT_COLLAR_COMPETITIVE"]["passed"].sum()
        ),
        "full_rmse_competitive_passes": int(
            selector_gates[selector_gates["gate"] == "S2_FULL_RMSE_SAFEGUARD"]["passed"].sum()
        ),
        "h1_competitive_passes": int(
            selector_gates[selector_gates["gate"] == "S3_H1_COMPETITIVE"]["passed"].sum()
        ),
        "selected_admissible_passes": int(
            selector_gates[selector_gates["gate"] == "S1_SELECTED_ADMISSIBLE"]["passed"].sum()
        ),
        "cost_competitive_passes": int(
            selector_gates[selector_gates["gate"] == "S4_COST_COMPETITIVE"]["passed"].sum()
        ),
        "residual_adaptive_full_rmse_N360_over_N120": endpoint_ratio(
            "RESIDUAL_ADAPTIVE", "full_rmse_median"
        ),
        "halton_full_rmse_N360_over_N120": endpoint_ratio("HALTON", "full_rmse_median"),
        "interpretation": (
            "The physical fingerprint detected localization, but local concentration was not a reliable "
            "allocation response. The selected branch failed 16/20 final rows; conditioning and cost were "
            "not the limiting mechanisms. Global coverage and conservation were more predictive of accuracy."
        ),
    }
    return family_summary, refinement, pairwise, {"decision": f_decision, "gate_counts": gate_counts}


def freeze_selector(f_audit: Mapping[str, Any], f_decision: Mapping[str, Any]) -> Dict[str, Any]:
    if f_audit.get("status") != "PASS":
        raise ValueError("Cannot freeze selector from an unaudited F archive")
    if f_decision.get("status") != "FAIL_DEFER":
        raise ValueError("This frozen contract is specifically the prospectively observed F failure branch")

    return {
        "contract_version": "CGF-ESF-SELECTOR-1.0",
        "freeze_stage": "after independent E, S training/holdout, and F production",
        "retrospective_F_retuning_allowed": False,
        "rules": [
            {
                "priority": 1,
                "if": "smooth homogeneous coefficient; no localized active layer",
                "select": "EQUAL_AREA_STRATIFIED",
                "interface_enrichment": False,
                "domain_decomposition": False,
                "evidence": "E selector 20/20",
            },
            {
                "priority": 2,
                "if": "localized state layer with homogeneous coefficient",
                "select": "HALTON",
                "interface_enrichment": False,
                "domain_decomposition": False,
                "evidence": "F residual activation failed; Halton is the coverage-preserving fallback",
            },
            {
                "priority": 3,
                "if": "rough or discontinuous coefficient, including SPE10 P",
                "select": "HALTON_PRIMARY",
                "matched_challenger": "PROTECTED_INTERFACE_GHG_WITH_COVERAGE_FLOOR",
                "interface_enrichment": "challenger only",
                "domain_decomposition": False,
                "evidence": (
                    "S supports the floor structurally but not a general accuracy transfer; F blocks "
                    "gradient/residual activation. P must preserve the primary/challenger labels."
                ),
            },
            {
                "priority": 4,
                "if": "moving interface, front-interface crossing, vug/matrix coupling, or DDM request",
                "select": "DEFER_PAPER_II",
                "domain_decomposition": False,
                "evidence": "Paper-I publication boundary and D4/R4 fail-defer",
            },
        ],
        "claim_boundary": {
            "authorized": [
                "The selector can correctly decline enrichment on E.",
                "The prospective localized-front activation rule failed on F.",
                "A protected low-discrepancy floor is a coverage safeguard, not a universal accuracy guarantee.",
                "P may test a frozen Halton-primary/protected-GHG-challenger comparison without DDM.",
            ],
            "blocked": [
                "Localized fronts generally benefit from residual-adaptive concentration.",
                "GHG dominates Halton globally.",
                "The E/S/F suite validates a positive cross-regime activation map.",
                "Paper I establishes a GHG-conditioned DDM method.",
            ],
        },
        "input_evidence": {
            "prior_ES": PRIOR_ES_EVIDENCE,
            "F_frozen_experiment_sha256": f_audit.get("frozen_experiment_sha256"),
            "F_selector_final_passages": f_decision.get("selector_final_passages"),
            "F_selector_rows": f_decision.get("selector_rows"),
        },
    }


def candidate_numeric_array(container: Mapping[str, Any]) -> Tuple[str, np.ndarray]:
    preferred = ["perm", "permx", "PERMX", "Kx", "kx", "rock"]
    for key in preferred:
        if key in container:
            value = container[key]
            if isinstance(value, np.ndarray) and value.size >= 60 * 220:
                return key, np.asarray(value)
    candidates: List[Tuple[str, np.ndarray]] = []
    for key, value in container.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and value.size >= 60 * 220:
            candidates.append((str(key), np.asarray(value)))
    if len(candidates) != 1:
        names = [(key, list(value.shape)) for key, value in candidates]
        raise ValueError(f"Ambiguous SPE10 numeric arrays; candidates={names}")
    return candidates[0]


def canonical_layer_from_array(
    array: np.ndarray, order: str, layer_one_based: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a canonical (j=220, i=60) layer without silently guessing flat order."""
    a = np.asarray(array)
    a = np.squeeze(a)
    layer_index = layer_one_based - 1
    if not 0 <= layer_index < 85:
        raise ValueError("SPE10 layer must lie in 1..85")

    # Direct two-dimensional layer.
    if a.ndim == 2 and a.shape in {(220, 60), (60, 220)}:
        layer = a if a.shape == (220, 60) else a.T
        return np.asarray(layer, dtype=float), {"source_shape": list(a.shape), "interpretation": "2D_LAYER"}

    # Some .mat files store permeability as (ncell, 3).  Use Kx for the
    # coefficient fingerprint; anisotropy can be added later when Ky is labeled.
    if a.ndim == 2 and 3 in a.shape and max(a.shape) == 60 * 220 * 85:
        a = a if a.shape[1] == 3 else a.T
        a = a[:, 0]

    if order == "auto":
        if a.shape == (85, 220, 60):
            canonical = a
            interpretation = "LABELED_K_J_I"
        elif a.shape == (220, 60, 85):
            canonical = np.transpose(a, (2, 0, 1))
            interpretation = "LABELED_J_I_K"
        elif a.shape == (60, 220, 85):
            canonical = np.transpose(a, (2, 1, 0))
            interpretation = "LABELED_I_J_K"
        else:
            raise ValueError(
                f"SPE10 array shape/order is ambiguous: {a.shape}. Set SPE10_ARRAY_ORDER explicitly."
            )
    elif order == "k_j_i_c":
        flat = np.asarray(a).reshape(-1)
        if flat.size not in {60 * 220 * 85, 3 * 60 * 220 * 85}:
            raise ValueError(f"Unexpected flat SPE10 size {flat.size}")
        if flat.size == 3 * 60 * 220 * 85:
            flat = flat[: 60 * 220 * 85]
        canonical = flat.reshape((85, 220, 60), order="C")
        interpretation = "EXPLICIT_K_J_I_C"
    elif order == "i_j_k_fortran":
        flat = np.asarray(a).reshape(-1)
        if flat.size not in {60 * 220 * 85, 3 * 60 * 220 * 85}:
            raise ValueError(f"Unexpected flat SPE10 size {flat.size}")
        if flat.size == 3 * 60 * 220 * 85:
            flat = flat[: 60 * 220 * 85]
        ijk = flat.reshape((60, 220, 85), order="F")
        canonical = np.transpose(ijk, (2, 1, 0))
        interpretation = "EXPLICIT_I_J_K_FORTRAN"
    else:
        raise ValueError(f"Unknown SPE10_ARRAY_ORDER={order!r}")

    layer = np.asarray(canonical[layer_index], dtype=float)
    return layer, {"source_shape": list(a.shape), "interpretation": interpretation}


def load_spe10_layer(path: Path, order: str, layer_one_based: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    suffix = path.suffix.lower()
    source_key = "direct"
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            source_key, array = candidate_numeric_array({key: data[key] for key in data.files})
    elif suffix == ".mat":
        source_key, array = candidate_numeric_array(loadmat(path))
    elif suffix in {".csv", ".txt", ".dat", ".inc"}:
        try:
            delimiter = "," if suffix == ".csv" else None
            array = np.loadtxt(path, delimiter=delimiter)
        except ValueError as exc:
            raise ValueError(
                "The text file is not a plain numeric array. Export Layer 68 as a 220x60 CSV or "
                "convert the official deck to NPY/NPZ with an explicit order."
            ) from exc
    else:
        raise ValueError(f"Unsupported SPE10 suffix {suffix!r}")

    layer, meta = canonical_layer_from_array(array, order, layer_one_based)
    if layer.shape != (220, 60):
        raise ValueError(f"Canonical Layer 68 shape should be (220,60), found {layer.shape}")
    if not np.isfinite(layer).all():
        raise ValueError("SPE10 layer contains non-finite values")
    if np.any(layer <= 0):
        raise ValueError(
            "SPE10 layer contains non-positive permeability. An inactive-cell mask and boundary "
            "contract must be supplied rather than silently clipping values."
        )
    meta.update({"source_key": source_key, "canonical_layer_shape": [220, 60]})
    return layer, meta


def spe10_fingerprint(layer: np.ndarray) -> Dict[str, Any]:
    logk = np.log10(layer)
    q01, q50, q99 = np.quantile(layer, [0.01, 0.50, 0.99])
    dx = np.abs(np.diff(logk, axis=1))
    dy = np.abs(np.diff(logk, axis=0))
    jump_threshold_decades = 1.0
    jump_edges = int(np.sum(dx >= jump_threshold_decades) + np.sum(dy >= jump_threshold_decades))
    total_edges = int(dx.size + dy.size)
    grad = np.zeros_like(logk)
    grad[:, 1:-1] += 0.5 * np.abs(logk[:, 2:] - logk[:, :-2])
    grad[1:-1, :] += 0.5 * np.abs(logk[2:, :] - logk[:-2, :])
    grad_threshold = float(np.quantile(grad, 0.90))
    return {
        "permeability_min": float(np.min(layer)),
        "permeability_q01": float(q01),
        "permeability_median": float(q50),
        "permeability_q99": float(q99),
        "permeability_max": float(np.max(layer)),
        "q99_to_q01_contrast": float(q99 / q01),
        "full_max_to_min_contrast": float(np.max(layer) / np.min(layer)),
        "log10_span_decades": float(np.max(logk) - np.min(logk)),
        "one_decade_jump_edge_fraction": float(jump_edges / total_edges),
        "gradient_active_fraction_top10pct": float(np.mean(grad >= grad_threshold)),
        "gradient_threshold": grad_threshold,
        "classification": "ROUGH_DISCONTINUOUS_COEFFICIENT",
        "frozen_primary_allocation": "HALTON_PRIMARY",
        "frozen_matched_challenger": "PROTECTED_INTERFACE_GHG_WITH_COVERAGE_FLOOR",
        "domain_decomposition": False,
    }


def write_spe10_outputs(
    source: Path,
    layer: np.ndarray,
    load_meta: Mapping[str, Any],
    output_dir: Path,
    layer_one_based: int,
) -> Dict[str, Any]:
    source_hash = sha256_file(source)
    layer_path = output_dir / "spe10_model2_layer68_coefficients.npz"
    np.savez_compressed(layer_path, permeability=layer)
    layer_hash = sha256_file(layer_path)
    fingerprint = spe10_fingerprint(layer)

    figure_path = output_dir / "spe10_model2_layer68_log10_permeability.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    image = ax.imshow(np.log10(layer), origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("SPE10 Model 2, Layer 68: log10 permeability")
    ax.set_xlabel("i cell (1..60)")
    ax.set_ylabel("j cell (1..220)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10 permeability in source units")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    record = {
        "status": "DATA_PREFLIGHT_COMPLETE_SOLVER_NOT_YET_AUTHORIZED",
        "source_path": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "layer_one_based": layer_one_based,
        "load_metadata": dict(load_meta),
        "canonical_layer_npz": layer_path.name,
        "canonical_layer_npz_sha256": layer_hash,
        "fingerprint": fingerprint,
        "remaining_protocol_locks": [
            "steady single-phase Darcy equation and pressure gauge",
            "source/well strengths and support radii",
            "outer boundary conditions",
            "inactive-cell treatment",
            "conservative rough-coefficient assembly",
            "fine-grid reference solve and conservation tolerance",
            "matched node budgets and timing boundary",
        ],
    }
    write_json(output_dir / "spe10_model2_layer68_preflight.json", record)
    pd.DataFrame([fingerprint]).to_csv(output_dir / "spe10_model2_layer68_fingerprint.csv", index=False)
    return record


def paper_tables(
    family_summary: pd.DataFrame,
    gate_counts: pd.DataFrame,
    f_decision: Mapping[str, Any],
    selector_contract: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    activation_rows = [
        {
            "regime": "E",
            "fingerprint": "smooth homogeneous; no localized layer",
            "prospective_selection": "EQUAL_AREA_STRATIFIED",
            "final_gate": "20/20 pass",
            "paper_status": "CONFIRMED NONACTIVATION",
            "authorized_interpretation": "minimum-complexity branch retained",
        },
        {
            "regime": "S",
            "fingerprint": "static discontinuous coefficient interface",
            "prospective_selection": "protected GHG candidate",
            "final_gate": "accuracy transfer not established",
            "paper_status": "QUALIFIED / FLOOR STRUCTURAL ONLY",
            "authorized_interpretation": "coverage floor transfers; no universal accuracy claim",
        },
        {
            "regime": "F",
            "fingerprint": "frozen localized state layer; homogeneous coefficient",
            "prospective_selection": "RESIDUAL_ADAPTIVE",
            "final_gate": f"{f_decision['selector_final_passages']}/{f_decision['selector_rows']} pass",
            "paper_status": "ACTIVATION REJECTED",
            "authorized_interpretation": "localization detection did not imply beneficial local concentration",
        },
        {
            "regime": "P",
            "fingerprint": "SPE10 Layer 68 rough coefficient",
            "prospective_selection": "HALTON primary; protected interface-GHG challenger",
            "final_gate": "not run",
            "paper_status": "PREFLIGHT / PROTOCOL LOCK REQUIRED",
            "authorized_interpretation": "no DDM and no retrospective promotion of challenger",
        },
    ]
    claim_rows = [
        {"claim": claim, "status": "AUTHORIZED"}
        for claim in selector_contract["claim_boundary"]["authorized"]
    ] + [
        {"claim": claim, "status": "BLOCKED"}
        for claim in selector_contract["claim_boundary"]["blocked"]
    ]
    return pd.DataFrame(activation_rows), pd.DataFrame(claim_rows)


def write_scientific_manifest(output_dir: Path, filenames: Iterable[str]) -> None:
    rows: List[Dict[str, Any]] = []
    for name in sorted(set(filenames)):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Cannot manifest missing output {path}")
        rows.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(rows, columns=["file", "bytes", "sha256"]).to_csv(
        output_dir / "scientific_manifest_sha256.csv", index=False
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-dir", type=Path, default=F_CAMPAIGN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--spe10-file", type=Path, default=SPE10_FILE)
    parser.add_argument(
        "--spe10-array-order",
        choices=["auto", "k_j_i_c", "i_j_k_fortran"],
        default=SPE10_ARRAY_ORDER,
    )
    parser.add_argument("--spe10-layer", type=int, default=SPE10_LAYER_ONE_BASED)
    parser.add_argument("--force", action="store_true", default=FORCE)
    args = parser.parse_args(argv)
    return RunConfig(
        f_campaign_dir=str(args.f_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        spe10_file=str(args.spe10_file.resolve()) if args.spe10_file else None,
        spe10_array_order=args.spe10_array_order,
        spe10_layer_one_based=args.spe10_layer,
        force=bool(args.force),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    f_root = Path(config.f_campaign_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completion_path = output_dir / "production_complete.json"
    config_path = output_dir / "run_config.json"
    proposed_config_hash = hashlib.sha256(canonical_json_bytes(asdict(config))).hexdigest()
    if completion_path.is_file() and config_path.is_file() and not config.force:
        previous = json.loads(completion_path.read_text(encoding="utf-8"))
        if previous.get("run_config_sha256") == proposed_config_hash:
            print(f"AUTO-RESUME: completed matching output retained at {output_dir}")
            return 0
        raise RuntimeError(
            f"Output {output_dir} contains a completion marker for a different config. "
            "Choose another OUTPUT_DIR or set FORCE=True."
        )

    if output_dir.exists() and config.force:
        # Delete only known products in the explicit output folder, never a broad tree.
        known = [
            "run_config.json",
            "run.log",
            "f_archive_audit.json",
            "f_family_summary.csv",
            "f_refinement_summary.csv",
            "f_pairwise_statistics.csv",
            "f_selector_gate_counts.csv",
            "f_decision.json",
            "esf_frozen_selector_contract.json",
            "paper_table_activation_ecology.csv",
            "paper_claim_status.csv",
            "next_action.txt",
            "spe10_model2_layer68_coefficients.npz",
            "spe10_model2_layer68_log10_permeability.png",
            "spe10_model2_layer68_preflight.json",
            "spe10_model2_layer68_fingerprint.csv",
            "scientific_manifest_sha256.csv",
            "production_complete.json",
        ]
        for name in known:
            target = output_dir / name
            if target.is_file():
                target.unlink()

    write_json(config_path, asdict(config))
    logger = OrderedLogger(output_dir / "run.log")
    start = time.time()
    scientific_files: List[str] = ["run_config.json"]
    try:
        logger.write("Starting frozen E/S/F adjudication; no retrospective F tuning is permitted.")
        logger.write(f"Auditing F archive: {f_root}")
        f_audit = audit_f_archive(f_root)
        write_json(output_dir / "f_archive_audit.json", f_audit)
        scientific_files.append("f_archive_audit.json")
        logger.write(
            f"F archive audit PASS: {f_audit['declared_manifest_entries']} declared hashes and "
            f"{f_audit['state_npz_files']} NPZ states verified."
        )

        family_summary, refinement, pairwise, decision_bundle = summarize_f(f_root)
        f_decision = decision_bundle["decision"]
        gate_counts = decision_bundle["gate_counts"]
        family_summary.to_csv(output_dir / "f_family_summary.csv", index=False)
        refinement.to_csv(output_dir / "f_refinement_summary.csv", index=False)
        pairwise.to_csv(output_dir / "f_pairwise_statistics.csv", index=False)
        gate_counts.to_csv(output_dir / "f_selector_gate_counts.csv", index=False)
        write_json(output_dir / "f_decision.json", f_decision)
        scientific_files.extend(
            [
                "f_family_summary.csv",
                "f_refinement_summary.csv",
                "f_pairwise_statistics.csv",
                "f_selector_gate_counts.csv",
                "f_decision.json",
            ]
        )
        logger.write(
            f"F adjudication: {f_decision['selector_final_passages']}/{f_decision['selector_rows']} "
            "final passages; activation branch FAIL/DEFER."
        )

        selector_contract = freeze_selector(f_audit, f_decision)
        write_json(output_dir / "esf_frozen_selector_contract.json", selector_contract)
        scientific_files.append("esf_frozen_selector_contract.json")
        logger.write("Frozen selector written: Halton-primary conservative P branch; DDM off.")

        activation_table, claim_table = paper_tables(
            family_summary, gate_counts, f_decision, selector_contract
        )
        activation_table.to_csv(output_dir / "paper_table_activation_ecology.csv", index=False)
        claim_table.to_csv(output_dir / "paper_claim_status.csv", index=False)
        scientific_files.extend(["paper_table_activation_ecology.csv", "paper_claim_status.csv"])

        spe10_status: Dict[str, Any]
        if config.spe10_file is None:
            spe10_status = {
                "status": "BLOCKED_DATA_FILE_NOT_SUPPLIED",
                "required": (
                    "Set SPE10_FILE to an explicit Model-2 permeability file, or pass --spe10-file. "
                    "For ambiguous flat files also set SPE10_ARRAY_ORDER."
                ),
            }
            logger.write("SPE10 data preflight not run: no file supplied. This is a declared blocker, not a failure.")
        else:
            source = Path(config.spe10_file)
            if not source.is_file():
                raise FileNotFoundError(f"SPE10 file not found: {source}")
            layer, load_meta = load_spe10_layer(source, config.spe10_array_order, config.spe10_layer_one_based)
            spe10_status = write_spe10_outputs(
                source, layer, load_meta, output_dir, config.spe10_layer_one_based
            )
            scientific_files.extend(
                [
                    "spe10_model2_layer68_coefficients.npz",
                    "spe10_model2_layer68_log10_permeability.png",
                    "spe10_model2_layer68_preflight.json",
                    "spe10_model2_layer68_fingerprint.csv",
                ]
            )
            logger.write("SPE10 Layer 68 data hash and coefficient fingerprint complete.")

        next_action = (
            "NEXT ACTION\n"
            "===========\n"
            "F is closed as a prospective negative activation result. Do not tune another localized "
            "allocation in Paper I. The next numerical solver may be constructed only after the P data "
            "preflight and the conservative rough-coefficient protocol are locked. The frozen P comparison "
            "is HALTON_PRIMARY versus PROTECTED_INTERFACE_GHG_WITH_COVERAGE_FLOOR as a matched challenger, "
            "with DDM off and no retrospective promotion.\n\n"
            f"SPE10 preflight status: {spe10_status['status']}\n"
        )
        (output_dir / "next_action.txt").write_text(next_action, encoding="utf-8")
        scientific_files.append("next_action.txt")

        write_scientific_manifest(output_dir, scientific_files)
        config_hash = sha256_file(config_path)
        completion = {
            "status": "COMPLETED",
            "scientific": True,
            "elapsed_s": time.time() - start,
            "run_config_sha256": config_hash,
            "f_status": f_decision["status"],
            "f_selector_final_passages": f_decision["selector_final_passages"],
            "spe10_preflight_status": spe10_status["status"],
            "note": "run.log and completion marker are intentionally outside the scientific manifest",
        }
        logger.write("Scientific manifest written after all scientific products.")
        logger.close()
        write_json(completion_path, completion)
        print(f"COMPLETED: {output_dir}", flush=True)
        return 0
    except Exception as exc:
        logger.write(f"FAILED: {type(exc).__name__}: {exc}")
        logger.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
