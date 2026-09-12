#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGF Paper I: frozen SPE10 Model 2, Layer 68 comparison.

This is the first and only Paper-I SPE10 P campaign.  It is deliberately a
prospective comparison, not a tuning program:

    HALTON_PRIMARY
        versus
    PROTECTED_INTERFACE_GHG_WITH_COVERAGE_FLOOR

The challenger retains a 45% scrambled-Halton coverage skeleton and uses the
remaining budget for deterministic coefficient-interface enrichment.  The
unsupported solution-gradient/residual component is not reintroduced.  Domain
decomposition is off.  The code never promotes or retunes the challenger.

The PDE is steady single-phase Darcy flow on the full 60 x 220 Layer-68 field,

    -div(K grad p) = 0,

with p=1 on the left, p=0 on the right, no-flow at top and bottom, and no
wells/sources.  Kx and Ky are used in mD.  Porosity is audited but does not
enter this steady incompressible pressure equation.  All finite positive-
permeability cells are active, including cells whose porosity is zero.

Numerical engines
-----------------
* Reference: conservative cell-centred finite volume on native, 2x, and 4x
  cell subdivisions, with harmonic face transmissibilities.
* Collocation comparison: the same conservative sparse linear-triangle
  Galerkin/collocation engine for both point allocations.  Its interior nodal
  equations are control-volume-like flux balances; left/right values are
  imposed strongly and top/bottom no-flow is natural.
* Conventional matched baseline: structured finite volume on a grid with a
  degree-of-freedom count as close as possible to each collocation budget,
  using fixed log-geometric block permeability upscaling.

The script is Spyder-friendly, single-threaded, ordered, restartable, and uses
only NumPy, SciPy, and Matplotlib.  It writes plain CSV tables, JSON contracts,
compressed state archives, publication-readable PNG figures, a SHA-256
manifest, and a completion/next-action handoff.

HOW TO RUN IN SPYDER
--------------------
1. Put this script beside SPE10.zip (the supplied archive), or set
   SPE10_INPUT below to its absolute path.
2. Leave RUN_MODE="production" for the scientific run.
3. Run the file.  Safe reruns resume completed reference levels and cases.

For a quick installation check only, set RUN_MODE="smoke".  Smoke outputs are
written to a different folder and are not scientific evidence.

Scientific decisions are collected in build_protocol(); changing one of them
creates a different contract hash.  Do not change those values after looking
at campaign outcomes.

Author-directed implementation for Arturo Ortiz-Tapia and Martin A.
Diaz-Viera, 2026-09-02.
"""

from __future__ import annotations

# Enforce the user's single-thread execution preference before NumPy/SciPy load.
import os

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import csv
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from scipy.io import loadmat
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, onenormest, splu
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import qmc


# =============================================================================
# USER-EDITABLE PATH/RUN CONTROLS (not scientific tuning parameters)
# =============================================================================

SPE10_INPUT: str | None = None
OUTPUT_DIR = "spe10_layer68_frozen_campaign"
RUN_MODE = "production"       # "production" or "smoke"
RESUME = True
FORCE_NEW_OUTPUT = False       # True permits replacing a conflicting output.
MAKE_FIGURES = True

# Environment overrides are useful outside Spyder and do not alter production
# scientific constants.  Example: CGF_SPE10_SMOKE=1 python this_script.py
if os.environ.get("CGF_SPE10_SMOKE", "0") == "1":
    RUN_MODE = "smoke"
if os.environ.get("CGF_SPE10_INPUT"):
    SPE10_INPUT = os.environ["CGF_SPE10_INPUT"]
if os.environ.get("CGF_SPE10_OUTPUT"):
    OUTPUT_DIR = os.environ["CGF_SPE10_OUTPUT"]


# =============================================================================
# FROZEN PAPER-I CONSTANTS
# =============================================================================

IMPLEMENTATION_VERSION = "CGF-SPE10-P-1.0"
SELECTOR_CONTRACT_VERSION = "CGF-ESF-SELECTOR-1.0"

NX_NATIVE = 60
NY_NATIVE = 220
NZ_NATIVE = 85
LAYER_ONE_BASED = 68
DX_FT = 20.0
DY_FT = 10.0
LX_FT = NX_NATIVE * DX_FT
LY_FT = NY_NATIVE * DY_FT

P_LEFT = 1.0
P_RIGHT = 0.0
SEEDS = (11, 23, 37, 53, 71)
INTERIOR_BUDGETS = (600, 1200, 2400, 4800)
REFERENCE_FACTORS = (1, 2, 4)

PROTECTED_HALTON_FRACTION = 0.45
CANDIDATE_MULTIPLIER = 16
INTERFACE_POSITIVE_JUMP_QUANTILE = 0.50
INTERFACE_LENGTH_CELLS = 2.5
INDICATOR_DENSITY_FLOOR = 0.05
INTERIOR_MARGIN_COEFFICIENT = 0.15

MASS_BALANCE_TOL = 1.0e-8
ALGEBRAIC_RESIDUAL_TOL = 1.0e-8
BOUNDARY_VALUE_TOL = 1.0e-12
CONDITION_PROXY_LIMIT = 1.0e10
REFERENCE_PRESSURE_DECREASE_REQUIRED = True

EXPECTED_HASHES = {
    "SPE10.zip": "f4063515f426238a486141424107b112d546a75aa5d3c5defa368783c119abe6",
    "spe_perm.dat": "7e5d613bf44dabaec2ed2dd7ebe87e79515bb77ac071dad9d33823efc37c7914",
    "spe_phi.dat": "53f95e004a0803b04293539e0b1d5f82f2d25d341ca300cb117f1414d9dc7d8a",
    "spe10_rock.mat": "0d53918f3b7a67636e43cda30e9425711142fcc5dc20dc9f5326b2f94b3ca306",
}
MILLI_DARCY_TO_M2 = 9.86923266716013e-16

PRIMARY = "HALTON_PRIMARY"
CHALLENGER = "PROTECTED_INTERFACE_GHG_WITH_COVERAGE_FLOOR"
CONVENTIONAL = "STRUCTURED_FV_LOG_GEOMETRIC_UPSCALED"


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_figure(fig: Any, path: Path) -> None:
    """Write a PNG completely before replacing any prior figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, format="png", bbox_inches="tight")
    os.replace(tmp, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value cannot be serialized: {value}")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class RunLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")


def script_directory() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


def resolve_input(user_value: str | None) -> Path:
    if user_value:
        p = Path(user_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured SPE10_INPUT does not exist: {p}")
        return p

    roots = [script_directory(), Path.cwd().resolve()]
    names = ("SPE10.zip", "spe10_rock.mat", "spe_perm.dat")
    checked: list[Path] = []
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate not in checked:
                checked.append(candidate)
                if candidate.exists():
                    return candidate
        for sub in ("SPE10", "spe10", "data", "input"):
            for name in names:
                candidate = root / sub / name
                if candidate not in checked:
                    checked.append(candidate)
                    if candidate.exists():
                        return candidate
    shown = "\n  ".join(str(p) for p in checked)
    raise FileNotFoundError(
        "Could not locate SPE10.zip, spe10_rock.mat, or spe_perm.dat. "
        f"Set SPE10_INPUT explicitly. Checked:\n  {shown}"
    )


def prepare_output(path: Path, force: bool) -> None:
    if force and path.exists():
        # Restrict deletion to the exact, resolved campaign directory.
        resolved = path.resolve()
        if resolved == Path("/") or len(resolved.parts) < 3:
            raise RuntimeError(f"Refusing unsafe output deletion: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)


def environment_record() -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


# =============================================================================
# INPUT PREFLIGHT AND FROZEN PROTOCOL
# =============================================================================


@dataclass
class LayerData:
    kx_md: np.ndarray       # shape (ny, nx)
    ky_md: np.ndarray       # shape (ny, nx)
    porosity: np.ndarray    # shape (ny, nx)
    audit: dict[str, Any]


def _parse_ascii_array(data: bytes, expected_count: int, label: str) -> np.ndarray:
    arr = np.fromstring(data.decode("ascii"), sep=" ", dtype=np.float64)
    if arr.size != expected_count:
        raise ValueError(f"{label}: expected {expected_count} values, found {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label}: non-finite values present")
    return arr


def _mat_rock_from_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    loaded = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    if "rock" not in loaded:
        raise ValueError("spe10_rock.mat does not contain a 'rock' struct")
    rock = loaded["rock"]
    perm_m2 = np.asarray(rock.perm, dtype=np.float64)
    porosity = np.asarray(rock.poro, dtype=np.float64).reshape(-1)
    expected_cells = NX_NATIVE * NY_NATIVE * NZ_NATIVE
    if perm_m2.shape != (expected_cells, 3):
        raise ValueError(f"Unexpected MATLAB permeability shape: {perm_m2.shape}")
    if porosity.size != expected_cells:
        raise ValueError(f"Unexpected MATLAB porosity count: {porosity.size}")
    return perm_m2 / MILLI_DARCY_TO_M2, porosity


def load_and_preflight_spe10(path: Path) -> LayerData:
    expected_cells = NX_NATIVE * NY_NATIVE * NZ_NATIVE
    source_hash = sha256_file(path)
    source_name = path.name
    member_hashes: dict[str, str] = {}
    crosschecks: dict[str, Any] = {}

    perm_flat_md: np.ndarray | None = None
    porosity_flat: np.ndarray | None = None
    mat_perm_md: np.ndarray | None = None
    mat_porosity: np.ndarray | None = None

    if path.suffix.lower() == ".zip":
        if source_name == "SPE10.zip" and source_hash != EXPECTED_HASHES["SPE10.zip"]:
            raise ValueError(
                "SPE10.zip SHA-256 differs from the supplied frozen archive. "
                f"Expected {EXPECTED_HASHES['SPE10.zip']}, got {source_hash}."
            )
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            required = {"spe_perm.dat", "spe_phi.dat"}
            if not required.issubset(names):
                raise ValueError(f"SPE10 archive lacks required members: {sorted(required - names)}")
            perm_bytes = archive.read("spe_perm.dat")
            phi_bytes = archive.read("spe_phi.dat")
            member_hashes["spe_perm.dat"] = sha256_bytes(perm_bytes)
            member_hashes["spe_phi.dat"] = sha256_bytes(phi_bytes)
            perm_flat_md = _parse_ascii_array(perm_bytes, 3 * expected_cells, "spe_perm.dat")
            porosity_flat = _parse_ascii_array(phi_bytes, expected_cells, "spe_phi.dat")
            if "spe10_rock.mat" in names:
                mat_bytes = archive.read("spe10_rock.mat")
                member_hashes["spe10_rock.mat"] = sha256_bytes(mat_bytes)
                mat_perm_md, mat_porosity = _mat_rock_from_bytes(mat_bytes)
    elif path.name == "spe_perm.dat" or path.suffix.lower() == ".dat":
        perm_bytes = path.read_bytes()
        member_hashes[path.name] = sha256_bytes(perm_bytes)
        perm_flat_md = _parse_ascii_array(perm_bytes, 3 * expected_cells, path.name)
        phi_path = path.with_name("spe_phi.dat")
        if not phi_path.exists():
            raise FileNotFoundError(f"Required sibling porosity file is missing: {phi_path}")
        phi_bytes = phi_path.read_bytes()
        member_hashes["spe_phi.dat"] = sha256_bytes(phi_bytes)
        porosity_flat = _parse_ascii_array(phi_bytes, expected_cells, "spe_phi.dat")
        mat_path = path.with_name("spe10_rock.mat")
        if mat_path.exists():
            mat_bytes = mat_path.read_bytes()
            member_hashes["spe10_rock.mat"] = sha256_bytes(mat_bytes)
            mat_perm_md, mat_porosity = _mat_rock_from_bytes(mat_bytes)
    elif path.suffix.lower() == ".mat":
        mat_bytes = path.read_bytes()
        member_hashes[path.name] = sha256_bytes(mat_bytes)
        mat_perm_md, mat_porosity = _mat_rock_from_bytes(mat_bytes)
    else:
        raise ValueError(f"Unsupported SPE10 input type: {path}")

    # Exact known-file identities are enforced whenever the member is present.
    for name, actual in member_hashes.items():
        if name in EXPECTED_HASHES and actual != EXPECTED_HASHES[name]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {EXPECTED_HASHES[name]}, got {actual}"
            )

    if perm_flat_md is not None:
        # Official raw ordering: three component blocks; x is fastest, then y,
        # then z.  MATLAB/Fortran reshape therefore resolves Layer 68 exactly.
        perm_rows_md = perm_flat_md.reshape(3, expected_cells).T
    elif mat_perm_md is not None:
        perm_rows_md = mat_perm_md
    else:
        raise RuntimeError("No permeability array was loaded")

    if porosity_flat is None:
        if mat_porosity is None:
            raise RuntimeError("No porosity array was loaded")
        porosity_flat = mat_porosity

    if mat_perm_md is not None and perm_flat_md is not None:
        # The MAT file stores SI m^2.  Compare in that stored unit so the
        # check is bit-exact; converting MAT -> mD introduces at most one
        # floating-point rounding bit even though the physical values agree.
        raw_perm_m2 = perm_rows_md * MILLI_DARCY_TO_M2
        mat_perm_m2 = mat_perm_md * MILLI_DARCY_TO_M2
        crosschecks["raw_md_converted_vs_mat_m2_exact"] = bool(
            np.array_equal(raw_perm_m2, mat_perm_m2)
        )
        crosschecks["raw_vs_mat_permeability_max_abs_md"] = float(np.max(np.abs(perm_rows_md - mat_perm_md)))
        if not crosschecks["raw_md_converted_vs_mat_m2_exact"]:
            raise ValueError("Raw and MATLAB permeability arrays do not agree exactly")
    if mat_porosity is not None and porosity_flat is not None:
        crosschecks["raw_vs_mat_porosity_exact"] = bool(np.array_equal(porosity_flat, mat_porosity))
        crosschecks["raw_vs_mat_porosity_max_abs"] = float(np.max(np.abs(porosity_flat - mat_porosity)))
        if not crosschecks["raw_vs_mat_porosity_exact"]:
            raise ValueError("Raw and MATLAB porosity arrays do not agree exactly")

    layer_index = LAYER_ONE_BASED - 1
    components = []
    for component in range(3):
        volume = perm_rows_md[:, component].reshape(
            (NX_NATIVE, NY_NATIVE, NZ_NATIVE), order="F"
        )
        # Internal numerical orientation is (y, x), origin lower in figures.
        components.append(np.asarray(volume[:, :, layer_index].T, dtype=np.float64))
    por_volume = porosity_flat.reshape((NX_NATIVE, NY_NATIVE, NZ_NATIVE), order="F")
    porosity = np.asarray(por_volume[:, :, layer_index].T, dtype=np.float64)
    kx_md, ky_md, kz_md = components

    if not np.all(np.isfinite(kx_md)) or not np.all(np.isfinite(ky_md)):
        raise ValueError("Layer 68 contains non-finite horizontal permeability")
    if np.any(kx_md <= 0.0) or np.any(ky_md <= 0.0):
        raise ValueError("Layer 68 contains non-positive Kx or Ky; frozen policy is hard failure")
    if not np.all(np.isfinite(porosity)):
        raise ValueError("Layer 68 contains non-finite porosity")

    audit = {
        "status": "PASS",
        "source_path": str(path),
        "source_name": source_name,
        "source_size_bytes": path.stat().st_size,
        "source_sha256": source_hash,
        "member_sha256": member_hashes,
        "crosschecks": crosschecks,
        "official_dimensions_xyz": [NX_NATIVE, NY_NATIVE, NZ_NATIVE],
        "raw_array_order": "component blocks; x fastest, then y, then z",
        "layer_one_based": LAYER_ONE_BASED,
        "layer_zero_based": layer_index,
        "internal_layer_shape_yx": list(kx_md.shape),
        "physical_orientation": {
            "x": "60 cells, 20 ft each, left to right",
            "y": "220 cells, 10 ft each, plotted bottom to top",
            "z": "one-based layer index",
        },
        "permeability_units_input": "mD",
        "used_components": ["Kx", "Ky"],
        "excluded_component": "Kz (out-of-plane in the two-dimensional layer solve)",
        "kx_md": summary_stats(kx_md),
        "ky_md": summary_stats(ky_md),
        "kz_md_audit_only": summary_stats(kz_md),
        "porosity": summary_stats(porosity),
        "zero_porosity_cells": int(np.count_nonzero(porosity == 0.0)),
        "inactive_cell_policy": (
            "No cell is made inactive by zero porosity: porosity does not enter the steady "
            "incompressible pressure equation. Non-finite or non-positive Kx/Ky is a hard error."
        ),
        "active_cells": int(kx_md.size),
    }
    return LayerData(kx_md=kx_md, ky_md=ky_md, porosity=porosity, audit=audit)


def summary_stats(a: np.ndarray) -> dict[str, float]:
    arr = np.asarray(a, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def build_protocol(input_audit: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "production":
        seeds = list(SEEDS)
        budgets = list(INTERIOR_BUDGETS)
        reference_factors = list(REFERENCE_FACTORS)
        candidate_multiplier = CANDIDATE_MULTIPLIER
    elif mode == "smoke":
        seeds = [SEEDS[0]]
        budgets = [80]
        reference_factors = [1]
        candidate_multiplier = 4
    else:
        raise ValueError("RUN_MODE must be 'production' or 'smoke'")

    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "selector_contract_version": SELECTOR_CONTRACT_VERSION,
        "run_mode": mode,
        "scientific_evidence": mode == "production",
        "input_identity": {
            "source_sha256": input_audit["source_sha256"],
            "member_sha256": input_audit["member_sha256"],
            "dimensions_xyz": input_audit["official_dimensions_xyz"],
            "array_order": input_audit["raw_array_order"],
            "layer_one_based": LAYER_ONE_BASED,
        },
        "problem": {
            "model": "SPE10 Model 2",
            "layer_one_based": LAYER_ONE_BASED,
            "domain_ft": [LX_FT, LY_FT],
            "cell_size_ft": [DX_FT, DY_FT],
            "equation": "-div(K grad p)=0; steady single-phase Darcy; constant mobility",
            "pressure_normalization": {"left": P_LEFT, "right": P_RIGHT},
            "top_bottom": "homogeneous no-flow",
            "wells_sources": "none",
            "pressure_gauge": "fixed by left/right Dirichlet data",
            "permeability": "K=diag(Kx,Ky), input mD; Kz excluded",
            "porosity": "audited but unused in steady pressure solve",
            "inactive_policy": input_audit["inactive_cell_policy"],
        },
        "frozen_selection": {
            "primary": PRIMARY,
            "challenger": CHALLENGER,
            "domain_decomposition": False,
            "retrospective_promotion_or_retuning": False,
            "primary_construction": "scrambled two-dimensional Halton",
            "challenger_construction": {
                "protected_halton_fraction": PROTECTED_HALTON_FRACTION,
                "enrichment": "coefficient-jump distance (Green/interface) indicator only",
                "solution_gradient_or_residual_component": False,
                "candidate_multiplier": candidate_multiplier,
                "selection": "fixed weighted sampling without replacement from scrambled Halton proposals",
                "indicator_positive_jump_quantile": INTERFACE_POSITIVE_JUMP_QUANTILE,
                "indicator_length_cells": INTERFACE_LENGTH_CELLS,
                "density_floor": INDICATOR_DENSITY_FLOOR,
            },
        },
        "replication": {
            "seeds": seeds,
            "interior_budgets": budgets,
            "boundary_nodes": "same deterministic budget-dependent rule for both allocations",
            "matched_conventional_baseline": CONVENTIONAL,
        },
        "numerics": {
            "reference": "harmonic conservative cell-centred finite volume",
            "reference_factors": reference_factors,
            "reference_linear_solver": "single-thread SuperLU sparse direct",
            "collocation_engine": "shared sparse P1 triangular Galerkin/collocation flux-balance operator",
            "collocation_linear_solver": "single-thread SuperLU sparse direct",
            "conventional_upscaling": "log-geometric mean in fixed contiguous native-cell blocks",
            "evaluation_cells": "all 60x220 native Layer-68 cell centres",
            "charged_time": "allocation/upscaling + assembly + factorization/solve + evaluation",
        },
        "reported_metrics": [
            "pressure RMSE/relative L2",
            "vector and component flux RMSE/relative L2",
            "pressure and flux errors on the frozen coefficient-jump mask",
            "total-flow error",
            "global mass balance",
            "interior algebraic conservation residual",
            "boundary error",
            "diagonally equilibrated one-norm condition proxy",
            "coverage, fill distance, separation",
            "charged wall time",
        ],
        "gates": {
            "mass_balance_tol": MASS_BALANCE_TOL,
            "algebraic_residual_tol": ALGEBRAIC_RESIDUAL_TOL,
            "boundary_value_tol": BOUNDARY_VALUE_TOL,
            "condition_proxy_limit": CONDITION_PROXY_LIMIT,
            "reference_pressure_change_must_decrease": REFERENCE_PRESSURE_DECREASE_REQUIRED,
            "accuracy_competitiveness": "reported as paired ratios; never used to retune or promote P",
        },
        "paper_boundary": {
            "authorized": "frozen P comparison and conventional reference only",
            "deferred": [
                "domain decomposition",
                "moving/crossing interfaces",
                "vug-matrix coupling",
                "residual or gradient retuning",
                "new transmission families",
            ],
        },
    }


# =============================================================================
# FROZEN INTERFACE INDICATOR AND NODE SETS
# =============================================================================


@dataclass
class InterfaceIndicator:
    logk: np.ndarray
    jump_score: np.ndarray
    mask: np.ndarray
    green: np.ndarray
    density: np.ndarray
    threshold: float
    length_ft: float


def make_interface_indicator(kx_md: np.ndarray, ky_md: np.ndarray) -> InterfaceIndicator:
    logk = 0.5 * (np.log10(kx_md) + np.log10(ky_md))
    score = np.zeros_like(logk)
    jump_x = np.abs(np.diff(logk, axis=1))
    jump_y = np.abs(np.diff(logk, axis=0))
    score[:, :-1] = np.maximum(score[:, :-1], jump_x)
    score[:, 1:] = np.maximum(score[:, 1:], jump_x)
    score[:-1, :] = np.maximum(score[:-1, :], jump_y)
    score[1:, :] = np.maximum(score[1:, :], jump_y)
    positive = score[score > 64.0 * np.finfo(float).eps]
    if positive.size == 0:
        raise ValueError("No positive coefficient jumps were found in SPE10 Layer 68")
    threshold = float(np.quantile(positive, INTERFACE_POSITIVE_JUMP_QUANTILE))
    mask = score >= threshold
    distance_ft = ndimage.distance_transform_edt(~mask, sampling=(DY_FT, DX_FT))
    length_ft = INTERFACE_LENGTH_CELLS * max(DX_FT, DY_FT)
    green = np.exp(-distance_ft / length_ft)
    span = float(np.max(green) - np.min(green))
    if span > 0.0:
        normalized = (green - np.min(green)) / span
    else:
        normalized = np.ones_like(green)
    density = INDICATOR_DENSITY_FLOOR + (1.0 - INDICATOR_DENSITY_FLOOR) * normalized
    return InterfaceIndicator(logk, score, mask, green, density, threshold, length_ft)


def interior_margin(n: int) -> float:
    return INTERIOR_MARGIN_COEFFICIENT / math.sqrt(float(n))


def halton_interior(n: int, seed: int) -> np.ndarray:
    margin = interior_margin(n)
    unit = qmc.Halton(d=2, scramble=True, seed=seed).random(n)
    unit = margin + (1.0 - 2.0 * margin) * unit
    return np.column_stack((unit[:, 0] * LX_FT, unit[:, 1] * LY_FT))


def indicator_values_at(points: np.ndarray, density_yx: np.ndarray) -> np.ndarray:
    ix = np.clip((points[:, 0] / DX_FT).astype(int), 0, NX_NATIVE - 1)
    iy = np.clip((points[:, 1] / DY_FT).astype(int), 0, NY_NATIVE - 1)
    return density_yx[iy, ix]


def protected_challenger_interior(
    n: int,
    seed: int,
    density_yx: np.ndarray,
    candidate_multiplier: int,
) -> tuple[np.ndarray, int]:
    n_protected = int(math.ceil(PROTECTED_HALTON_FRACTION * n))
    n_enrich = n - n_protected
    skeleton = halton_interior(n_protected, seed)
    if n_enrich == 0:
        return skeleton, n_protected

    n_candidates = max(candidate_multiplier * n_enrich, n_enrich)
    margin = interior_margin(n)
    proposals = qmc.Halton(d=3, scramble=True, seed=seed + 104729).random(n_candidates)
    xy_unit = margin + (1.0 - 2.0 * margin) * proposals[:, :2]
    candidates = np.column_stack((xy_unit[:, 0] * LX_FT, xy_unit[:, 1] * LY_FT))
    weights = indicator_values_at(candidates, density_yx)
    u = np.clip(proposals[:, 2], np.finfo(float).tiny, 1.0)
    # Fixed weighted selection without replacement (Efraimidis-Spirakis keys).
    keys = -np.log(u) / weights
    selected_idx = np.argpartition(keys, n_enrich - 1)[:n_enrich]
    selected_idx.sort()  # stable file ordering, not an additional selection step
    enriched = candidates[selected_idx]
    return np.vstack((skeleton, enriched)), n_protected


def boundary_nodes(n_interior: int) -> np.ndarray:
    nx_intervals = max(4, int(math.ceil(math.sqrt(n_interior * NX_NATIVE / NY_NATIVE))))
    ny_intervals = max(4, int(math.ceil(math.sqrt(n_interior * NY_NATIVE / NX_NATIVE))))
    bottom = np.column_stack((np.linspace(0.0, LX_FT, nx_intervals + 1), np.zeros(nx_intervals + 1)))
    top = np.column_stack((np.linspace(0.0, LX_FT, nx_intervals + 1), np.full(nx_intervals + 1, LY_FT)))
    y_inner = np.linspace(0.0, LY_FT, ny_intervals + 1)[1:-1]
    left = np.column_stack((np.zeros(y_inner.size), y_inner))
    right = np.column_stack((np.full(y_inner.size, LX_FT), y_inner))
    return np.vstack((bottom, top, left, right))


def coverage_metrics(interior: np.ndarray) -> dict[str, float]:
    unit = np.column_stack((interior[:, 0] / LX_FT, interior[:, 1] / LY_FT))
    bins = 20
    counts, _, _ = np.histogram2d(unit[:, 0], unit[:, 1], bins=(bins, bins), range=((0, 1), (0, 1)))
    empty_fraction = float(np.mean(counts == 0))
    occupancy_cv = float(np.std(counts) / max(np.mean(counts), np.finfo(float).tiny))
    tree = cKDTree(unit)
    distances, _ = tree.query(unit, k=2)
    min_separation = float(np.min(distances[:, 1]))
    x = (np.arange(NX_NATIVE) + 0.5) / NX_NATIVE
    y = (np.arange(NY_NATIVE) + 0.5) / NY_NATIVE
    xx, yy = np.meshgrid(x, y)
    evaluation = np.column_stack((xx.ravel(), yy.ravel()))
    fill, _ = tree.query(evaluation, k=1)
    return {
        "coverage_empty_fraction_20x20": empty_fraction,
        "coverage_occupancy_cv_20x20": occupancy_cv,
        "coverage_min_separation_unit": min_separation,
        "coverage_fill_max_unit": float(np.max(fill)),
        "coverage_fill_rms_unit": float(np.sqrt(np.mean(fill ** 2))),
    }


# =============================================================================
# CONSERVATIVE STRUCTURED FINITE-VOLUME ENGINE
# =============================================================================


@dataclass
class LinearSolve:
    solution: np.ndarray
    lu: Any
    factor_solve_seconds: float


def sparse_direct_solve(matrix: csr_matrix, rhs: np.ndarray) -> LinearSolve:
    start = time.perf_counter()
    lu = splu(csc_matrix(matrix), permc_spec="MMD_AT_PLUS_A")
    solution = lu.solve(np.asarray(rhs, dtype=np.float64))
    elapsed = time.perf_counter() - start
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("Sparse direct solve returned non-finite values")
    return LinearSolve(solution, lu, elapsed)


def assemble_fv(
    kx: np.ndarray,
    ky: np.ndarray,
    widths_x: np.ndarray,
    widths_y: np.ndarray,
) -> tuple[csr_matrix, np.ndarray, dict[str, np.ndarray]]:
    ny, nx = kx.shape
    if ky.shape != (ny, nx):
        raise ValueError("Kx/Ky shapes differ")
    if widths_x.size != nx or widths_y.size != ny:
        raise ValueError("Finite-volume width arrays do not match permeability")
    ids = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)

    left_ids = ids[:, :-1].ravel()
    right_ids = ids[:, 1:].ravel()
    resistance_x = (
        0.5 * widths_x[:-1][None, :] / kx[:, :-1]
        + 0.5 * widths_x[1:][None, :] / kx[:, 1:]
    )
    tx = (widths_y[:, None] / resistance_x).ravel()

    bottom_ids = ids[:-1, :].ravel()
    top_ids = ids[1:, :].ravel()
    resistance_y = (
        0.5 * widths_y[:-1][:, None] / ky[:-1, :]
        + 0.5 * widths_y[1:][:, None] / ky[1:, :]
    )
    ty = (widths_x[None, :] / resistance_y).ravel()

    rows = np.concatenate((left_ids, left_ids, right_ids, right_ids,
                           bottom_ids, bottom_ids, top_ids, top_ids))
    cols = np.concatenate((left_ids, right_ids, left_ids, right_ids,
                           bottom_ids, top_ids, bottom_ids, top_ids))
    vals = np.concatenate((tx, -tx, -tx, tx, ty, -ty, -ty, ty))

    t_left = widths_y * kx[:, 0] / (0.5 * widths_x[0])
    t_right = widths_y * kx[:, -1] / (0.5 * widths_x[-1])
    boundary_ids = np.concatenate((ids[:, 0], ids[:, -1]))
    rows = np.concatenate((rows, boundary_ids))
    cols = np.concatenate((cols, boundary_ids))
    vals = np.concatenate((vals, t_left, t_right))

    matrix = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    rhs = np.zeros(nx * ny, dtype=np.float64)
    rhs[ids[:, 0]] += t_left * P_LEFT
    rhs[ids[:, -1]] += t_right * P_RIGHT
    faces = {
        "tx": tx.reshape(ny, nx - 1),
        "ty": ty.reshape(ny - 1, nx),
        "t_left": t_left,
        "t_right": t_right,
    }
    return matrix, rhs, faces


def fv_flux_and_balance(
    pressure_yx: np.ndarray,
    kx_md: np.ndarray,
    ky_md: np.ndarray,
    widths_x: np.ndarray,
    widths_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ny, nx = pressure_yx.shape
    qx_faces = np.zeros((ny, nx + 1), dtype=np.float64)
    qy_faces = np.zeros((ny + 1, nx), dtype=np.float64)

    rx = (
        0.5 * widths_x[:-1][None, :] / kx_md[:, :-1]
        + 0.5 * widths_x[1:][None, :] / kx_md[:, 1:]
    )
    qx_faces[:, 1:nx] = (pressure_yx[:, :-1] - pressure_yx[:, 1:]) / rx
    qx_faces[:, 0] = (P_LEFT - pressure_yx[:, 0]) / (0.5 * widths_x[0] / kx_md[:, 0])
    qx_faces[:, nx] = (pressure_yx[:, -1] - P_RIGHT) / (0.5 * widths_x[-1] / kx_md[:, -1])

    ry = (
        0.5 * widths_y[:-1][:, None] / ky_md[:-1, :]
        + 0.5 * widths_y[1:][:, None] / ky_md[1:, :]
    )
    qy_faces[1:ny, :] = (pressure_yx[:-1, :] - pressure_yx[1:, :]) / ry
    # top and bottom remain exactly no-flow.

    qx_cell = 0.5 * (qx_faces[:, :-1] + qx_faces[:, 1:])
    qy_cell = 0.5 * (qy_faces[:-1, :] + qy_faces[1:, :])
    inflow = float(np.sum(qx_faces[:, 0] * widths_y))
    outflow = float(np.sum(qx_faces[:, -1] * widths_y))
    mass_balance = abs(inflow - outflow) / max(0.5 * (abs(inflow) + abs(outflow)), np.finfo(float).tiny)
    return qx_cell, qy_cell, 0.5 * (abs(inflow) + abs(outflow)), mass_balance


def restrict_to_native(a: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(a)
    return a.reshape(NY_NATIVE, factor, NX_NATIVE, factor).mean(axis=(1, 3))


def run_reference(
    layer: LayerData,
    factors: Sequence[int],
    output: Path,
    protocol_hash: str,
    resume: bool,
    log: RunLog,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    ref_dir = output / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    scale_md = float(np.exp(np.mean(np.log(np.sqrt(layer.kx_md * layer.ky_md)))))
    states: dict[int, dict[str, Any]] = {}

    for factor in factors:
        state_path = ref_dir / f"reference_factor_{factor}.npz"
        meta_path = ref_dir / f"reference_factor_{factor}.json"
        if resume and state_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("protocol_hash") != protocol_hash:
                raise RuntimeError(f"Reference checkpoint contract mismatch: {meta_path}")
            with np.load(state_path) as saved:
                states[factor] = {
                    "p": saved["pressure_native"],
                    "qx": saved["qx_native"],
                    "qy": saved["qy_native"],
                    "meta": meta,
                }
            log.write(f"Reference factor {factor}: resumed")
            continue

        log.write(f"Reference factor {factor}: assembling {NX_NATIVE*factor} x {NY_NATIVE*factor}")
        start_total = time.perf_counter()
        kx = np.repeat(np.repeat(layer.kx_md, factor, axis=0), factor, axis=1)
        ky = np.repeat(np.repeat(layer.ky_md, factor, axis=0), factor, axis=1)
        widths_x = np.full(NX_NATIVE * factor, DX_FT / factor)
        widths_y = np.full(NY_NATIVE * factor, DY_FT / factor)
        t0 = time.perf_counter()
        matrix, rhs, _ = assemble_fv(kx / scale_md, ky / scale_md, widths_x, widths_y)
        assembly_seconds = time.perf_counter() - t0
        solve = sparse_direct_solve(matrix, rhs)
        pressure = solve.solution.reshape(NY_NATIVE * factor, NX_NATIVE * factor)
        qx, qy, total_flow, mass_balance = fv_flux_and_balance(
            pressure, kx, ky, widths_x, widths_y
        )
        residual = matrix @ solve.solution - rhs
        algebraic = float(np.max(np.abs(residual)) / max(np.max(np.abs(rhs)), np.finfo(float).tiny))
        p_native = restrict_to_native(pressure, factor)
        qx_native = restrict_to_native(qx, factor)
        qy_native = restrict_to_native(qy, factor)
        total_seconds = time.perf_counter() - start_total
        meta = {
            "protocol_hash": protocol_hash,
            "factor": factor,
            "grid_nx": NX_NATIVE * factor,
            "grid_ny": NY_NATIVE * factor,
            "degrees_of_freedom": int(matrix.shape[0]),
            "assembly_seconds": assembly_seconds,
            "factor_solve_seconds": solve.factor_solve_seconds,
            "total_seconds": total_seconds,
            "global_mass_balance_relative": mass_balance,
            "algebraic_residual_relative_max": algebraic,
            "total_flow_md": total_flow,
            "permeability_scale_md": scale_md,
        }
        atomic_npz(
            state_path,
            pressure_native=p_native,
            qx_native=qx_native,
            qy_native=qy_native,
        )
        atomic_json(meta_path, json_ready(meta))
        states[factor] = {"p": p_native, "qx": qx_native, "qy": qy_native, "meta": meta}
        log.write(
            f"Reference factor {factor}: solved in {total_seconds:.2f} s; "
            f"mass balance={mass_balance:.3e}"
        )

    convergence: list[dict[str, Any]] = []
    sorted_factors = sorted(states)
    previous_pressure_change: float | None = None
    for low, high in zip(sorted_factors[:-1], sorted_factors[1:]):
        dp = states[high]["p"] - states[low]["p"]
        dqx = states[high]["qx"] - states[low]["qx"]
        dqy = states[high]["qy"] - states[low]["qy"]
        pressure_change = float(np.sqrt(np.mean(dp ** 2)))
        flux_change = float(np.sqrt(np.mean(dqx ** 2 + dqy ** 2)))
        row = {
            "factor_low": low,
            "factor_high": high,
            "pressure_rmse_change": pressure_change,
            "flux_vector_rmse_change": flux_change,
            "pressure_change_ratio_to_previous": (
                "" if previous_pressure_change is None
                else pressure_change / max(previous_pressure_change, np.finfo(float).tiny)
            ),
            "high_global_mass_balance_relative": states[high]["meta"]["global_mass_balance_relative"],
        }
        convergence.append(row)
        previous_pressure_change = pressure_change
    atomic_csv(output / "reference_convergence.csv", convergence)

    if len(convergence) >= 2 and REFERENCE_PRESSURE_DECREASE_REQUIRED:
        latest = float(convergence[-1]["pressure_rmse_change"])
        preceding = float(convergence[-2]["pressure_rmse_change"])
        if not latest < preceding:
            raise RuntimeError(
                "Frozen reference convergence gate failed: the latest pressure change "
                f"({latest:.6e}) did not decrease below the preceding change ({preceding:.6e})."
            )

    finest = states[max(sorted_factors)]
    reference = {
        "p": finest["p"],
        "qx": finest["qx"],
        "qy": finest["qy"],
        "total_flow_md": np.array(finest["meta"]["total_flow_md"]),
        "pressure_uncertainty": np.array(
            convergence[-1]["pressure_rmse_change"] if convergence else 0.0
        ),
        "flux_uncertainty": np.array(
            convergence[-1]["flux_vector_rmse_change"] if convergence else 0.0
        ),
    }
    return reference, convergence


# =============================================================================
# SHARED CONSERVATIVE UNSTRUCTURED COLLOCATION ENGINE
# =============================================================================


@dataclass
class TriangularSystem:
    triangulation: Delaunay
    matrix: csr_matrix
    triangle_gradients: np.ndarray
    triangle_kx_md: np.ndarray
    triangle_ky_md: np.ndarray
    left: np.ndarray
    right: np.ndarray
    free: np.ndarray
    assembly_seconds: float
    permeability_scale_md: float


def assemble_triangular_system(nodes: np.ndarray, layer: LayerData) -> TriangularSystem:
    start = time.perf_counter()
    tri = Delaunay(nodes)
    simplices = tri.simplices
    xyz = nodes[simplices]
    x1, y1 = xyz[:, 0, 0], xyz[:, 0, 1]
    x2, y2 = xyz[:, 1, 0], xyz[:, 1, 1]
    x3, y3 = xyz[:, 2, 0], xyz[:, 2, 1]
    twice_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(twice_area_signed)
    if np.any(area <= 1.0e-14 * LX_FT * LY_FT / max(nodes.shape[0], 1)):
        raise RuntimeError("Degenerate or near-degenerate triangle detected")
    denom = twice_area_signed[:, None]
    gradients = np.empty((simplices.shape[0], 3, 2), dtype=np.float64)
    gradients[:, :, 0] = np.column_stack((y2 - y3, y3 - y1, y1 - y2)) / denom
    gradients[:, :, 1] = np.column_stack((x3 - x2, x1 - x3, x2 - x1)) / denom

    centroids = np.mean(xyz, axis=1)
    ix = np.clip((centroids[:, 0] / DX_FT).astype(int), 0, NX_NATIVE - 1)
    iy = np.clip((centroids[:, 1] / DY_FT).astype(int), 0, NY_NATIVE - 1)
    kx_md = layer.kx_md[iy, ix]
    ky_md = layer.ky_md[iy, ix]
    scale_md = float(np.exp(np.mean(np.log(np.sqrt(layer.kx_md * layer.ky_md)))))
    kx = kx_md / scale_md
    ky = ky_md / scale_md

    local = area[:, None, None] * (
        kx[:, None, None] * gradients[:, :, 0, None] * gradients[:, None, :, 0]
        + ky[:, None, None] * gradients[:, :, 1, None] * gradients[:, None, :, 1]
    )
    rows = np.repeat(simplices, 3, axis=1).ravel()
    cols = np.tile(simplices, (1, 3)).ravel()
    matrix = coo_matrix((local.ravel(), (rows, cols)), shape=(nodes.shape[0], nodes.shape[0])).tocsr()

    boundary_eps = 128.0 * np.finfo(float).eps * max(LX_FT, LY_FT)
    left = np.abs(nodes[:, 0]) <= boundary_eps
    right = np.abs(nodes[:, 0] - LX_FT) <= boundary_eps
    fixed = left | right
    free = ~fixed
    if not np.any(left) or not np.any(right):
        raise RuntimeError("Triangular system lacks left or right Dirichlet nodes")
    return TriangularSystem(
        triangulation=tri,
        matrix=matrix,
        triangle_gradients=gradients,
        triangle_kx_md=kx_md,
        triangle_ky_md=ky_md,
        left=left,
        right=right,
        free=free,
        assembly_seconds=time.perf_counter() - start,
        permeability_scale_md=scale_md,
    )


def equilibrated_condition_proxy(a_free: csr_matrix, lu: Any) -> float:
    diagonal = np.asarray(a_free.diagonal(), dtype=np.float64)
    if np.any(diagonal <= 0.0):
        return float(CONDITION_PROXY_LIMIT * 10.0)
    inv_sqrt = 1.0 / np.sqrt(diagonal)
    scaled = diags(inv_sqrt) @ a_free @ diags(inv_sqrt)
    norm_a = float(onenormest(scaled))
    sqrt_diag = np.sqrt(diagonal)

    def inverse_matvec(x: np.ndarray) -> np.ndarray:
        vector = np.asarray(x, dtype=np.float64).reshape(-1)
        return sqrt_diag * lu.solve(sqrt_diag * vector)

    def inverse_matmat(x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        return sqrt_diag[:, None] * lu.solve(sqrt_diag[:, None] * matrix)

    inverse = LinearOperator(
        shape=a_free.shape,
        matvec=inverse_matvec,
        rmatvec=inverse_matvec,
        matmat=inverse_matmat,
        rmatmat=inverse_matmat,
        dtype=np.float64,
    )
    return norm_a * float(onenormest(inverse))


def evaluate_triangular_solution(
    system: TriangularSystem,
    nodes: np.ndarray,
    pressure: np.ndarray,
    evaluation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = system.triangulation
    simplex = tri.find_simplex(evaluation)
    if np.any(simplex < 0):
        raise RuntimeError(f"{np.count_nonzero(simplex < 0)} evaluation points lie outside triangulation")
    transforms = tri.transform[simplex]
    delta = evaluation - transforms[:, 2, :]
    bary12 = np.einsum("nij,nj->ni", transforms[:, :2, :], delta)
    bary = np.column_stack((bary12, 1.0 - np.sum(bary12, axis=1)))
    vertices = tri.simplices[simplex]
    p_eval = np.sum(bary * pressure[vertices], axis=1)

    triangle_pressure = pressure[tri.simplices]
    gradp = np.einsum("ti,tij->tj", triangle_pressure, system.triangle_gradients)
    qx_triangle = -system.triangle_kx_md * gradp[:, 0]
    qy_triangle = -system.triangle_ky_md * gradp[:, 1]
    return p_eval, qx_triangle[simplex], qy_triangle[simplex]


def run_collocation_case(
    method: str,
    seed: int,
    budget: int,
    layer: LayerData,
    indicator: InterfaceIndicator,
    reference: dict[str, np.ndarray],
    candidate_multiplier: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    charged_start = time.perf_counter()
    allocation_start = time.perf_counter()
    if method == PRIMARY:
        interior = halton_interior(budget, seed)
        n_protected = budget
    elif method == CHALLENGER:
        interior, n_protected = protected_challenger_interior(
            budget, seed, indicator.density, candidate_multiplier
        )
    else:
        raise ValueError(f"Unknown collocation method: {method}")
    boundary = boundary_nodes(budget)
    nodes = np.vstack((interior, boundary))
    allocation_seconds = time.perf_counter() - allocation_start
    coverage = coverage_metrics(interior)

    system = assemble_triangular_system(nodes, layer)
    pressure = np.zeros(nodes.shape[0], dtype=np.float64)
    pressure[system.left] = P_LEFT
    pressure[system.right] = P_RIGHT
    a_free = system.matrix[system.free][:, system.free].tocsr()
    rhs = -system.matrix[system.free][:, ~system.free] @ pressure[~system.free]
    solve = sparse_direct_solve(a_free, rhs)
    pressure[system.free] = solve.solution

    condition_start = time.perf_counter()
    condition_proxy = equilibrated_condition_proxy(a_free, solve.lu)
    condition_seconds = time.perf_counter() - condition_start

    x_eval = (np.arange(NX_NATIVE) + 0.5) * DX_FT
    y_eval = (np.arange(NY_NATIVE) + 0.5) * DY_FT
    xx, yy = np.meshgrid(x_eval, y_eval)
    evaluation = np.column_stack((xx.ravel(), yy.ravel()))
    evaluation_start = time.perf_counter()
    p_eval, qx_eval, qy_eval = evaluate_triangular_solution(
        system, nodes, pressure, evaluation
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    residual = system.matrix @ pressure
    left_reaction_scaled = float(np.sum(residual[system.left]))
    right_reaction_scaled = float(np.sum(residual[system.right]))
    total_flow_md = 0.5 * (abs(left_reaction_scaled) + abs(right_reaction_scaled)) * system.permeability_scale_md
    global_balance = abs(left_reaction_scaled + right_reaction_scaled) / max(
        0.5 * (abs(left_reaction_scaled) + abs(right_reaction_scaled)), np.finfo(float).tiny
    )
    algebraic = float(
        np.max(np.abs(residual[system.free]))
        / max(0.5 * (abs(left_reaction_scaled) + abs(right_reaction_scaled)), np.finfo(float).tiny)
    )
    boundary_error = float(
        max(
            np.max(np.abs(pressure[system.left] - P_LEFT)),
            np.max(np.abs(pressure[system.right] - P_RIGHT)),
        )
    )
    charged_seconds = time.perf_counter() - charged_start

    metrics = calculate_error_metrics(
        p_eval.reshape(NY_NATIVE, NX_NATIVE),
        qx_eval.reshape(NY_NATIVE, NX_NATIVE),
        qy_eval.reshape(NY_NATIVE, NX_NATIVE),
        reference,
        indicator.mask,
    )
    metrics.update(coverage)
    metrics.update({
        "method": method,
        "seed": seed,
        "interior_budget": budget,
        "n_interior": int(interior.shape[0]),
        "n_boundary": int(boundary.shape[0]),
        "degrees_of_freedom": int(nodes.shape[0]),
        "n_protected_halton": int(n_protected),
        "protected_fraction_realized": float(n_protected / budget),
        "triangles": int(system.triangulation.simplices.shape[0]),
        "global_mass_balance_relative": global_balance,
        "algebraic_residual_relative_max": algebraic,
        "boundary_value_error_max": boundary_error,
        "condition_proxy_equilibrated_1norm": condition_proxy,
        "total_flow_md": total_flow_md,
        "total_flow_relative_error": abs(total_flow_md - float(reference["total_flow_md"])) / max(
            abs(float(reference["total_flow_md"])), np.finfo(float).tiny
        ),
        "allocation_seconds": allocation_seconds,
        "assembly_seconds": system.assembly_seconds,
        "factor_solve_seconds": solve.factor_solve_seconds,
        "condition_estimate_seconds": condition_seconds,
        "evaluation_seconds": evaluation_seconds,
        "charged_total_seconds": charged_seconds,
    })
    metrics.update(make_gates(metrics, reference))
    state = {
        "nodes": nodes,
        "interior": interior,
        "pressure_nodes": pressure,
        "pressure_eval": p_eval.reshape(NY_NATIVE, NX_NATIVE),
        "qx_eval": qx_eval.reshape(NY_NATIVE, NX_NATIVE),
        "qy_eval": qy_eval.reshape(NY_NATIVE, NX_NATIVE),
    }
    return json_ready(metrics), state


# =============================================================================
# MATCHED CONVENTIONAL STRUCTURED BASELINE
# =============================================================================


def choose_structured_dimensions(target_dof: int) -> tuple[int, int]:
    native_ratio = NY_NATIVE / NX_NATIVE
    best: tuple[float, int, int] | None = None
    for nx in range(4, NX_NATIVE + 1):
        for ny in range(4, NY_NATIVE + 1):
            budget_mismatch = abs(nx * ny - target_dof) / target_dof
            shape_mismatch = abs(math.log((ny / nx) / native_ratio))
            score = budget_mismatch + 0.02 * shape_mismatch
            candidate = (score, nx, ny)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def contiguous_groups(n_native: int, n_coarse: int) -> list[np.ndarray]:
    return [np.asarray(group, dtype=int) for group in np.array_split(np.arange(n_native), n_coarse)]


def log_geometric_upscale(
    field_yx: np.ndarray,
    x_groups: list[np.ndarray],
    y_groups: list[np.ndarray],
) -> np.ndarray:
    coarse = np.empty((len(y_groups), len(x_groups)), dtype=np.float64)
    log_field = np.log(field_yx)
    for jy, yg in enumerate(y_groups):
        for ix, xg in enumerate(x_groups):
            coarse[jy, ix] = math.exp(float(np.mean(log_field[np.ix_(yg, xg)])))
    return coarse


def run_structured_baseline(
    target_dof: int,
    nominal_budget: int,
    layer: LayerData,
    indicator: InterfaceIndicator,
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    charged_start = time.perf_counter()
    upscale_start = time.perf_counter()
    nx, ny = choose_structured_dimensions(target_dof)
    x_groups = contiguous_groups(NX_NATIVE, nx)
    y_groups = contiguous_groups(NY_NATIVE, ny)
    kx = log_geometric_upscale(layer.kx_md, x_groups, y_groups)
    ky = log_geometric_upscale(layer.ky_md, x_groups, y_groups)
    widths_x = np.array([len(group) * DX_FT for group in x_groups], dtype=np.float64)
    widths_y = np.array([len(group) * DY_FT for group in y_groups], dtype=np.float64)
    upscale_seconds = time.perf_counter() - upscale_start

    scale_md = float(np.exp(np.mean(np.log(np.sqrt(layer.kx_md * layer.ky_md)))))
    assembly_start = time.perf_counter()
    matrix, rhs, _ = assemble_fv(kx / scale_md, ky / scale_md, widths_x, widths_y)
    assembly_seconds = time.perf_counter() - assembly_start
    solve = sparse_direct_solve(matrix, rhs)
    p_coarse = solve.solution.reshape(ny, nx)

    condition_start = time.perf_counter()
    condition_proxy = equilibrated_condition_proxy(matrix, solve.lu)
    condition_seconds = time.perf_counter() - condition_start
    qx_coarse, qy_coarse, total_flow_md, global_balance = fv_flux_and_balance(
        p_coarse, kx, ky, widths_x, widths_y
    )
    residual = matrix @ solve.solution - rhs
    algebraic = float(np.max(np.abs(residual)) / max(np.max(np.abs(rhs)), np.finfo(float).tiny))

    ix_map = np.empty(NX_NATIVE, dtype=int)
    iy_map = np.empty(NY_NATIVE, dtype=int)
    for i, group in enumerate(x_groups):
        ix_map[group] = i
    for j, group in enumerate(y_groups):
        iy_map[group] = j
    p_eval = p_coarse[iy_map[:, None], ix_map[None, :]]
    qx_eval = qx_coarse[iy_map[:, None], ix_map[None, :]]
    qy_eval = qy_coarse[iy_map[:, None], ix_map[None, :]]
    charged_seconds = time.perf_counter() - charged_start

    metrics = calculate_error_metrics(p_eval, qx_eval, qy_eval, reference, indicator.mask)
    metrics.update({
        "method": CONVENTIONAL,
        "seed": 0,
        "interior_budget": nominal_budget,
        "n_interior": int(nx * ny),
        "n_boundary": 0,
        "degrees_of_freedom": int(nx * ny),
        "n_protected_halton": 0,
        "protected_fraction_realized": 0.0,
        "structured_nx": nx,
        "structured_ny": ny,
        "dof_mismatch_relative": abs(nx * ny - target_dof) / target_dof,
        "global_mass_balance_relative": global_balance,
        "algebraic_residual_relative_max": algebraic,
        "boundary_value_error_max": 0.0,
        "condition_proxy_equilibrated_1norm": condition_proxy,
        "total_flow_md": total_flow_md,
        "total_flow_relative_error": abs(total_flow_md - float(reference["total_flow_md"])) / max(
            abs(float(reference["total_flow_md"])), np.finfo(float).tiny
        ),
        "allocation_seconds": upscale_seconds,
        "assembly_seconds": assembly_seconds,
        "factor_solve_seconds": solve.factor_solve_seconds,
        "condition_estimate_seconds": condition_seconds,
        "evaluation_seconds": 0.0,
        "charged_total_seconds": charged_seconds,
    })
    metrics.update(make_gates(metrics, reference))
    state = {
        "pressure_eval": p_eval,
        "qx_eval": qx_eval,
        "qy_eval": qy_eval,
        "pressure_coarse": p_coarse,
        "qx_coarse": qx_coarse,
        "qy_coarse": qy_coarse,
        "widths_x": widths_x,
        "widths_y": widths_y,
    }
    return json_ready(metrics), state


# =============================================================================
# METRICS, GATES, AGGREGATION, AND FIGURES
# =============================================================================


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def calculate_error_metrics(
    pressure: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    reference: dict[str, np.ndarray],
    high_contrast_mask: np.ndarray,
) -> dict[str, float]:
    p_ref = reference["p"]
    qx_ref = reference["qx"]
    qy_ref = reference["qy"]
    dp = pressure - p_ref
    dqx = qx - qx_ref
    dqy = qy - qy_ref
    vector_error = np.sqrt(dqx ** 2 + dqy ** 2)
    vector_reference = np.sqrt(qx_ref ** 2 + qy_ref ** 2)
    return {
        "pressure_rmse": rms(dp),
        "pressure_mae": float(np.mean(np.abs(dp))),
        "pressure_relative_l2": rms(dp) / max(rms(p_ref), np.finfo(float).tiny),
        "flux_vector_rmse": rms(vector_error),
        "flux_vector_relative_l2": rms(vector_error) / max(rms(vector_reference), np.finfo(float).tiny),
        "flux_x_rmse": rms(dqx),
        "flux_y_rmse": rms(dqy),
        "high_contrast_pressure_rmse": rms(dp[high_contrast_mask]),
        "high_contrast_flux_vector_rmse": rms(vector_error[high_contrast_mask]),
        "high_contrast_flux_relative_l2": rms(vector_error[high_contrast_mask]) / max(
            rms(vector_reference[high_contrast_mask]), np.finfo(float).tiny
        ),
    }


def make_gates(metrics: dict[str, Any], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    numeric_keys = (
        "pressure_rmse",
        "flux_vector_rmse",
        "global_mass_balance_relative",
        "algebraic_residual_relative_max",
        "condition_proxy_equilibrated_1norm",
    )
    finite_gate = all(math.isfinite(float(metrics[key])) for key in numeric_keys)
    conservation_gate = float(metrics["global_mass_balance_relative"]) <= MASS_BALANCE_TOL
    algebraic_gate = float(metrics["algebraic_residual_relative_max"]) <= ALGEBRAIC_RESIDUAL_TOL
    boundary_gate = float(metrics["boundary_value_error_max"]) <= BOUNDARY_VALUE_TOL
    condition_gate = float(metrics["condition_proxy_equilibrated_1norm"]) <= CONDITION_PROXY_LIMIT
    pressure_floor = float(reference["pressure_uncertainty"])
    flux_floor = float(reference["flux_uncertainty"])
    above_pressure_floor = float(metrics["pressure_rmse"]) >= pressure_floor
    above_flux_floor = float(metrics["flux_vector_rmse"]) >= flux_floor
    return {
        "gate_finite": finite_gate,
        "gate_global_conservation": conservation_gate,
        "gate_algebraic_conservation": algebraic_gate,
        "gate_boundary": boundary_gate,
        "gate_condition": condition_gate,
        "gate_error_above_reference_pressure_change": above_pressure_floor,
        "gate_error_above_reference_flux_change": above_flux_floor,
        "gate_admissible": bool(
            finite_gate and conservation_gate and algebraic_gate and boundary_gate and condition_gate
        ),
    }


def case_stem(method: str, budget: int, seed: int) -> str:
    safe = method.lower().replace("_", "-")
    return f"{safe}__N{budget:05d}__seed{seed:03d}"


def load_case_metrics(case_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(case_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def median(values: Iterable[float]) -> float:
    return float(np.median(np.asarray(list(values), dtype=np.float64)))


def aggregate_results(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    collocation = [r for r in rows if r["method"] in (PRIMARY, CHALLENGER)]
    lookup = {(r["method"], int(r["interior_budget"]), int(r["seed"])): r for r in collocation}
    paired: list[dict[str, Any]] = []
    for budget in sorted({int(r["interior_budget"]) for r in collocation}):
        seeds = sorted({int(r["seed"]) for r in collocation if int(r["interior_budget"]) == budget})
        for seed in seeds:
            h = lookup.get((PRIMARY, budget, seed))
            g = lookup.get((CHALLENGER, budget, seed))
            if h is None or g is None:
                continue
            paired.append({
                "interior_budget": budget,
                "seed": seed,
                "pressure_ratio_challenger_over_halton": g["pressure_rmse"] / h["pressure_rmse"],
                "flux_ratio_challenger_over_halton": g["flux_vector_rmse"] / h["flux_vector_rmse"],
                "high_contrast_pressure_ratio": g["high_contrast_pressure_rmse"] / h["high_contrast_pressure_rmse"],
                "high_contrast_flux_ratio": g["high_contrast_flux_vector_rmse"] / h["high_contrast_flux_vector_rmse"],
                "cost_ratio_challenger_over_halton": g["charged_total_seconds"] / h["charged_total_seconds"],
                "condition_ratio_challenger_over_halton": g["condition_proxy_equilibrated_1norm"] / h["condition_proxy_equilibrated_1norm"],
                "challenger_pressure_win": bool(g["pressure_rmse"] < h["pressure_rmse"]),
                "challenger_flux_win": bool(g["flux_vector_rmse"] < h["flux_vector_rmse"]),
                "challenger_high_contrast_pressure_win": bool(g["high_contrast_pressure_rmse"] < h["high_contrast_pressure_rmse"]),
                "challenger_high_contrast_flux_win": bool(g["high_contrast_flux_vector_rmse"] < h["high_contrast_flux_vector_rmse"]),
                "both_admissible": bool(h["gate_admissible"] and g["gate_admissible"]),
            })

    summary: list[dict[str, Any]] = []
    paper: list[dict[str, Any]] = []
    baselines = {int(r["interior_budget"]): r for r in rows if r["method"] == CONVENTIONAL}
    for budget in sorted({int(r["interior_budget"]) for r in collocation}):
        h_rows = [r for r in collocation if r["method"] == PRIMARY and int(r["interior_budget"]) == budget]
        g_rows = [r for r in collocation if r["method"] == CHALLENGER and int(r["interior_budget"]) == budget]
        p_rows = [r for r in paired if int(r["interior_budget"]) == budget]
        if not h_rows or not g_rows:
            continue
        base = baselines.get(budget)
        row = {
            "interior_budget": budget,
            "pairs": len(p_rows),
            "median_total_dof_halton": median(r["degrees_of_freedom"] for r in h_rows),
            "median_pressure_rmse_halton": median(r["pressure_rmse"] for r in h_rows),
            "median_pressure_rmse_challenger": median(r["pressure_rmse"] for r in g_rows),
            "median_pressure_ratio_challenger_over_halton": median(r["pressure_ratio_challenger_over_halton"] for r in p_rows),
            "challenger_pressure_wins": sum(bool(r["challenger_pressure_win"]) for r in p_rows),
            "median_flux_rmse_halton": median(r["flux_vector_rmse"] for r in h_rows),
            "median_flux_rmse_challenger": median(r["flux_vector_rmse"] for r in g_rows),
            "median_flux_ratio_challenger_over_halton": median(r["flux_ratio_challenger_over_halton"] for r in p_rows),
            "challenger_flux_wins": sum(bool(r["challenger_flux_win"]) for r in p_rows),
            "median_high_contrast_pressure_ratio": median(r["high_contrast_pressure_ratio"] for r in p_rows),
            "median_high_contrast_flux_ratio": median(r["high_contrast_flux_ratio"] for r in p_rows),
            "median_charged_seconds_halton": median(r["charged_total_seconds"] for r in h_rows),
            "median_charged_seconds_challenger": median(r["charged_total_seconds"] for r in g_rows),
            "admissible_halton": sum(bool(r["gate_admissible"]) for r in h_rows),
            "admissible_challenger": sum(bool(r["gate_admissible"]) for r in g_rows),
        }
        if base is not None:
            row.update({
                "conventional_dof": base["degrees_of_freedom"],
                "conventional_pressure_rmse": base["pressure_rmse"],
                "conventional_flux_rmse": base["flux_vector_rmse"],
                "conventional_charged_seconds": base["charged_total_seconds"],
                "conventional_admissible": base["gate_admissible"],
            })
        summary.append(json_ready(row))
        paper.append(json_ready(row.copy()))
    return paired, summary, paper


def gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "method", "interior_budget", "seed", "degrees_of_freedom",
        "gate_finite", "gate_global_conservation", "gate_algebraic_conservation",
        "gate_boundary", "gate_condition", "gate_error_above_reference_pressure_change",
        "gate_error_above_reference_flux_change", "gate_admissible",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def save_figures(
    output: Path,
    layer: LayerData,
    indicator: InterfaceIndicator,
    reference: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    case_dir: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    extent = (0.0, LX_FT, 0.0, LY_FT)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 12.0), constrained_layout=True)
    panels = [
        (np.log10(layer.kx_md), r"$\log_{10} K_x$ (mD)", "viridis"),
        (layer.porosity, "Porosity (audited; unused in pressure solve)", "magma"),
        (indicator.jump_score, "Frozen coefficient-jump score", "inferno"),
        (reference["p"], "Conservative finest-reference pressure", "coolwarm"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        image = ax.imshow(data, origin="lower", extent=extent, aspect="equal", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")
        fig.colorbar(image, ax=ax, shrink=0.82)
    atomic_figure(fig, output / "spe10_layer68_properties_and_reference.png")
    plt.close(fig)

    collocation_rows = [r for r in rows if r["method"] in (PRIMARY, CHALLENGER)]
    if collocation_rows:
        max_budget = max(int(r["interior_budget"]) for r in collocation_rows)
        example_seed = min(int(r["seed"]) for r in collocation_rows if int(r["interior_budget"]) == max_budget)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.0), constrained_layout=True)
        for ax, method, title in zip(
            axes,
            (PRIMARY, CHALLENGER),
            ("Frozen Halton primary", "Protected interface-GHG challenger"),
        ):
            stem = case_stem(method, max_budget, example_seed)
            with np.load(case_dir / f"{stem}.npz") as saved:
                interior = saved["interior"]
            ax.imshow(
                indicator.mask.astype(float), origin="lower", extent=extent,
                aspect="equal", cmap="Greys", alpha=0.28, interpolation="nearest"
            )
            ax.scatter(interior[:, 0], interior[:, 1], s=2.2, alpha=0.7, color="#1261a0")
            ax.set_title(f"{title}\nN={max_budget}, seed={example_seed}")
            ax.set_xlabel("x (ft)")
            ax.set_ylabel("y (ft)")
        atomic_figure(fig, output / "spe10_layer68_node_allocations.png")
        plt.close(fig)

    if summary:
        budgets = np.array([r["interior_budget"] for r in summary], dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
        plots = [
            ("median_pressure_rmse_halton", "median_pressure_rmse_challenger", "conventional_pressure_rmse", "Pressure RMSE"),
            ("median_flux_rmse_halton", "median_flux_rmse_challenger", "conventional_flux_rmse", "Flux-vector RMSE"),
            ("median_charged_seconds_halton", "median_charged_seconds_challenger", "conventional_charged_seconds", "Charged time (s)"),
        ]
        for ax, (hk, gk, bk, ylabel) in zip(axes.ravel()[:3], plots):
            ax.plot(budgets, [r[hk] for r in summary], "o-", label="Halton primary")
            ax.plot(budgets, [r[gk] for r in summary], "s-", label="Protected challenger")
            if all(bk in r for r in summary):
                ax.plot(budgets, [r[bk] for r in summary], "^-", label="Structured FV")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Interior-node budget")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
        ax = axes.ravel()[3]
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.plot(budgets, [r["median_pressure_ratio_challenger_over_halton"] for r in summary], "o-", label="Pressure")
        ax.plot(budgets, [r["median_flux_ratio_challenger_over_halton"] for r in summary], "s-", label="Flux")
        ax.plot(budgets, [r["median_high_contrast_pressure_ratio"] for r in summary], "^-", label="High-contrast pressure")
        ax.plot(budgets, [r["median_high_contrast_flux_ratio"] for r in summary], "d-", label="High-contrast flux")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Interior-node budget")
        ax.set_ylabel("Challenger / Halton paired median")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        atomic_figure(fig, output / "spe10_layer68_accuracy_cost_and_ratios.png")
        plt.close(fig)

    if paired:
        metrics = [
            ("pressure_ratio_challenger_over_halton", "Pressure"),
            ("flux_ratio_challenger_over_halton", "Flux"),
            ("high_contrast_pressure_ratio", "High-contrast pressure"),
            ("high_contrast_flux_ratio", "High-contrast flux"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), constrained_layout=True)
        for ax, (key, title) in zip(axes.ravel(), metrics):
            for seed in sorted({int(r["seed"]) for r in paired}):
                selected = sorted((r for r in paired if int(r["seed"]) == seed), key=lambda r: r["interior_budget"])
                ax.plot(
                    [r["interior_budget"] for r in selected],
                    [r[key] for r in selected], marker="o", lw=1.0, alpha=0.75,
                    label=f"seed {seed}",
                )
            ax.axhline(1.0, color="black", lw=1.0, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Interior-node budget")
            ax.set_ylabel("Challenger / Halton")
            ax.grid(True, which="both", alpha=0.25)
        axes.ravel()[0].legend(ncol=2)
        atomic_figure(fig, output / "spe10_layer68_seed_resolved_challenger_ratios.png")
        plt.close(fig)


def scientific_manifest(output: Path) -> list[dict[str, Any]]:
    excluded = {
        "scientific_manifest_sha256.csv",
        "run.log",
        "production_complete.json",
    }
    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.name in excluded or path.suffix == ".tmp":
            continue
        rows.append({
            "relative_path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


# =============================================================================
# CAMPAIGN DRIVER
# =============================================================================


def main() -> None:
    mode = RUN_MODE.strip().lower()
    input_path = resolve_input(SPE10_INPUT)
    output_name = OUTPUT_DIR if mode == "production" else OUTPUT_DIR + "_SMOKE_NOT_EVIDENCE"
    output = (script_directory() / output_name).resolve()
    prepare_output(output, FORCE_NEW_OUTPUT)
    log = RunLog(output / "run.log")
    log.write(f"Starting {IMPLEMENTATION_VERSION}; mode={mode}; input={input_path}")

    layer = load_and_preflight_spe10(input_path)
    atomic_json(output / "spe10_input_audit.json", json_ready(layer.audit))
    log.write(
        "SPE10 preflight PASS: raw order resolved; Layer 68 is 220 x 60; "
        f"zero-porosity cells retained={layer.audit['zero_porosity_cells']}"
    )

    protocol = build_protocol(layer.audit, mode)
    protocol_hash = sha256_bytes(canonical_json_bytes(protocol))
    protocol["protocol_sha256"] = protocol_hash
    protocol_path = output / "spe10_p_frozen_protocol.json"
    if protocol_path.exists() and RESUME:
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(
                "Existing output has a different frozen protocol. Use a new OUTPUT_DIR; "
                "do not mix scientific contracts."
            )
    atomic_json(protocol_path, json_ready(protocol))
    atomic_json(output / "environment.json", environment_record())
    if "__file__" in globals():
        source_copy = output / "source" / Path(__file__).name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), source_copy)
    log.write(f"Frozen protocol SHA-256: {protocol_hash}")

    indicator = make_interface_indicator(layer.kx_md, layer.ky_md)
    indicator_record = {
        "positive_jump_quantile": INTERFACE_POSITIVE_JUMP_QUANTILE,
        "threshold": indicator.threshold,
        "mask_cells": int(np.count_nonzero(indicator.mask)),
        "mask_fraction": float(np.mean(indicator.mask)),
        "length_ft": indicator.length_ft,
        "density_floor": INDICATOR_DENSITY_FLOOR,
        "jump_score": summary_stats(indicator.jump_score),
        "density": summary_stats(indicator.density),
    }
    atomic_json(output / "interface_indicator_audit.json", json_ready(indicator_record))
    atomic_npz(
        output / "interface_indicator_fields.npz",
        log10_geometric_horizontal_k=indicator.logk,
        jump_score=indicator.jump_score,
        high_contrast_mask=indicator.mask,
        green_indicator=indicator.green,
        enrichment_density=indicator.density,
    )

    reference, convergence = run_reference(
        layer,
        protocol["numerics"]["reference_factors"],
        output,
        protocol_hash,
        RESUME,
        log,
    )
    log.write("Reference stage complete")

    case_dir = output / "cases"
    state_dir = output / "states"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in protocol["replication"]["interior_budgets"]]
    seeds = [int(x) for x in protocol["replication"]["seeds"]]
    candidate_multiplier = int(
        protocol["frozen_selection"]["challenger_construction"]["candidate_multiplier"]
    )

    expected_collocation = 2 * len(budgets) * len(seeds)
    completed = 0
    for budget in budgets:
        for seed in seeds:
            for method in (PRIMARY, CHALLENGER):
                stem = case_stem(method, budget, seed)
                metric_path = case_dir / f"{stem}.json"
                state_path = state_dir / f"{stem}.npz"
                if RESUME and metric_path.exists() and state_path.exists():
                    saved = json.loads(metric_path.read_text(encoding="utf-8"))
                    if saved.get("protocol_hash") != protocol_hash:
                        raise RuntimeError(f"Case checkpoint contract mismatch: {metric_path}")
                    completed += 1
                    log.write(f"Case resumed: {method}, N={budget}, seed={seed}")
                    continue
                log.write(f"Case running: {method}, N={budget}, seed={seed}")
                metrics, state = run_collocation_case(
                    method, seed, budget, layer, indicator, reference, candidate_multiplier
                )
                metrics["protocol_hash"] = protocol_hash
                atomic_npz(state_path, **state)
                atomic_json(metric_path, json_ready(metrics))
                completed += 1
                log.write(
                    f"Case complete: {method}, N={budget}, seed={seed}; "
                    f"pRMSE={metrics['pressure_rmse']:.4e}; "
                    f"qRMSE={metrics['flux_vector_rmse']:.4e}; "
                    f"admissible={metrics['gate_admissible']}"
                )
                # A current aggregate CSV makes interrupted campaigns inspectable.
                atomic_csv(output / "campaign_metrics.csv", load_case_metrics(case_dir))

        # One deterministic conventional baseline per budget, matched to the
        # common total DOF of the collocation methods at that budget.
        method = CONVENTIONAL
        stem = case_stem(method, budget, 0)
        metric_path = case_dir / f"{stem}.json"
        state_path = state_dir / f"{stem}.npz"
        if RESUME and metric_path.exists() and state_path.exists():
            saved = json.loads(metric_path.read_text(encoding="utf-8"))
            if saved.get("protocol_hash") != protocol_hash:
                raise RuntimeError(f"Baseline checkpoint contract mismatch: {metric_path}")
            log.write(f"Conventional baseline resumed: N={budget}")
        else:
            target_dof = budget + boundary_nodes(budget).shape[0]
            log.write(f"Conventional baseline running: nominal N={budget}, target DOF={target_dof}")
            metrics, state = run_structured_baseline(
                target_dof, budget, layer, indicator, reference
            )
            metrics["protocol_hash"] = protocol_hash
            atomic_npz(state_path, **state)
            atomic_json(metric_path, json_ready(metrics))
            log.write(
                f"Conventional baseline complete: N={budget}; "
                f"DOF={metrics['degrees_of_freedom']}; pRMSE={metrics['pressure_rmse']:.4e}"
            )
        atomic_csv(output / "campaign_metrics.csv", load_case_metrics(case_dir))

    all_rows = load_case_metrics(case_dir)
    all_rows.sort(key=lambda r: (int(r["interior_budget"]), str(r["method"]), int(r["seed"])))
    atomic_csv(output / "campaign_metrics.csv", all_rows)
    paired, summary, paper = aggregate_results(all_rows)
    atomic_csv(output / "paired_comparison.csv", paired)
    atomic_csv(output / "budget_summary.csv", summary)
    atomic_csv(output / "paper_table_spe10.csv", paper)
    atomic_csv(output / "campaign_gate_matrix.csv", gate_rows(all_rows))

    if MAKE_FIGURES:
        log.write("Generating publication-readable figures")
        save_figures(output, layer, indicator, reference, all_rows, paired, summary, state_dir)

    expected_baselines = len(budgets)
    actual_collocation = sum(r["method"] in (PRIMARY, CHALLENGER) for r in all_rows)
    actual_baselines = sum(r["method"] == CONVENTIONAL for r in all_rows)
    status = "COMPLETE" if (
        actual_collocation == expected_collocation and actual_baselines == expected_baselines
    ) else "INCOMPLETE"

    next_action_text = (
        "CGF PAPER I — SPE10 P CAMPAIGN HANDOFF\n"
        "=====================================\n"
        f"Status: {status}\n"
        f"Run mode: {mode}\n"
        f"Protocol SHA-256: {protocol_hash}\n\n"
        "Do not tune or rerun a promoted challenger after inspecting these outcomes. "
        "Use campaign_metrics.csv, paired_comparison.csv, budget_summary.csv, "
        "paper_table_spe10.csv, campaign_gate_matrix.csv, reference_convergence.csv, "
        "and the generated figures to update the LaTeX manuscript whether the result "
        "is favorable, neutral, or negative. Domain decomposition and all new P-specific "
        "allocation/transmission development remain Paper II.\n"
    )
    (output / "next_action.txt").write_text(next_action_text, encoding="utf-8", newline="\n")

    manifest = scientific_manifest(output)
    atomic_csv(output / "scientific_manifest_sha256.csv", manifest)
    completion = {
        "status": status,
        "run_mode": mode,
        "scientific_evidence": mode == "production",
        "completed_utc": utc_now(),
        "protocol_hash": protocol_hash,
        "expected_collocation_cases": expected_collocation,
        "completed_collocation_cases": actual_collocation,
        "expected_conventional_baselines": expected_baselines,
        "completed_conventional_baselines": actual_baselines,
        "reference_factors": protocol["numerics"]["reference_factors"],
        "reference_convergence_rows": len(convergence),
        "manifest_entries": len(manifest),
        "all_admissible_cases": bool(all(bool(r["gate_admissible"]) for r in all_rows)),
    }
    atomic_json(output / "production_complete.json", json_ready(completion))
    log.write(
        f"Campaign {status}: collocation {actual_collocation}/{expected_collocation}; "
        f"baselines {actual_baselines}/{expected_baselines}; output={output}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL ERROR\n===========", file=sys.stderr)
        traceback.print_exc()
        raise
