#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGF smooth-homogeneous non-activation and matched-baseline campaign.

Scientific question
-------------------
Does the physical fingerprint retain a simple global collocation method when
the operator and solution are smooth and homogeneous, rather than activating
interface enrichment or domain decomposition?

The manufactured benchmark is

    -Delta u = f  in (0,1)^2,
           u = 0  on the boundary,
    u(x,y) = sin(pi*x) sin(pi*y),
    f(x,y) = 2*pi^2 sin(pi*x) sin(pi*y).

The exact state is used only for the declared Dirichlet trace and final error
evaluation.  The physical fingerprint and both adaptive challengers use a
numerical finite-difference pilot and declared PDE data, never exact interior
values.

Four equal-budget allocations use the same global PHS r^3 RBF-FD engine,
quadratic polynomial augmentation, row equilibration, boundary accounting,
evaluation grid, and admissibility gates:

  1. HALTON: scrambled two-dimensional Halton reference.
  2. EQUAL_AREA_STRATIFIED: one seeded point in each equal-area cell.
  3. RESIDUAL_ADAPTIVE: protected Halton skeleton plus numerical-pilot
     residual enrichment.
  4. GHG_PROTECTED_CHALLENGER: protected Halton skeleton plus the existing
     nominal Green/Halton/Gradient mixture.  Because K=1 has no material
     interface, its Green/interface indicator is spatially constant and no
     interface collar or subdomain is created.

The selector is evaluated before any allocation outcome.  Its frozen E branch
classifies a problem as smooth/homogeneous when the coefficient contrast,
coefficient-jump area, and pilot high-frequency fraction pass the prospective
thresholds.  That branch selects EQUAL_AREA_STRATIFIED and explicitly records

    domain_decomposition_activated = False
    interface_enrichment_activated = False.

The selected equal-area method must then be admissible and competitive with the best eligible matched
baseline under frozen RMSE, H1-seminorm, and measured-cost ratios.  A failure
is a scientific result; the script does not retune the selector after E is
inspected.

Spyder use
----------
Leave PROFILE = "production" and press Run.  The five seeds and four budgets
resume automatically.  For a quick implementation check, use PROFILE =
"smoke"; smoke outputs are labelled non-scientific and never satisfy R3/A1.

Paper-I boundary
----------------
This code addresses R3/A1 and the E part of M1.  It contains no transmission
condition, overlap, coarse space, moving front, coefficient interface, or
domain-decomposition loop.  E can validate only the non-activation branch;
the final activation thresholds are frozen later from E/S/F before SPE10 P.
"""

from __future__ import annotations

# Single-thread defaults must be set before NumPy/SciPy imports.
import os
import tempfile

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "cgf_matplotlib")
)

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import qmc


# =============================================================================
# SPYDER-FACING CONFIGURATION
# =============================================================================

PROFILE = "production"          # "production" or non-scientific "smoke"
CAMPAIGN_DIR = ""               # blank -> profile-specific default directory
FORCE = False                    # True recomputes completed seed/budget rows


@dataclass
class Config:
    # Campaign and storage.
    profile: str = PROFILE
    campaign_dir: str = CAMPAIGN_DIR
    force: bool = FORCE
    save_full_arrays: bool = True

    # Numerical pilot and independent evaluation.
    pilot_nx: int = 33
    pilot_ny: int = 33
    residual_grid_n: int = 65
    independent_residual_grid_n: int = 21
    eval_nx: int = 101
    eval_ny: int = 101

    # Matched node constructions.
    scramble_halton: bool = True
    candidate_multiplier: int = 16
    residual_skeleton_fraction: float = 0.60
    ghg_coverage_floor_fraction: float = 0.45
    alpha_green: float = 1.0 / 3.0
    beta_halton: float = 1.0 / 3.0
    gamma_gradient: float = 1.0 / 3.0
    density_floor: float = 0.05
    equal_area_jitter_fraction: float = 0.80

    # Same local approximation engine as the S experiment.
    phs_power: int = 3
    poly_degree: int = 2
    stencil_size: int = 25
    boundary_factor: float = 1.55
    min_boundary_each_edge: int = 16

    # Prospective physical-fingerprint thresholds for the E branch.
    homogeneous_contrast_max: float = 1.05
    coefficient_jump_area_max: float = 0.01
    pilot_high_frequency_fraction_max: float = 0.05
    selected_global_family: str = "EQUAL_AREA_STRATIFIED"

    # Frozen admissibility and competitiveness gates.
    condition_proxy_max: float = 2.0e6
    range_violation_max: float = 2.0e-2
    conservation_relative_defect_max: float = 5.0e-2
    independent_residual_relative_max: float = 3.5e-1
    empty_cell_fraction_max: float = 0.85
    separation_min: float = 1.0e-7
    competitive_rmse_ratio_max: float = 1.25
    competitive_h1_ratio_max: float = 1.25
    competitive_cost_ratio_max: float = 1.50

    def validate(self) -> None:
        if self.profile not in ("production", "smoke"):
            raise ValueError("profile must be 'production' or 'smoke'.")
        if self.phs_power != 3 or self.poly_degree != 2:
            raise ValueError("This matched engine is frozen at PHS r^3 / degree 2.")
        if self.stencil_size < 12:
            raise ValueError("stencil_size is too small for quadratic augmentation.")
        if self.pilot_nx < 9 or self.pilot_ny < 9:
            raise ValueError("pilot grid is too small.")
        if self.residual_grid_n <= max(self.pilot_nx, self.pilot_ny):
            raise ValueError("independent residual grid must refine the pilot grid.")
        if not (0 < self.residual_skeleton_fraction < 1):
            raise ValueError("residual_skeleton_fraction must lie in (0,1).")
        if not (0 < self.ghg_coverage_floor_fraction < 1):
            raise ValueError("ghg_coverage_floor_fraction must lie in (0,1).")
        if not np.isclose(
            self.alpha_green + self.beta_halton + self.gamma_gradient, 1.0
        ):
            raise ValueError("alpha_green + beta_halton + gamma_gradient must equal 1.")
        if self.selected_global_family not in ALLOCATION_FAMILIES:
            raise ValueError("selected_global_family is not a declared allocation.")


ALLOCATION_FAMILIES = [
    "HALTON",
    "EQUAL_AREA_STRATIFIED",
    "RESIDUAL_ADAPTIVE",
    "GHG_PROTECTED_CHALLENGER",
]


def profile_seeds_budgets(profile: str) -> Tuple[List[int], List[int]]:
    if profile == "smoke":
        return [11], [60]
    if profile == "production":
        return [11, 23, 37, 53, 71], [120, 180, 260, 360]
    raise ValueError(profile)


# =============================================================================
# Utilities and frozen provenance
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = 14) -> str:
    return canonical_hash(obj)[:length]


def json_dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def default_campaign_dir(profile: str) -> str:
    return (
        "smooth_homogeneous_smoke"
        if profile == "smoke"
        else "smooth_homogeneous_nonactivation_campaign"
    )


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"CGF_E_{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def environment_record() -> Dict[str, Any]:
    import matplotlib
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "single_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
            )
        },
    }


def frozen_experiment_contract(cfg: Config) -> Dict[str, Any]:
    seeds, budgets = profile_seeds_budgets("production")
    contract: Dict[str, Any] = {
        "contract_version": "CGF-E-1.0",
        "paper_I_scope": "smooth homogeneous non-activation control E",
        "benchmark": {
            "domain": "(0,1)^2",
            "operator": "-Delta",
            "coefficient": 1.0,
            "manufactured_state": "sin(pi*x)*sin(pi*y)",
            "dirichlet_trace": 0.0,
        },
        "seeds": seeds,
        "budgets": budgets,
        "allocation_families": ALLOCATION_FAMILIES,
        "node_construction": {
            "scramble_halton": cfg.scramble_halton,
            "candidate_multiplier": cfg.candidate_multiplier,
            "residual_skeleton_fraction": cfg.residual_skeleton_fraction,
            "ghg_coverage_floor_fraction": cfg.ghg_coverage_floor_fraction,
            "nominal_ghg_weights": [
                cfg.alpha_green, cfg.beta_halton, cfg.gamma_gradient
            ],
            "density_floor": cfg.density_floor,
            "equal_area_jitter_fraction": cfg.equal_area_jitter_fraction,
        },
        "discretization": {
            "pilot_grid": [cfg.pilot_nx, cfg.pilot_ny],
            "residual_grid_n": cfg.residual_grid_n,
            "independent_residual_grid_n": cfg.independent_residual_grid_n,
            "evaluation_grid": [cfg.eval_nx, cfg.eval_ny],
            "phs_power": cfg.phs_power,
            "poly_degree": cfg.poly_degree,
            "stencil_size": cfg.stencil_size,
            "boundary_factor": cfg.boundary_factor,
            "min_boundary_each_edge": cfg.min_boundary_each_edge,
            "row_equilibration": True,
        },
        "fingerprint_thresholds": {
            "homogeneous_contrast_max": cfg.homogeneous_contrast_max,
            "coefficient_jump_area_max": cfg.coefficient_jump_area_max,
            "pilot_high_frequency_fraction_max": (
                cfg.pilot_high_frequency_fraction_max
            ),
            "selected_global_family": cfg.selected_global_family,
            "domain_decomposition_activated": False,
            "interface_enrichment_activated": False,
        },
        "gates": {
            "condition_proxy_max": cfg.condition_proxy_max,
            "range_violation_max": cfg.range_violation_max,
            "conservation_relative_defect_max": (
                cfg.conservation_relative_defect_max
            ),
            "independent_residual_relative_max": (
                cfg.independent_residual_relative_max
            ),
            "empty_cell_fraction_max": cfg.empty_cell_fraction_max,
            "separation_min": cfg.separation_min,
            "competitive_rmse_ratio_max": cfg.competitive_rmse_ratio_max,
            "competitive_h1_ratio_max": cfg.competitive_h1_ratio_max,
            "competitive_cost_ratio_max": cfg.competitive_cost_ratio_max,
        },
        "publication_rule": (
            "E passes R3 only if the pre-solve fingerprint selects a simple "
            "global family, explicitly declines DDM/interface enrichment, and "
            "that selected family is admissible and competitive on every "
            "declared seed-budget row. A failure is reported without retuning."
        ),
    }
    contract["frozen_experiment_sha256"] = canonical_hash(contract)
    return contract


def write_manifest(directory: Path, filename: str) -> None:
    rows = []
    for item in sorted(p for p in directory.iterdir() if p.is_file()):
        if item.name == filename:
            continue
        rows.append({
            "file": item.name,
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    pd.DataFrame(rows).to_csv(directory / filename, index=False)


def trapz1(values: np.ndarray, coordinates: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def normalize01(values: np.ndarray, floor: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("indicator contains non-finite values.")
    if hi - lo < 1.0e-15:
        scaled = np.ones_like(values)
    else:
        scaled = (values - lo) / (hi - lo)
    if floor > 0:
        scaled = floor + (1.0 - floor) * scaled
    return scaled


# =============================================================================
# Smooth homogeneous benchmark and deployable numerical pilot
# =============================================================================

def exact_u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * np.asarray(x)) * np.sin(np.pi * np.asarray(y))


def exact_ux(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.pi * np.cos(np.pi * np.asarray(x)) * np.sin(np.pi * np.asarray(y))


def exact_uy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.pi * np.sin(np.pi * np.asarray(x)) * np.cos(np.pi * np.asarray(y))


def source_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi**2 * exact_u(x, y)


def build_structured_pilot(cfg: Config) -> Dict[str, Any]:
    """Solve the PDE on a conservative five-point grid; no exact interior use."""
    t0 = time.perf_counter()
    nx, ny = cfg.pilot_nx, cfg.pilot_ny
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    def idx(i: int, j: int) -> int:
        return i * ny + j

    matrix = lil_matrix((nx * ny, nx * ny), dtype=float)
    rhs = np.zeros(nx * ny, dtype=float)
    for i in range(nx):
        for j in range(ny):
            row = idx(i, j)
            if i in (0, nx - 1) or j in (0, ny - 1):
                matrix[row, row] = 1.0
                rhs[row] = 0.0
                continue
            matrix[row, row] = 2.0 / dx**2 + 2.0 / dy**2
            matrix[row, idx(i - 1, j)] = -1.0 / dx**2
            matrix[row, idx(i + 1, j)] = -1.0 / dx**2
            matrix[row, idx(i, j - 1)] = -1.0 / dy**2
            matrix[row, idx(i, j + 1)] = -1.0 / dy**2
            rhs[row] = float(source_f(x[i], y[j]))
    u = spsolve(csr_matrix(matrix), rhs).reshape(nx, ny)
    ux = np.gradient(u, dx, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    grad = gaussian_filter(np.hypot(ux, uy), sigma=0.8)

    # Independent residual indicator: interpolate the pilot to a finer tensor
    # grid and apply a finite-difference Laplacian there.
    xr = np.linspace(0.0, 1.0, cfg.residual_grid_n)
    yr = np.linspace(0.0, 1.0, cfg.residual_grid_n)
    XR, YR = np.meshgrid(xr, yr, indexing="ij")
    points_r = np.column_stack([XR.ravel(), YR.ravel()])
    interp = RegularGridInterpolator((x, y), u, bounds_error=True)
    ur = interp(points_r).reshape(XR.shape)
    dr = xr[1] - xr[0]
    d2x = np.gradient(np.gradient(ur, dr, axis=0, edge_order=2), dr, axis=0, edge_order=2)
    d2y = np.gradient(np.gradient(ur, dr, axis=1, edge_order=2), dr, axis=1, edge_order=2)
    residual = np.abs(-(d2x + d2y) - source_f(XR, YR))
    residual[[0, 1, -2, -1], :] = 0.0
    residual[:, [0, 1, -2, -1]] = 0.0
    residual = gaussian_filter(residual, sigma=1.0)

    # A coefficient-jump indicator is identically zero because K=1.  The GHG
    # challenger interprets its Green/interface field as spatially uniform.
    coefficient = np.ones_like(X)
    coefficient_jump = np.zeros_like(X)
    green_uniform = np.ones_like(X)

    # Spectral diagnostic from the numerical pilot only.
    interior = u[1:-1, 1:-1]
    centered = interior - np.mean(interior)
    spectrum = np.abs(np.fft.rfft2(centered)) ** 2
    kx = np.fft.fftfreq(centered.shape[0])[:, None]
    ky = np.fft.rfftfreq(centered.shape[1])[None, :]
    radial = np.sqrt(kx**2 + ky**2)
    high = radial >= 0.35 * float(np.max(radial))
    high_frequency_fraction = float(
        spectrum[high].sum() / max(float(spectrum.sum()), 1.0e-300)
    )

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "u": u,
        "ux": ux,
        "uy": uy,
        "gradient": normalize01(grad, cfg.density_floor),
        "coefficient": coefficient,
        "coefficient_jump": coefficient_jump,
        "green": green_uniform,
        "residual_x": xr,
        "residual_y": yr,
        "residual": normalize01(residual, cfg.density_floor),
        "pilot_high_frequency_fraction": high_frequency_fraction,
        "pilot_time_s": time.perf_counter() - t0,
    }


def physical_fingerprint(pilot: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    coefficient = np.asarray(pilot["coefficient"], float)
    jump = np.asarray(pilot["coefficient_jump"], float)
    contrast = float(coefficient.max() / coefficient.min())
    jump_area = float(np.mean(jump > 0.0))
    high_frequency = float(pilot["pilot_high_frequency_fraction"])
    smooth_homogeneous = bool(
        contrast <= cfg.homogeneous_contrast_max
        and jump_area <= cfg.coefficient_jump_area_max
        and high_frequency <= cfg.pilot_high_frequency_fraction_max
    )
    selected = cfg.selected_global_family if smooth_homogeneous else "ESCALATE_AFTER_E"
    return {
        "coefficient_contrast_ratio": contrast,
        "coefficient_jump_area_fraction": jump_area,
        "pilot_high_frequency_energy_fraction": high_frequency,
        "homogeneous_threshold": cfg.homogeneous_contrast_max,
        "jump_area_threshold": cfg.coefficient_jump_area_max,
        "high_frequency_threshold": cfg.pilot_high_frequency_fraction_max,
        "classified_smooth_homogeneous": smooth_homogeneous,
        "problem_class": "SMOOTH_HOMOGENEOUS" if smooth_homogeneous else "ESCALATE",
        "selected_architecture": selected,
        "domain_decomposition_activated": False,
        "interface_enrichment_activated": False,
        "selection_timing": "before allocation outcomes",
        "pilot_time_s": float(pilot["pilot_time_s"]),
    }


def fingerprint_for_hash(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    """Remove measured runtime from the canonical scientific fingerprint."""
    return {
        key: value
        for key, value in fingerprint.items()
        if key != "pilot_time_s"
    }


# =============================================================================
# Four matched global allocation families
# =============================================================================

def halton_unit(n: int, seed: int, scramble: bool, skip: int = 1) -> np.ndarray:
    sampler = qmc.Halton(d=2, scramble=scramble, seed=seed if scramble else None)
    points = sampler.random(n + skip)[skip:]
    return np.clip(points, 1.0e-10, 1.0 - 1.0e-10)


def factor_grid(n: int) -> Tuple[int, int]:
    """Nearest-to-square exact factorization; all declared budgets factor."""
    left = int(math.floor(math.sqrt(n)))
    while left > 1 and n % left != 0:
        left -= 1
    if n % left != 0:
        return 1, n
    return left, n // left


def equal_area_centers(n: int) -> np.ndarray:
    nx, ny = factor_grid(n)
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def equal_area_stratified(n: int, seed: int, jitter_fraction: float) -> np.ndarray:
    nx, ny = factor_grid(n)
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(nx):
        for j in range(ny):
            jx, jy = rng.uniform(-0.5, 0.5, size=2) * jitter_fraction
            cells.append(((i + 0.5 + jx) / nx, (j + 0.5 + jy) / ny))
    return np.asarray(cells, dtype=float)


def systematic_weighted_pick(
    candidates: np.ndarray, weights: np.ndarray, n: int
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    candidates = np.asarray(candidates, float)
    weights = np.maximum(np.asarray(weights, float).ravel(), 0.0)
    if len(candidates) != len(weights):
        raise ValueError("candidate and weight arrays have different lengths.")
    if float(weights.sum()) <= 0.0:
        return candidates[np.linspace(0, len(candidates) - 1, n, dtype=int)]
    cumulative = np.cumsum(weights / weights.sum())
    target = (np.arange(n) + 0.5) / n
    proposed = np.searchsorted(cumulative, target, side="left")
    proposed = np.clip(proposed, 0, len(candidates) - 1)
    chosen: List[int] = []
    seen = set()
    for value in proposed:
        ivalue = int(value)
        if ivalue not in seen:
            chosen.append(ivalue)
            seen.add(ivalue)
    if len(chosen) < n:
        for value in np.argsort(-weights):
            ivalue = int(value)
            if ivalue not in seen:
                chosen.append(ivalue)
                seen.add(ivalue)
                if len(chosen) == n:
                    break
    return candidates[np.asarray(chosen[:n], dtype=int)]


def unique_fill(points: np.ndarray, n: int, fill_pool: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float)
    _, keep = np.unique(np.round(points, 12), axis=0, return_index=True)
    output = points[np.sort(keep)]
    if len(output) >= n:
        return output[:n]
    existing = {tuple(row) for row in np.round(output, 12)}
    additions = []
    for point in fill_pool:
        key = tuple(np.round(point, 12))
        if key not in existing:
            additions.append(point)
            existing.add(key)
            if len(output) + len(additions) == n:
                break
    if len(output) + len(additions) < n:
        raise RuntimeError("could not replenish allocation after de-duplication.")
    return np.vstack([output, np.asarray(additions)])[:n]


def grid_interpolator(
    x: np.ndarray, y: np.ndarray, values: np.ndarray
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (x, y), values, bounds_error=False, fill_value=None
    )


def build_interior_nodes(
    n: int,
    family: str,
    pilot: Dict[str, Any],
    seed: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if family == "HALTON":
        points = halton_unit(n, seed + 101, cfg.scramble_halton, skip=7)
        return points, np.array(["Halton"] * n, dtype=object), {"halton": n}

    if family == "EQUAL_AREA_STRATIFIED":
        points = equal_area_stratified(n, seed + 211, cfg.equal_area_jitter_fraction)
        return points, np.array(["EqualArea"] * n, dtype=object), {"equal_area": n}

    fill_pool = halton_unit(8 * max(n, 8), seed + 9001, cfg.scramble_halton, skip=97)

    if family == "RESIDUAL_ADAPTIVE":
        skeleton_n = int(round(cfg.residual_skeleton_fraction * n))
        adaptive_n = n - skeleton_n
        skeleton = halton_unit(skeleton_n, seed + 307, cfg.scramble_halton, skip=13)
        candidates = halton_unit(
            cfg.candidate_multiplier * max(adaptive_n, 8),
            seed + 401,
            cfg.scramble_halton,
            skip=31,
        )
        residual_interp = grid_interpolator(
            pilot["residual_x"], pilot["residual_y"], pilot["residual"]
        )
        adaptive = systematic_weighted_pick(
            candidates, residual_interp(candidates), adaptive_n
        )
        points = unique_fill(np.vstack([skeleton, adaptive]), n, fill_pool)
        labels = np.array(
            ["ResidualSkeleton"] * skeleton_n + ["PilotResidual"] * adaptive_n,
            dtype=object,
        )
        if len(labels) < n:
            labels = np.append(labels, ["CoverageFill"] * (n - len(labels)))
        return points, labels[:n], {
            "coverage_skeleton": skeleton_n,
            "pilot_residual": adaptive_n,
        }

    if family != "GHG_PROTECTED_CHALLENGER":
        raise ValueError(f"unknown allocation family: {family}")

    floor_n = int(round(cfg.ghg_coverage_floor_fraction * n))
    remainder = n - floor_n
    weights = np.array(
        [cfg.alpha_green, cfg.beta_halton, cfg.gamma_gradient], dtype=float
    )
    raw_counts = remainder * weights / weights.sum()
    component_counts = np.floor(raw_counts).astype(int)
    while int(component_counts.sum()) < remainder:
        component_counts[int(np.argmax(raw_counts - component_counts))] += 1
    green_n, halton_n, gradient_n = [int(value) for value in component_counts]

    skeleton = halton_unit(floor_n, seed + 503, cfg.scramble_halton, skip=17)
    pieces = [skeleton]
    labels: List[str] = ["CoverageFloor"] * floor_n

    green_candidates = halton_unit(
        cfg.candidate_multiplier * max(green_n, 8),
        seed + 601,
        cfg.scramble_halton,
        skip=41,
    )
    green_interp = grid_interpolator(pilot["x"], pilot["y"], pilot["green"])
    green_points = systematic_weighted_pick(
        green_candidates, green_interp(green_candidates), green_n
    )
    pieces.append(green_points)
    labels.extend(["GreenUniform"] * green_n)

    remainder_halton = halton_unit(
        halton_n, seed + 701, cfg.scramble_halton, skip=53
    )
    pieces.append(remainder_halton)
    labels.extend(["HaltonRemainder"] * halton_n)

    gradient_candidates = halton_unit(
        cfg.candidate_multiplier * max(gradient_n, 8),
        seed + 809,
        cfg.scramble_halton,
        skip=67,
    )
    gradient_interp = grid_interpolator(
        pilot["x"], pilot["y"], pilot["gradient"]
    )
    gradient_points = systematic_weighted_pick(
        gradient_candidates, gradient_interp(gradient_candidates), gradient_n
    )
    pieces.append(gradient_points)
    labels.extend(["PilotGradient"] * gradient_n)

    points = unique_fill(np.vstack(pieces), n, fill_pool)
    if len(labels) < n:
        labels.extend(["CoverageFill"] * (n - len(labels)))
    return points, np.asarray(labels[:n], dtype=object), {
        "coverage_floor": floor_n,
        "green_uniform": green_n,
        "halton_remainder": halton_n,
        "pilot_gradient": gradient_n,
    }


# =============================================================================
# Global PHS-RBF-FD approximation engine
# =============================================================================

POLY_POWERS_DEG2 = [
    (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)
]


def poly_matrix(local_coordinates: np.ndarray) -> np.ndarray:
    return np.column_stack([
        (local_coordinates[:, 0] ** px) * (local_coordinates[:, 1] ** py)
        for px, py in POLY_POWERS_DEG2
    ])


def poly_rhs(operator: str, scale: float) -> np.ndarray:
    if operator == "lap":
        return np.array([
            0.0, 0.0, 0.0,
            2.0 / scale**2, 0.0, 2.0 / scale**2,
        ])
    if operator == "dx":
        return np.array([0.0, 1.0 / scale, 0.0, 0.0, 0.0, 0.0])
    if operator == "dy":
        return np.array([0.0, 0.0, 1.0 / scale, 0.0, 0.0, 0.0])
    if operator == "value":
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    raise ValueError(operator)


def phs_rhs_at_center(
    local_stencil: np.ndarray, operator: str, scale: float
) -> np.ndarray:
    radius = np.sqrt(np.sum(local_stencil**2, axis=1))
    if operator == "lap":
        return 9.0 * radius / scale**2
    if operator == "dx":
        return -3.0 * radius * local_stencil[:, 0] / scale
    if operator == "dy":
        return -3.0 * radius * local_stencil[:, 1] / scale
    if operator == "value":
        return radius**3
    raise ValueError(operator)


def rbffd_weights(
    nodes: np.ndarray,
    center: np.ndarray,
    stencil_index: np.ndarray,
    operator: str,
) -> np.ndarray:
    stencil = nodes[stencil_index]
    scale = float(np.max(np.linalg.norm(stencil - center, axis=1)))
    if scale <= 1.0e-14:
        raise RuntimeError("degenerate RBF-FD stencil scale.")
    local = (stencil - center) / scale
    pairwise = local[:, None, :] - local[None, :, :]
    phi = np.linalg.norm(pairwise, axis=2) ** 3
    polynomial = poly_matrix(local)
    m, q = len(stencil_index), polynomial.shape[1]
    system = np.zeros((m + q, m + q), dtype=float)
    system[:m, :m] = phi
    system[:m, m:] = polynomial
    system[m:, :m] = polynomial.T
    rhs = np.zeros(m + q, dtype=float)
    rhs[:m] = phs_rhs_at_center(local, operator, scale)
    rhs[m:] = poly_rhs(operator, scale)
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(system, rhs, rcond=1.0e-12)
    return solution[:m]


def boundary_count(budget: int, cfg: Config) -> int:
    return max(
        cfg.min_boundary_each_edge,
        int(round(cfg.boundary_factor * math.sqrt(max(budget, 1)))),
    )


def square_boundary(points_per_edge: int) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, points_per_edge)
    bottom = np.column_stack([parameter, np.zeros_like(parameter)])
    top = np.column_stack([parameter, np.ones_like(parameter)])
    left = np.column_stack([np.zeros(points_per_edge - 2), parameter[1:-1]])
    right = np.column_stack([np.ones(points_per_edge - 2), parameter[1:-1]])
    return np.vstack([bottom, top, left, right])


@dataclass
class Cloud:
    family: str
    nodes: np.ndarray
    interior_idx: np.ndarray
    boundary_idx: np.ndarray
    labels: np.ndarray
    counts: Dict[str, int]
    lap_stencils: List[Tuple[np.ndarray, np.ndarray]]
    build_time_s: float


def build_cloud(
    family: str,
    interior: np.ndarray,
    labels: np.ndarray,
    counts: Dict[str, int],
    points_per_edge: int,
    cfg: Config,
) -> Cloud:
    t0 = time.perf_counter()
    boundary = square_boundary(points_per_edge)
    nodes = np.vstack([interior, boundary])
    n_interior = len(interior)
    interior_idx = np.arange(n_interior, dtype=int)
    boundary_idx = np.arange(n_interior, len(nodes), dtype=int)
    all_labels = np.concatenate([
        labels,
        np.array(["PhysicalBoundary"] * len(boundary), dtype=object),
    ])
    tree = cKDTree(nodes)
    stencil_size = min(cfg.stencil_size, len(nodes))
    lap_stencils: List[Tuple[np.ndarray, np.ndarray]] = []
    for row in interior_idx:
        _, stencil_index = tree.query(nodes[row], k=stencil_size)
        stencil_index = np.atleast_1d(stencil_index).astype(int)
        weights = rbffd_weights(nodes, nodes[row], stencil_index, "lap")
        lap_stencils.append((stencil_index, weights))
    return Cloud(
        family=family,
        nodes=nodes,
        interior_idx=interior_idx,
        boundary_idx=boundary_idx,
        labels=all_labels,
        counts=counts,
        lap_stencils=lap_stencils,
        build_time_s=time.perf_counter() - t0,
    )


def equilibrate_rows(matrix: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale = 1.0 / np.maximum(norms, 1.0e-14)
    return csr_matrix(diags(scale) @ matrix), scale


def condition_proxy_dense(matrix: csr_matrix) -> float:
    if matrix.shape[0] > 850:
        return float("nan")
    return float(np.linalg.cond(matrix.toarray()))


def solve_global(cloud: Cloud) -> Dict[str, Any]:
    t0 = time.perf_counter()
    size = len(cloud.nodes)
    matrix = lil_matrix((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)
    for row, (stencil_index, weights) in zip(
        cloud.interior_idx, cloud.lap_stencils
    ):
        matrix[row, stencil_index] = -weights
        x, y = cloud.nodes[row]
        rhs[row] = float(source_f(x, y))
    for row in cloud.boundary_idx:
        matrix[row, row] = 1.0
        x, y = cloud.nodes[row]
        rhs[row] = float(exact_u(x, y))
    raw = csr_matrix(matrix)
    equilibrated, row_scale = equilibrate_rows(raw)
    assembly_time = time.perf_counter() - t0
    condition = condition_proxy_dense(equilibrated)
    t1 = time.perf_counter()
    values = spsolve(equilibrated, row_scale * rhs)
    solve_time = time.perf_counter() - t1
    return {
        "values": values,
        "condition_proxy": condition,
        "assembly_time_s": assembly_time,
        "solve_time_s": solve_time,
        "matrix_shape": equilibrated.shape,
    }


# =============================================================================
# Independent evaluation and geometric diagnostics
# =============================================================================

def interpolated_grid(
    cloud: Cloud, values: np.ndarray, ex: np.ndarray, ey: np.ndarray
) -> np.ndarray:
    EX, EY = np.meshgrid(ex, ey, indexing="ij")
    points = np.column_stack([EX.ravel(), EY.ravel()])
    interpolator = LinearNDInterpolator(cloud.nodes, values, fill_value=np.nan)
    field = interpolator(points)
    if np.any(~np.isfinite(field)):
        count = int(np.sum(~np.isfinite(field)))
        raise RuntimeError(f"interpolation produced {count} non-finite values.")
    return field.reshape(EX.shape)


def field_metrics(
    field: np.ndarray, ex: np.ndarray, ey: np.ndarray
) -> Dict[str, float]:
    EX, EY = np.meshgrid(ex, ey, indexing="ij")
    exact = exact_u(EX, EY)
    error = field - exact
    dx, dy = ex[1] - ex[0], ey[1] - ey[0]
    ux = np.gradient(field, dx, axis=0, edge_order=2)
    uy = np.gradient(field, dy, axis=1, edge_order=2)
    gradient_error = np.hypot(ux - exact_ux(EX, EY), uy - exact_uy(EX, EY))

    d2x = np.gradient(ux, dx, axis=0, edge_order=2)
    d2y = np.gradient(uy, dy, axis=1, edge_order=2)
    residual = -(d2x + d2y) - source_f(EX, EY)
    residual_interior = residual[2:-2, 2:-2]
    source_interior = source_f(EX, EY)[2:-2, 2:-2]
    residual_rmse = float(np.sqrt(np.mean(residual_interior**2)))
    source_rmse = float(np.sqrt(np.mean(source_interior**2)))

    source_integral = trapz1(
        np.array([trapz1(source_f(ex, y), ex) for y in ey]), ey
    )
    outward_flux = (
        trapz1(-ux[0, :], ey)
        + trapz1(ux[-1, :], ey)
        + trapz1(-uy[:, 0], ex)
        + trapz1(uy[:, -1], ex)
    )
    conservation_defect = abs((-outward_flux) - source_integral) / max(
        abs(source_integral), 1.0e-12
    )
    minimum, maximum = float(field.min()), float(field.max())
    range_violation = max(0.0, -minimum) + max(0.0, maximum - 1.0)
    return {
        "full_rmse": float(np.sqrt(np.mean(error**2))),
        "full_linf": float(np.max(np.abs(error))),
        "h1_seminorm_rmse": float(np.sqrt(np.mean(gradient_error**2))),
        "interpolated_grid_residual_rmse": residual_rmse,
        "interpolated_grid_residual_relative": residual_rmse / max(source_rmse, 1.0e-12),
        "range_min": minimum,
        "range_max": maximum,
        "range_violation": float(range_violation),
        "source_integral": float(source_integral),
        "boundary_outward_flux_integral": float(outward_flux),
        "conservation_rel_defect": float(conservation_defect),
    }


def node_metrics(points: np.ndarray) -> Dict[str, float]:
    points = np.asarray(points, float)
    tree = cKDTree(points)
    nearest = tree.query(points, k=2)[0][:, 1]
    grid = np.linspace(0.0, 1.0, 101)
    GX, GY = np.meshgrid(grid, grid, indexing="ij")
    reference_points = np.column_stack([GX.ravel(), GY.ravel()])
    fill_distance = float(np.max(tree.query(reference_points, k=1)[0]))

    cell_count = 16
    ix = np.clip((points[:, 0] * cell_count).astype(int), 0, cell_count - 1)
    iy = np.clip((points[:, 1] * cell_count).astype(int), 0, cell_count - 1)
    occupancy = np.zeros((cell_count, cell_count), dtype=int)
    for i, j in zip(ix, iy):
        occupancy[i, j] += 1

    centered_reference = equal_area_centers(len(points))
    transport_cost = distance.cdist(points, centered_reference, metric="sqeuclidean")
    row_index, column_index = linear_sum_assignment(transport_cost)
    w2_matching = float(np.sqrt(np.mean(transport_cost[row_index, column_index])))
    discrepancy = float(qmc.discrepancy(np.clip(points, 0.0, 1.0), method="CD"))
    return {
        "n": int(len(points)),
        "fill_distance_101x101": fill_distance,
        "separation_min": float(nearest.min()),
        "nearest_neighbor_median": float(np.median(nearest)),
        "nearest_neighbor_p90": float(np.percentile(nearest, 90)),
        "nearest_neighbor_p99": float(np.percentile(nearest, 99)),
        "empty_cell_fraction_16x16": float(np.mean(occupancy == 0)),
        "centered_L2_discrepancy": discrepancy,
        "uniform_reference_w2_matching": w2_matching,
    }


def independent_rbffd_residual(
    cloud: Cloud, values: np.ndarray, grid_n: int, stencil_size: int
) -> Dict[str, float]:
    """
    Evaluate -Delta u_h-f at withheld tensor points with fresh local stencils.

    None of these points is a collocation equation.  The diagnostic therefore
    tests the nodal state away from the enforced PDE rows without differentiating
    the piecewise-linear visualization interpolant.
    """
    coordinate = np.linspace(0.0, 1.0, grid_n + 2)[1:-1]
    X, Y = np.meshgrid(coordinate, coordinate, indexing="ij")
    test_points = np.column_stack([X.ravel(), Y.ravel()])
    tree = cKDTree(cloud.nodes)
    k = min(stencil_size, len(cloud.nodes))
    residuals = np.empty(len(test_points), dtype=float)
    sources = source_f(test_points[:, 0], test_points[:, 1])
    for index, point in enumerate(test_points):
        _, stencil_index = tree.query(point, k=k)
        stencil_index = np.atleast_1d(stencil_index).astype(int)
        weights = rbffd_weights(cloud.nodes, point, stencil_index, "lap")
        residuals[index] = -float(np.dot(weights, values[stencil_index])) - float(sources[index])
    rmse = float(np.sqrt(np.mean(residuals**2)))
    source_rmse = float(np.sqrt(np.mean(np.asarray(sources)**2)))
    return {
        "independent_residual_rmse": rmse,
        "independent_residual_relative": rmse / max(source_rmse, 1.0e-12),
        "independent_residual_points": int(len(test_points)),
    }


def local_collocation_residual(cloud: Cloud, values: np.ndarray) -> float:
    residuals = []
    for row, (stencil_index, weights) in zip(
        cloud.interior_idx, cloud.lap_stencils
    ):
        x, y = cloud.nodes[row]
        residuals.append(
            -float(np.dot(weights, values[stencil_index])) - float(source_f(x, y))
        )
    residuals = np.asarray(residuals)
    return float(np.sqrt(np.mean(residuals**2)))


def allocation_gate_rows(
    metrics: pd.DataFrame, fingerprint: Dict[str, Any], cfg: Config
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in metrics.itertuples(index=False):
        checks = [
            (
                "G0_FINGERPRINT_SMOOTH_HOMOGENEOUS",
                bool(fingerprint["classified_smooth_homogeneous"]),
                "pre-solve homogeneous/smooth classification",
            ),
            (
                "G1_CONDITION",
                bool(item.condition_proxy <= cfg.condition_proxy_max),
                f"condition <= {cfg.condition_proxy_max:g}",
            ),
            (
                "G2_RANGE",
                bool(item.range_violation <= cfg.range_violation_max),
                f"range violation <= {cfg.range_violation_max:g}",
            ),
            (
                "G3_CONSERVATION",
                bool(
                    item.conservation_rel_defect
                    <= cfg.conservation_relative_defect_max
                ),
                (
                    "relative conservation defect <= "
                    f"{cfg.conservation_relative_defect_max:g}"
                ),
            ),
            (
                "G4_INDEPENDENT_RESIDUAL",
                bool(
                    item.independent_residual_relative
                    <= cfg.independent_residual_relative_max
                ),
                (
                    "independent relative PDE residual <= "
                    f"{cfg.independent_residual_relative_max:g}"
                ),
            ),
            (
                "G5_COVERAGE_SEPARATION",
                bool(
                    item.empty_cell_fraction_16x16
                    <= cfg.empty_cell_fraction_max
                    and item.separation_min >= cfg.separation_min
                ),
                "declared empty-cell and separation safeguards",
            ),
        ]
        all_pass = all(value for _, value, _ in checks)
        checks.append(("ALL_ADMISSIBILITY_GATES", all_pass, "all G0-G5 gates"))
        for gate, passed, reason in checks:
            rows.append({
                "seed": int(item.seed),
                "budget": int(item.budget),
                "allocation_family": item.allocation_family,
                "gate": gate,
                "passed": bool(passed),
                "reason": reason,
            })
    return pd.DataFrame(rows)


def selector_gate_rows(
    metrics: pd.DataFrame,
    allocation_gates: pd.DataFrame,
    fingerprint: Dict[str, Any],
    cfg: Config,
) -> pd.DataFrame:
    selected_name = str(fingerprint["selected_architecture"])
    selected = metrics[metrics["allocation_family"] == selected_name]
    if len(selected) != 1:
        raise RuntimeError(f"expected exactly one selected row; found {len(selected)}")
    selected_row = selected.iloc[0]
    admissible_families = set(
        allocation_gates[
            (allocation_gates["gate"] == "ALL_ADMISSIBILITY_GATES")
            & allocation_gates["passed"]
        ]["allocation_family"]
    )
    eligible = metrics[metrics["allocation_family"].isin(admissible_families)]
    if eligible.empty:
        best_rmse = best_h1 = fastest = float("nan")
        rmse_ratio = h1_ratio = cost_ratio = float("inf")
    else:
        best_rmse = float(eligible["full_rmse"].min())
        best_h1 = float(eligible["h1_seminorm_rmse"].min())
        fastest = float(eligible["charged_total_time_s"].min())
        rmse_ratio = float(selected_row["full_rmse"] / max(best_rmse, 1.0e-300))
        h1_ratio = float(
            selected_row["h1_seminorm_rmse"] / max(best_h1, 1.0e-300)
        )
        cost_ratio = float(
            selected_row["charged_total_time_s"] / max(fastest, 1.0e-300)
        )

    selected_admissible = selected_name in admissible_families
    no_complex_activation = bool(
        fingerprint["classified_smooth_homogeneous"]
        and not fingerprint["domain_decomposition_activated"]
        and not fingerprint["interface_enrichment_activated"]
        and selected_name in ("HALTON", "EQUAL_AREA_STRATIFIED")
    )
    checks = [
        (
            "S0_MINIMUM_ARCHITECTURE",
            no_complex_activation,
            "simple global method selected before outcomes; DDM/interface enrichment false",
        ),
        (
            "S1_SELECTED_ADMISSIBLE",
            selected_admissible,
            "selected method passes all allocation admissibility gates",
        ),
        (
            "S2_RMSE_COMPETITIVE",
            rmse_ratio <= cfg.competitive_rmse_ratio_max,
            f"selected/best eligible RMSE={rmse_ratio:.6g}",
        ),
        (
            "S3_H1_COMPETITIVE",
            h1_ratio <= cfg.competitive_h1_ratio_max,
            f"selected/best eligible H1={h1_ratio:.6g}",
        ),
        (
            "S4_COST_COMPETITIVE",
            cost_ratio <= cfg.competitive_cost_ratio_max,
            f"selected/fastest eligible charged cost={cost_ratio:.6g}",
        ),
    ]
    final = all(bool(passed) for _, passed, _ in checks)
    checks.append(("SELECTOR_FINAL_GATE", final, "all S0-S4 gates"))
    return pd.DataFrame([
        {
            "seed": int(selected_row["seed"]),
            "budget": int(selected_row["budget"]),
            "selected_architecture": selected_name,
            "gate": gate,
            "passed": bool(passed),
            "reason": reason,
            "best_eligible_rmse": best_rmse,
            "best_eligible_h1": best_h1,
            "fastest_eligible_charged_time_s": fastest,
            "selected_rmse_ratio": rmse_ratio,
            "selected_h1_ratio": h1_ratio,
            "selected_cost_ratio": cost_ratio,
        }
        for gate, passed, reason in checks
    ])


# =============================================================================
# One seed-budget run with automatic resume
# =============================================================================

def run_one(
    seed: int,
    budget: int,
    pilot: Dict[str, Any],
    fingerprint: Dict[str, Any],
    cfg: Config,
    campaign_dir: Path,
) -> Dict[str, Any]:
    run_dir = campaign_dir / f"seed_{seed:04d}_budget_{budget:04d}"
    ensure_dir(run_dir)
    status_path = run_dir / "status.json"
    if status_path.exists() and not cfg.force:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "COMPLETED":
                return {
                    "seed": seed,
                    "budget": budget,
                    "status": "COMPLETED",
                    "run_dir": str(run_dir),
                    "resumed": True,
                }
        except Exception:
            pass

    logger = setup_logger(run_dir)
    run_started = time.time()
    try:
        logger.info("Starting E run: seed=%d budget=%d", seed, budget)
        run_config = {
            "seed": seed,
            "budget": budget,
            "profile": cfg.profile,
            "config": asdict(cfg),
            "fingerprint": fingerprint,
            "no_domain_decomposition": True,
            "no_interface_enrichment": True,
        }
        run_config["provenance_key"] = short_hash({
            "seed": seed,
            "budget": budget,
            "profile": cfg.profile,
            "frozen_contract": frozen_experiment_contract(cfg)[
                "frozen_experiment_sha256"
            ],
            "fingerprint": fingerprint_for_hash(fingerprint),
            "allocation_families": ALLOCATION_FAMILIES,
        })
        json_dump(run_dir / "run_config.json", run_config)
        json_dump(run_dir / "physical_fingerprint.json", fingerprint)

        points_per_edge = boundary_count(budget, cfg)
        allocation_rows: List[Dict[str, Any]] = []
        node_rows: List[Dict[str, Any]] = []
        state_arrays: Dict[str, np.ndarray] = {}
        ex = np.linspace(0.0, 1.0, cfg.eval_nx)
        ey = np.linspace(0.0, 1.0, cfg.eval_ny)

        for family_index, family in enumerate(ALLOCATION_FAMILIES):
            logger.info("Matched global solve: %s", family)
            generation_start = time.perf_counter()
            interior, labels, counts = build_interior_nodes(
                budget, family, pilot, seed + 10007 * family_index, cfg
            )
            cloud = build_cloud(
                family, interior, labels, counts, points_per_edge, cfg
            )
            generation_time = time.perf_counter() - generation_start
            solved = solve_global(cloud)
            field = interpolated_grid(cloud, solved["values"], ex, ey)
            field_row = field_metrics(field, ex, ey)
            field_row.update(independent_rbffd_residual(
                cloud,
                solved["values"],
                cfg.independent_residual_grid_n,
                cfg.stencil_size,
            ))
            node_row = node_metrics(interior)
            collocation_residual = local_collocation_residual(
                cloud, solved["values"]
            )
            charged_time = (
                float(fingerprint["pilot_time_s"])
                + generation_time
                + solved["assembly_time_s"]
                + solved["solve_time_s"]
            )
            provenance = short_hash({
                "seed": seed,
                "budget": budget,
                "family": family,
                "counts": counts,
                "fingerprint": fingerprint_for_hash(fingerprint),
                "engine": {
                    "phs": cfg.phs_power,
                    "poly": cfg.poly_degree,
                    "stencil": cfg.stencil_size,
                    "boundary_points_per_edge": points_per_edge,
                },
            })
            allocation_rows.append({
                "seed": seed,
                "budget": budget,
                "allocation_family": family,
                **field_row,
                "local_collocation_residual_rmse": collocation_residual,
                "condition_proxy": solved["condition_proxy"],
                "n_interior": budget,
                "n_boundary": len(cloud.boundary_idx),
                "n_total": len(cloud.nodes),
                "points_per_boundary_edge": points_per_edge,
                "pilot_time_s": float(fingerprint["pilot_time_s"]),
                "generation_time_s": generation_time,
                "cloud_build_time_s": cloud.build_time_s,
                "assembly_time_s": solved["assembly_time_s"],
                "solve_time_s": solved["solve_time_s"],
                "charged_total_time_s": charged_time,
                "domain_decomposition_activated": False,
                "interface_enrichment_activated": False,
                "selected_by_fingerprint": family == fingerprint["selected_architecture"],
                "provenance_key": provenance,
            })
            node_rows.append({
                "seed": seed,
                "budget": budget,
                "allocation_family": family,
                **node_row,
                **{f"count_{name}": int(value) for name, value in counts.items()},
                "provenance_key": provenance,
            })
            pd.DataFrame({
                "x": interior[:, 0],
                "y": interior[:, 1],
                "component_label": labels,
            }).to_csv(run_dir / f"nodes_{family}.csv", index=False)
            if cfg.save_full_arrays:
                state_arrays[f"{family}__nodes"] = cloud.nodes
                state_arrays[f"{family}__values"] = solved["values"]
                state_arrays[f"{family}__evaluation_field"] = field

        allocation = pd.DataFrame(allocation_rows)
        nodes = pd.DataFrame(node_rows)
        gates = allocation_gate_rows(allocation.merge(
            nodes[
                [
                    "seed", "budget", "allocation_family",
                    "empty_cell_fraction_16x16", "separation_min",
                ]
            ],
            on=["seed", "budget", "allocation_family"],
            validate="one_to_one",
        ), fingerprint, cfg)
        selector = selector_gate_rows(allocation, gates, fingerprint, cfg)
        allocation.to_csv(run_dir / "allocation_metrics.csv", index=False)
        nodes.to_csv(run_dir / "node_metrics.csv", index=False)
        gates.to_csv(run_dir / "allocation_gate_matrix.csv", index=False)
        selector.to_csv(run_dir / "selector_gate_matrix.csv", index=False)
        if cfg.save_full_arrays:
            state_arrays["evaluation_x"] = ex
            state_arrays["evaluation_y"] = ey
            state_arrays["pilot_x"] = pilot["x"]
            state_arrays["pilot_y"] = pilot["y"]
            state_arrays["pilot_u"] = pilot["u"]
            state_arrays["pilot_gradient"] = pilot["gradient"]
            state_arrays["pilot_residual"] = pilot["residual"]
            np.savez_compressed(run_dir / "state_arrays.npz", **state_arrays)

        status = {
            "status": "COMPLETED",
            "seed": seed,
            "budget": budget,
            "profile": cfg.profile,
            "provenance_key": run_config["provenance_key"],
            "completed_unix": time.time(),
            "elapsed_s": time.time() - run_started,
        }
        json_dump(status_path, status)
        logger.info("Completed E run: seed=%d budget=%d", seed, budget)
        close_logger(logger)
        write_manifest(run_dir, "run_manifest_sha256.csv")
        return {
            "seed": seed,
            "budget": budget,
            "status": "COMPLETED",
            "run_dir": str(run_dir),
            "resumed": False,
        }
    except Exception as exc:
        logger.error("Run failed: %s", exc)
        logger.error(traceback.format_exc())
        json_dump(status_path, {
            "status": "FAILED",
            "seed": seed,
            "budget": budget,
            "profile": cfg.profile,
            "error": repr(exc),
            "failed_unix": time.time(),
        })
        close_logger(logger)
        write_manifest(run_dir, "run_manifest_sha256.csv")
        raise


# =============================================================================
# Campaign aggregation and paper-facing figures
# =============================================================================

def read_completed_tables(campaign_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(campaign_dir.glob("seed_*_budget_*")):
        status_path = run_dir / "status.json"
        table_path = run_dir / filename
        if not status_path.exists() or not table_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETED":
            frames.append(pd.read_csv(table_path))
    if not frames:
        raise RuntimeError(f"no completed {filename} tables found.")
    return pd.concat(frames, ignore_index=True)


def median_range(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.groupby(["allocation_family", "budget"])[metric].agg(
        median="median", minimum="min", maximum="max"
    ).reset_index()


def plot_accuracy(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    colors = dict(zip(ALLOCATION_FAMILIES, plt.cm.tab10.colors[:4]))
    for axis, metric, title in [
        (axes[0], "full_rmse", "Field RMSE"),
        (axes[1], "h1_seminorm_rmse", "H1-seminorm RMSE"),
    ]:
        summary = median_range(metrics, metric)
        for family in ALLOCATION_FAMILIES:
            sub = summary[summary["allocation_family"] == family]
            axis.plot(sub["budget"], sub["median"], "o-", label=family, color=colors[family])
            axis.fill_between(
                sub["budget"], sub["minimum"], sub["maximum"],
                alpha=0.14, color=colors[family],
            )
        axis.set_yscale("log")
        axis.set_xlabel("Interior-node budget N")
        axis.set_ylabel(title)
        axis.set_title(title + ": median and seed range")
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_residual_condition(metrics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    colors = dict(zip(ALLOCATION_FAMILIES, plt.cm.tab10.colors[:4]))
    for axis, metric, title in [
        (axes[0], "independent_residual_relative", "Independent relative PDE residual"),
        (axes[1], "condition_proxy", "Row-equilibrated condition proxy"),
    ]:
        summary = median_range(metrics, metric)
        for family in ALLOCATION_FAMILIES:
            sub = summary[summary["allocation_family"] == family]
            axis.plot(sub["budget"], sub["median"], "o-", label=family, color=colors[family])
            axis.fill_between(
                sub["budget"], sub["minimum"], sub["maximum"],
                alpha=0.14, color=colors[family],
            )
        axis.set_yscale("log")
        axis.set_xlabel("Interior-node budget N")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_node_diagnostics(nodes: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    metrics = [
        ("fill_distance_101x101", "Fill distance"),
        ("empty_cell_fraction_16x16", "Empty-cell fraction"),
        ("centered_L2_discrepancy", "Centered L2 discrepancy"),
        ("uniform_reference_w2_matching", "Uniform-reference W2 matching proxy"),
    ]
    colors = dict(zip(ALLOCATION_FAMILIES, plt.cm.tab10.colors[:4]))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        summary = median_range(nodes, metric)
        for family in ALLOCATION_FAMILIES:
            sub = summary[summary["allocation_family"] == family]
            axis.plot(sub["budget"], sub["median"], "o-", label=family, color=colors[family])
            axis.fill_between(
                sub["budget"], sub["minimum"], sub["maximum"],
                alpha=0.14, color=colors[family],
            )
        axis.set_xlabel("Interior-node budget N")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_selector(selector: pd.DataFrame, output: Path) -> None:
    final = selector[selector["gate"] == "SELECTOR_FINAL_GATE"].copy()
    selected_name = str(final["selected_architecture"].iloc[0])
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for column, label, color in [
        ("selected_rmse_ratio", "RMSE ratio", "#1f77b4"),
        ("selected_h1_ratio", "H1 ratio", "#ff7f0e"),
        ("selected_cost_ratio", "Cost ratio", "#2ca02c"),
    ]:
        summary = final.groupby("budget")[column].agg(
            median="median", minimum="min", maximum="max"
        ).reset_index()
        axes[0].plot(summary["budget"], summary["median"], "o-", label=label, color=color)
        axes[0].fill_between(
            summary["budget"], summary["minimum"], summary["maximum"],
            alpha=0.14, color=color,
        )
    axes[0].axhline(1.25, color="#555555", linestyle="--", linewidth=1.0, label="error limit")
    axes[0].axhline(1.50, color="#999999", linestyle=":", linewidth=1.0, label="cost limit")
    axes[0].set_xlabel("Interior-node budget N")
    axes[0].set_ylabel("Selected / best eligible")
    axes[0].set_title(f"Preselected {selected_name} competitiveness")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    counts = selector.groupby("gate")["passed"].agg(["sum", "count"]).reset_index()
    axes[1].barh(counts["gate"], counts["sum"], color="#3b82f6")
    axes[1].set_xlim(0, max(1, counts["count"].max()))
    axes[1].set_xlabel("Pass count")
    axes[1].set_title("Selector gates across seed-budget rows")
    axes[1].grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_allocation_example(campaign_dir: Path, budgets: List[int], output: Path) -> None:
    run_dir = campaign_dir / f"seed_{11:04d}_budget_{max(budgets):04d}"
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 9.0), sharex=True, sharey=True)
    for axis, family in zip(axes.ravel(), ALLOCATION_FAMILIES):
        nodes = pd.read_csv(run_dir / f"nodes_{family}.csv")
        axis.scatter(nodes["x"], nodes["y"], s=8, alpha=0.75)
        axis.set_title(family)
        axis.set_aspect("equal")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.15)
    figure.suptitle(f"Matched global allocations, seed 11, N={max(budgets)}")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def campaign_summary_text(
    cfg: Config,
    fingerprint: Dict[str, Any],
    allocation: pd.DataFrame,
    nodes: pd.DataFrame,
    selector: pd.DataFrame,
    seeds: List[int],
    budgets: List[int],
) -> str:
    final = selector[selector["gate"] == "SELECTOR_FINAL_GATE"]
    family_summary = allocation.groupby("allocation_family").agg(
        rows=("seed", "size"),
        full_rmse_median=("full_rmse", "median"),
        h1_median=("h1_seminorm_rmse", "median"),
        residual_median=("independent_residual_relative", "median"),
        condition_max=("condition_proxy", "max"),
        charged_time_median=("charged_total_time_s", "median"),
    )
    node_summary = nodes.groupby("allocation_family").agg(
        fill_median=("fill_distance_101x101", "median"),
        empty_median=("empty_cell_fraction_16x16", "median"),
        discrepancy_median=("centered_L2_discrepancy", "median"),
        w2_median=("uniform_reference_w2_matching", "median"),
    )
    scientific = cfg.profile == "production"
    decision = (
        "R3 E-BRANCH PASS"
        if scientific and len(final) == len(seeds) * len(budgets) and final["passed"].all()
        else (
            "R3 E-BRANCH FAIL/DEFER"
            if scientific
            else "NON-SCIENTIFIC SMOKE: NO R3/A1 ADJUDICATION"
        )
    )
    return "\n".join([
        "CGF PAPER-I SMOOTH HOMOGENEOUS NON-ACTIVATION SUMMARY",
        "=" * 72,
        f"Profile: {cfg.profile}",
        f"Seeds: {seeds}",
        f"Budgets: {budgets}",
        f"Problem class: {fingerprint['problem_class']}",
        f"Preselected architecture: {fingerprint['selected_architecture']}",
        f"Domain decomposition activated: {fingerprint['domain_decomposition_activated']}",
        f"Interface enrichment activated: {fingerprint['interface_enrichment_activated']}",
        f"Pilot high-frequency fraction: {fingerprint['pilot_high_frequency_energy_fraction']:.8g}",
        "",
        "ALLOCATION METRICS",
        "------------------",
        family_summary.to_string(),
        "",
        "NODE METRICS",
        "------------",
        node_summary.to_string(),
        "",
        "SELECTOR",
        "--------",
        f"Final selector passages: {int(final['passed'].sum())}/{len(final)}",
        decision,
        "",
        "Interpretation discipline: E tests correct non-activation only. It contains",
        "no DDM, interface collar, or transmission law. The final E/S/F thresholds",
        "are frozen only after the independent F experiment is complete.",
        "",
    ])


def aggregate_campaign(
    campaign_dir: Path,
    cfg: Config,
    pilot: Dict[str, Any],
    fingerprint: Dict[str, Any],
    seeds: List[int],
    budgets: List[int],
) -> None:
    allocation = read_completed_tables(campaign_dir, "allocation_metrics.csv")
    nodes = read_completed_tables(campaign_dir, "node_metrics.csv")
    gates = read_completed_tables(campaign_dir, "allocation_gate_matrix.csv")
    selector = read_completed_tables(campaign_dir, "selector_gate_matrix.csv")
    expected_rows = len(seeds) * len(budgets) * len(ALLOCATION_FAMILIES)
    if len(allocation) != expected_rows or len(nodes) != expected_rows:
        raise RuntimeError(
            f"incomplete campaign rectangle: expected {expected_rows} allocation rows."
        )
    if allocation.duplicated(["seed", "budget", "allocation_family"]).any():
        raise RuntimeError("duplicate allocation rows detected.")
    allocation.to_csv(campaign_dir / "campaign_allocation_metrics.csv", index=False)
    nodes.to_csv(campaign_dir / "campaign_node_metrics.csv", index=False)
    gates.to_csv(campaign_dir / "campaign_allocation_gate_matrix.csv", index=False)
    selector.to_csv(campaign_dir / "campaign_selector_gate_matrix.csv", index=False)

    refinement = allocation.groupby(["allocation_family", "budget"]).agg(
        seeds=("seed", "nunique"),
        full_rmse_median=("full_rmse", "median"),
        full_rmse_min=("full_rmse", "min"),
        full_rmse_max=("full_rmse", "max"),
        h1_median=("h1_seminorm_rmse", "median"),
        h1_min=("h1_seminorm_rmse", "min"),
        h1_max=("h1_seminorm_rmse", "max"),
        residual_median=("independent_residual_relative", "median"),
        condition_median=("condition_proxy", "median"),
        charged_time_median=("charged_total_time_s", "median"),
    ).reset_index()
    refinement.to_csv(campaign_dir / "seed_refinement_summary.csv", index=False)

    plot_accuracy(allocation, campaign_dir / "smooth_homogeneous_accuracy_refinement.png")
    plot_residual_condition(
        allocation, campaign_dir / "smooth_homogeneous_residual_condition.png"
    )
    plot_node_diagnostics(
        nodes, campaign_dir / "smooth_homogeneous_node_diagnostics.png"
    )
    plot_selector(
        selector, campaign_dir / "smooth_homogeneous_selector_competitiveness.png"
    )
    plot_allocation_example(
        campaign_dir, budgets, campaign_dir / "smooth_homogeneous_allocations.png"
    )

    figure_sources = [
        ("smooth_homogeneous_accuracy_refinement.png", "campaign_allocation_metrics.csv", "Results: E accuracy"),
        ("smooth_homogeneous_residual_condition.png", "campaign_allocation_metrics.csv", "Results: E admissibility"),
        ("smooth_homogeneous_node_diagnostics.png", "campaign_node_metrics.csv", "Results: E allocation geometry"),
        ("smooth_homogeneous_selector_competitiveness.png", "campaign_selector_gate_matrix.csv", "Algorithm: non-activation"),
        ("smooth_homogeneous_allocations.png", "nodes_* per-run CSV", "Methods: matched allocations"),
    ]
    provenance_rows = []
    for figure_name, source_table, destination in figure_sources:
        figure_path = campaign_dir / figure_name
        provenance_rows.append({
            "figure_filename": figure_name,
            "sha256": sha256_file(figure_path),
            "source_table": source_table,
            "paper_destination": destination,
            "profile": cfg.profile,
        })
    pd.DataFrame(provenance_rows).to_csv(
        campaign_dir / "figure_provenance.csv", index=False
    )

    gap_rows = [
        ("M1-E", "Smooth homogeneous benchmark definition", "frozen_experiment_contract.json", "Model Problems"),
        ("R3", "Non-activation case", "campaign_selector_gate_matrix.csv", "Main Results"),
        ("A1-E", "Executable E fingerprint branch", "physical_fingerprint.json", "Algorithm"),
        ("F1-E", "E paper-facing figures", "figure_provenance.csv", "Figures"),
        ("X1-E", "Replay provenance", "manifests/configs/state arrays", "Data/Code"),
    ]
    pd.DataFrame(
        gap_rows,
        columns=["gap_id", "manuscript_object", "output", "destination"],
    ).to_csv(campaign_dir / "manuscript_gap_map.csv", index=False)

    json_dump(campaign_dir / "physical_fingerprint.json", fingerprint)
    np.savez_compressed(
        campaign_dir / "pilot_state.npz",
        x=pilot["x"],
        y=pilot["y"],
        u=pilot["u"],
        gradient=pilot["gradient"],
        residual_x=pilot["residual_x"],
        residual_y=pilot["residual_y"],
        residual=pilot["residual"],
    )
    summary = campaign_summary_text(
        cfg, fingerprint, allocation, nodes, selector, seeds, budgets
    )
    (campaign_dir / "nonactivation_decision_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    write_manifest(campaign_dir, "campaign_manifest_sha256.csv")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CGF smooth-homogeneous non-activation campaign"
    )
    parser.add_argument("--profile", choices=["production", "smoke"], default=PROFILE)
    parser.add_argument("--campaign-dir", default=CAMPAIGN_DIR)
    parser.add_argument("--force", action="store_true", default=FORCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        profile=args.profile,
        campaign_dir=args.campaign_dir,
        force=args.force,
    )
    cfg.validate()
    seeds, budgets = profile_seeds_budgets(cfg.profile)
    campaign_dir = Path(
        cfg.campaign_dir or default_campaign_dir(cfg.profile)
    ).resolve()
    ensure_dir(campaign_dir)

    contract = frozen_experiment_contract(cfg)
    json_dump(campaign_dir / "frozen_experiment_contract.json", contract)
    json_dump(campaign_dir / "campaign_config.json", {
        **asdict(cfg),
        "seeds": seeds,
        "budgets": budgets,
        "allocation_families": ALLOCATION_FAMILIES,
        "frozen_experiment_sha256": contract["frozen_experiment_sha256"],
        "scientific_profile": cfg.profile == "production",
    })
    json_dump(campaign_dir / "environment.json", environment_record())

    pilot = build_structured_pilot(cfg)
    fingerprint = physical_fingerprint(pilot, cfg)
    if not fingerprint["classified_smooth_homogeneous"]:
        raise RuntimeError(
            "The declared E benchmark did not pass its pre-solve fingerprint. "
            "Do not inspect allocation outcomes or retune thresholds."
        )
    if fingerprint["selected_architecture"] != cfg.selected_global_family:
        raise RuntimeError("pre-solve selector did not retain the frozen global family.")

    run_index = []
    for seed in seeds:
        for budget in budgets:
            run_index.append(
                run_one(seed, budget, pilot, fingerprint, cfg, campaign_dir)
            )
            pd.DataFrame(run_index).to_csv(
                campaign_dir / "campaign_run_index.csv", index=False
            )

    aggregate_campaign(
        campaign_dir, cfg, pilot, fingerprint, seeds, budgets
    )
    marker_name = (
        "production_complete.json"
        if cfg.profile == "production"
        else "smoke_complete_NON_SCIENTIFIC.json"
    )
    json_dump(campaign_dir / marker_name, {
        "status": "COMPLETED",
        "profile": cfg.profile,
        "scientific": cfg.profile == "production",
        "completed_seed_budget_rows": len(seeds) * len(budgets),
        "frozen_experiment_sha256": contract["frozen_experiment_sha256"],
        "completed_unix": time.time(),
    })
    write_manifest(campaign_dir, "campaign_manifest_sha256.csv")
    print((campaign_dir / "nonactivation_decision_summary.txt").read_text(encoding="utf-8"))
    print(f"Campaign directory: {campaign_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
