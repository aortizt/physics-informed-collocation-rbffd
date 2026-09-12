#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGF static-interface production and untuned-holdout experiment.

This standalone Spyder script closes the Paper-I static-interface evidence
block using one frozen numerical experiment.  It keeps the PHS r^3 RBF-FD
engine, quadratic augmentation, protected low-discrepancy skeleton, evolving
coarse residual correction, and the five previously declared transmission
conditions.  No sixth transmission condition and no post-hoc threshold tuning
are permitted after the production run begins.

The production benchmark is xi=0.50, K_L=1, K_R=100.  The holdout benchmark is
xi=0.37, K_L=1, K_R=30.  Both use seeds [11,23,37,53,71] and total interior
budgets [120,180,260,360].  The holdout profile refuses to run unless it finds
a completed production marker with the identical frozen-experiment hash.

The causal comparisons are:
  HALTON; GHG_PROTECTED; and four same-engine ablations removing the Green
  enrichment, Halton remainder enrichment, coefficient-gradient enrichment,
  or the protected coverage floor.  Domain-decomposition branches remain the
  full 2-allocation x 5-transmission factorial for HALTON and GHG_PROTECTED.

Gate 3 requires q_net<0.95, q_tail<0.95, and terminal physical mismatch
<=5e-4.  The PHS condition-proxy ceiling is 2e6, chosen prospectively as more
than 100 times the largest viable P21-v2 screen value while still excluding
the earlier dense-Gaussian pathology.

Spyder use:
  1. Leave PROFILE="production" and press Run.  Completed seed/budget folders
     are resumed automatically.
  2. After production completes, set PROFILE="holdout" and press Run.
  3. Read production_decision_summary.txt and production_gate_matrix.csv.

Scientific cutoff: if ROBIN_GHG_LEGACY_CONTROL fails any final coupling or
assembly gate, integrated GHG-conditioned domain decomposition is FAIL/DEFER
for Paper I.  The failure remains a mechanistic result; operator invention and
retuning belong to Paper II.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Single-thread default MUST be set before NumPy/SciPy are imported.
# -----------------------------------------------------------------------------
import os
import tempfile
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "cgf_matplotlib"))

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, bmat, diags
from scipy.sparse.linalg import splu, spsolve
from scipy.stats import qmc


# =============================================================================
# TOP-LEVEL CONFIGURATION
# =============================================================================

# Spyder-facing switches.  Pressing Run starts/resumes the production
# experiment.  Change only PROFILE to "holdout" after production completes.
PROFILE = "production"
FORCE = False

@dataclass
class Config:
    # Campaign / provenance
    profile: str = PROFILE
    campaign_dir: str = ""
    force: bool = FORCE
    save_full_arrays: bool = True

    # Benchmark S
    interface_x: float = 0.50
    k_left: float = 1.0
    k_right: float = 100.0

    # Non-oracle pilot used ONLY for GHG detection and initialization
    pilot_nx: int = 41
    pilot_ny: int = 41
    collar_factor: float = 2.5
    green_decay_factor: float = 1.0
    density_floor: float = 0.05

    # GHG nominal mixture.  The protected coverage floor is a separate
    # operational construction constraint, not a redefinition of beta.
    alpha_green: float = 1.0 / 3.0
    beta_halton: float = 1.0 / 3.0
    gamma_gradient: float = 1.0 / 3.0
    coverage_floor_fraction: float = 0.45
    candidate_multiplier: int = 16

    # RBF-FD engine
    phs_power: int = 3
    poly_degree: int = 2
    stencil_size: int = 25

    # Boundary/interface point scaling with budget
    boundary_factor: float = 1.55
    interface_factor: float = 1.45
    min_boundary_each_edge: int = 14
    min_interface: int = 16

    # Schwarz / coarse correction
    max_schwarz_iter: int = 18
    schwarz_relaxation: float = 0.75
    # Even nx places the x=0.5 material interface on a finite-volume face,
    # so the coarse correction has an unambiguous conservative interface flux.
    coarse_nx: int = 14
    coarse_ny: int = 13
    coarse_relaxation: float = 0.55
    ddm_tol: float = 5e-4

    # Transmission constants
    robin_scalar_lambda: float = 8.0
    robin_ghg_strength: float = 1.0

    # Evaluation
    eval_nx: int = 101
    eval_ny: int = 101

    # Prospectively frozen production gates.
    q_margin: float = 0.95
    range_violation_tol: float = 2e-2
    conservation_rel_tol: float = 5e-2
    condition_proxy_tol: float = 2e6
    competitive_error_ratio: float = 1.25
    ghg_coverage_empty_ratio_tol: float = 1.50

    # Randomized low-discrepancy robustness.  All compared allocation
    # families use the same scramble status and declared seed.
    scramble_halton: bool = True

    def validate(self) -> None:
        w = self.alpha_green + self.beta_halton + self.gamma_gradient
        if not np.isclose(w, 1.0):
            raise ValueError(f"alpha+beta+gamma must equal one; got {w}")
        if min(self.alpha_green, self.beta_halton, self.gamma_gradient) < 0:
            raise ValueError("GHG weights must be nonnegative.")
        if not (0 < self.coverage_floor_fraction < 1):
            raise ValueError("coverage_floor_fraction must lie in (0,1).")
        if self.coverage_floor_fraction < self.beta_halton:
            raise ValueError(
                "The protected coverage floor must not be smaller than the nominal beta share."
            )
        if self.phs_power != 3:
            raise ValueError("This implementation is derived explicitly for PHS r^3.")
        if self.poly_degree != 2:
            raise ValueError("This implementation currently uses quadratic augmentation.")
        if self.stencil_size < 12:
            raise ValueError("stencil_size too small for stable quadratic augmentation.")
        if not (0 < self.interface_x < 1):
            raise ValueError("interface_x must lie inside the domain.")
        if min(self.k_left, self.k_right) <= 0:
            raise ValueError("Diffusion/permeability coefficients must be positive.")


BASE_CONFIG = Config()


def profile_seeds_budgets(profile: str) -> Tuple[List[int], List[int]]:
    """Declared profiles; production and holdout are both 5-seed x 4-budget."""
    p = profile.lower()
    if p == "smoke":
        return [11], [60]
    if p in ("production", "holdout"):
        return [11, 23, 37, 53, 71], [120, 180, 260, 360]
    raise ValueError(f"Unknown profile: {profile}")


TRANSMISSION_LIBRARY = [
    "DIRICHLET",
    "ROBIN_SCALAR",
    "ROBIN_COEFFICIENT_SCALED",
    "ROBIN_DTN_ASYMMETRIC",
    "ROBIN_GHG_LEGACY_CONTROL",
]

ALLOCATION_FAMILIES = [
    "HALTON",
    "GHG_PROTECTED",
    "GHG_NO_GREEN",
    "GHG_NO_HALTON_ENRICHMENT",
    "GHG_NO_GRADIENT",
    "GHG_NO_COVERAGE_FLOOR",
]

DDM_ALLOCATION_FAMILIES = ["HALTON", "GHG_PROTECTED"]


# =============================================================================
# Utilities and provenance
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(obj: Any, n: int = 14) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:n]


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def canonical_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def default_campaign_dir(profile: str) -> str:
    return {
        "smoke": "static_interface_smoke",
        "production": "static_interface_production_campaign",
        "holdout": "static_interface_holdout_campaign",
    }[profile]


def frozen_experiment_contract(cfg: Config) -> Dict[str, Any]:
    """Scientific choices shared by production and holdout; runtime paths omitted."""
    contract = {
        "contract_version": "CGF-S-1.0",
        "manuscript_scope": "Paper I, static discontinuous interface benchmark S",
        "production_benchmark": {"interface_x": 0.50, "k_left": 1.0, "k_right": 100.0},
        "untuned_holdout_benchmark": {"interface_x": 0.37, "k_left": 1.0, "k_right": 30.0},
        "seeds": [11, 23, 37, 53, 71],
        "budgets": [120, 180, 260, 360],
        "allocation_families": ALLOCATION_FAMILIES,
        "ddm_allocation_families": DDM_ALLOCATION_FAMILIES,
        "transmission_library": TRANSMISSION_LIBRARY,
        "allocation_parameters": {
            "alpha_green": cfg.alpha_green,
            "beta_halton": cfg.beta_halton,
            "gamma_gradient": cfg.gamma_gradient,
            "coverage_floor_fraction": cfg.coverage_floor_fraction,
            "collar_factor": cfg.collar_factor,
            "green_decay_factor": cfg.green_decay_factor,
            "density_floor": cfg.density_floor,
            "candidate_multiplier": cfg.candidate_multiplier,
        },
        "discretization": {
            "phs_power": cfg.phs_power,
            "poly_degree": cfg.poly_degree,
            "stencil_size": cfg.stencil_size,
            "boundary_factor": cfg.boundary_factor,
            "interface_factor": cfg.interface_factor,
            "pilot_nx": cfg.pilot_nx,
            "pilot_ny": cfg.pilot_ny,
        },
        "schwarz_and_coarse_correction": {
            "max_iter": cfg.max_schwarz_iter,
            "schwarz_relaxation": cfg.schwarz_relaxation,
            "coarse_nx": cfg.coarse_nx,
            "coarse_ny": cfg.coarse_ny,
            "coarse_relaxation": cfg.coarse_relaxation,
            "robin_scalar_lambda": cfg.robin_scalar_lambda,
            "robin_ghg_strength": cfg.robin_ghg_strength,
        },
        "final_gates": {
            "q_net_max_exclusive": cfg.q_margin,
            "q_tail_max_exclusive": cfg.q_margin,
            "terminal_physical_mismatch_max": cfg.ddm_tol,
            "range_violation_max": cfg.range_violation_tol,
            "conservation_relative_defect_max": cfg.conservation_rel_tol,
            "condition_proxy_max": cfg.condition_proxy_tol,
            "competitive_error_ratio_max": cfg.competitive_error_ratio,
            "ghg_empty_cell_fraction_over_halton_max": cfg.ghg_coverage_empty_ratio_tol,
        },
        "condition_threshold_rationale": (
            "2e6 is prospective and exceeds the largest viable P21-v2 PHS-RBF-FD "
            "screen proxy by more than 100x while excluding the dense-Gaussian pathology."
        ),
        "paper_I_stop_rule": (
            "If ROBIN_GHG_LEGACY_CONTROL fails a final production or untuned-holdout "
            "coupling/assembly gate, integrated GHG-conditioned domain decomposition "
            "is FAIL/DEFER and no new transmission family is added in Paper I."
        ),
    }
    contract["frozen_experiment_sha256"] = canonical_hash(contract)
    return contract


def verify_holdout_authorization(production_dir: Path, contract: Dict[str, Any]) -> None:
    contract_path = production_dir / "frozen_experiment_contract.json"
    marker_path = production_dir / "production_complete.json"
    if not contract_path.exists() or not marker_path.exists():
        raise RuntimeError(
            "Holdout is locked. Complete the production profile first; both the frozen "
            "contract and production completion marker are required."
        )
    stored = json.loads(contract_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = contract["frozen_experiment_sha256"]
    if stored.get("frozen_experiment_sha256") != expected:
        raise RuntimeError("Holdout contract mismatch: scientific settings changed after production.")
    if marker.get("frozen_experiment_sha256") != expected or marker.get("status") != "COMPLETED":
        raise RuntimeError("Production completion marker is absent, incomplete, or incompatible.")


def environment_record() -> Dict[str, Any]:
    import scipy
    import matplotlib
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "single_thread_env": {
            k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"
            )
        },
    }


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"cgf_static_interface_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def normalize01(a: np.ndarray, floor: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    if not np.isfinite(amin) or not np.isfinite(amax):
        raise ValueError("Indicator contains non-finite values.")
    if amax - amin < 1e-15:
        z = np.ones_like(a)
    else:
        z = (a - amin) / (amax - amin)
    if floor > 0:
        z = floor + (1.0 - floor) * z
    return z


def trapz1(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapezoid is newer; np.trapz remains broadly compatible.
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


# =============================================================================
# Exact benchmark S (reference used only for BCs and final evaluation)
# =============================================================================

def resistance_coordinate(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, float)
    xi = cfg.interface_x
    total = xi / cfg.k_left + (1.0 - xi) / cfg.k_right
    raw = np.where(
        x <= xi,
        x / cfg.k_left,
        xi / cfg.k_left + (x - xi) / cfg.k_right,
    )
    return raw / total


def exact_u(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    return resistance_coordinate(x, cfg) * np.sin(np.pi * np.asarray(y))


def exact_ux(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x)
    xi = cfg.interface_x
    total = xi / cfg.k_left + (1.0 - xi) / cfg.k_right
    invk = np.where(x <= xi, 1.0 / cfg.k_left, 1.0 / cfg.k_right)
    return invk / total * np.sin(np.pi * np.asarray(y))


def exact_uy(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    return resistance_coordinate(x, cfg) * np.pi * np.cos(np.pi * np.asarray(y))


def permeability(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    del y
    return np.where(np.asarray(x) <= cfg.interface_x, cfg.k_left, cfg.k_right)


def source_f(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    K = permeability(x, y, cfg)
    return K * np.pi**2 * resistance_coordinate(x, cfg) * np.sin(np.pi * np.asarray(y))


# =============================================================================
# Conservative structured pilot and coarse-grid operators
# =============================================================================

def harmonic(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b + 1e-300)


def build_structured_operator(
    nx: int,
    ny: int,
    cfg: Config,
    correction_bc_zero: bool = False,
) -> Dict[str, Any]:
    """Conservative 5-point diffusion operator with harmonic face coefficients."""
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    K = permeability(X, Y, cfg)
    F = source_f(X, Y, cfg)

    N = nx * ny
    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)
    boundary_mask = np.zeros(N, dtype=bool)

    def idx(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)
            is_boundary = i == 0 or i == nx - 1 or j == 0 or j == ny - 1
            if is_boundary:
                boundary_mask[p] = True
                A[p, p] = 1.0
                b[p] = 0.0 if correction_bc_zero else float(exact_u(xs[i], ys[j], cfg))
                continue

            kw = harmonic(K[i, j], K[i - 1, j])
            ke = harmonic(K[i, j], K[i + 1, j])
            ks = harmonic(K[i, j], K[i, j - 1])
            kn = harmonic(K[i, j], K[i, j + 1])

            aw, ae = kw / dx**2, ke / dx**2
            ass, an = ks / dy**2, kn / dy**2
            A[p, p] = aw + ae + ass + an
            A[p, idx(i - 1, j)] = -aw
            A[p, idx(i + 1, j)] = -ae
            A[p, idx(i, j - 1)] = -ass
            A[p, idx(i, j + 1)] = -an
            b[p] = float(F[i, j])

    return {
        "x": xs, "y": ys, "X": X, "Y": Y, "K": K, "F": F,
        "A": csr_matrix(A), "b": b, "boundary_mask": boundary_mask,
        "dx": dx, "dy": dy,
    }


def build_pilot(cfg: Config) -> Dict[str, Any]:
    t0 = time.perf_counter()
    op = build_structured_operator(cfg.pilot_nx, cfg.pilot_ny, cfg, correction_bc_zero=False)
    u = spsolve(op["A"], op["b"]).reshape(cfg.pilot_nx, cfg.pilot_ny)
    dx, dy = op["dx"], op["dy"]

    ux = np.gradient(u, dx, axis=0, edge_order=2)
    uy = np.gradient(u, dy, axis=1, edge_order=2)
    grad = np.hypot(ux, uy)

    logK = np.log10(np.maximum(op["K"], 1e-300))
    gx = np.gradient(logK, dx, axis=0, edge_order=1)
    gy = np.gradient(logK, dy, axis=1, edge_order=1)
    jump = np.hypot(gx, gy)
    positive = jump[jump > 0]
    threshold = np.percentile(positive, 50) if positive.size else np.inf
    interface_mask = jump >= threshold if np.isfinite(threshold) else np.zeros_like(jump, bool)
    dist = distance_transform_edt(~interface_mask) * min(dx, dy)

    collar_width = cfg.collar_factor * dx
    ell_g = max(cfg.green_decay_factor * collar_width, dx)
    G = np.exp(-dist / ell_g)
    R = gaussian_filter(grad, sigma=0.8)

    op.update({
        "u": u,
        "ux": ux,
        "uy": uy,
        "grad": grad,
        "jump": jump,
        "interface_mask": interface_mask.astype(float),
        "dist_interface": dist,
        "G": normalize01(G, cfg.density_floor),
        "R": normalize01(R, cfg.density_floor),
        "collar_width": collar_width,
        "pilot_time_s": time.perf_counter() - t0,
    })
    return op


def physical_fingerprint(pilot: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    K = pilot["K"]
    grad = np.maximum(pilot["grad"], 0)
    jump = np.maximum(pilot["jump"], 0)
    contrast = float(K.max() / K.min())
    logk = np.log10(np.maximum(K, 1e-300))
    gflat = grad.ravel()
    if gflat.sum() > 0:
        cutoff = np.percentile(gflat, 90)
        localization = float(gflat[gflat >= cutoff].sum() / gflat.sum())
    else:
        localization = 0.0
    positive_jump = jump[jump > 0]
    if positive_jump.size:
        jt = np.percentile(positive_jump, 50)
        jump_fraction = float(np.mean(jump >= jt))
    else:
        jump_fraction = 0.0

    # Pre-production heuristic only; final activation thresholds are frozen later
    # using E/S/F and applied prospectively to P.
    if contrast < 5 and localization < 0.35:
        recommendation = "GLOBAL_HALTON_REFERENCE"
    elif contrast < 50:
        recommendation = "GLOBAL_GHG_CANDIDATE"
    else:
        recommendation = "STATIC_INTERFACE_DDM_CANDIDATE"

    return {
        "contrast_ratio": contrast,
        "log10K_IQR": float(np.percentile(logk, 75) - np.percentile(logk, 25)),
        "gradient_localization_top10_mass": localization,
        "jump_area_fraction": jump_fraction,
        "collar_width": float(pilot["collar_width"]),
        "preproduction_architecture_recommendation": recommendation,
        "pilot_time_s": float(pilot["pilot_time_s"]),
    }


def grid_interpolator(pilot: Dict[str, Any], key: str) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (pilot["x"], pilot["y"]), pilot[key], bounds_error=False, fill_value=None
    )


# =============================================================================
# Low-discrepancy / GHG node construction
# =============================================================================

def halton_unit(n: int, seed: int, scramble: bool, skip: int = 1) -> np.ndarray:
    sampler = qmc.Halton(d=2, scramble=scramble, seed=seed if scramble else None)
    pts = sampler.random(n + skip)
    return pts[skip:]


def map_to_rect(base: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    p = np.array(base, dtype=float, copy=True)
    p[:, 0] = xmin + (xmax - xmin) * p[:, 0]
    return p


def systematic_weighted_pick(candidates: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    candidates = np.asarray(candidates, float)
    w = np.maximum(np.asarray(weights, float).ravel(), 0.0)
    if len(candidates) != len(w):
        raise ValueError("Candidate and weight lengths differ.")
    if w.sum() <= 0:
        idx = np.linspace(0, len(candidates) - 1, n, dtype=int)
        return candidates[idx]
    cdf = np.cumsum(w / w.sum())
    targets = (np.arange(n) + 0.5) / n
    idx = np.searchsorted(cdf, targets, side="left")
    idx = np.clip(idx, 0, len(candidates) - 1)

    # De-duplicate deterministically and replenish by descending weight.
    chosen: List[int] = []
    seen = set()
    for i in idx:
        ii = int(i)
        if ii not in seen:
            chosen.append(ii)
            seen.add(ii)
    if len(chosen) < n:
        for ii in np.argsort(-w):
            jj = int(ii)
            if jj not in seen:
                chosen.append(jj)
                seen.add(jj)
                if len(chosen) >= n:
                    break
    return candidates[np.array(chosen[:n], dtype=int)]


def unique_fill(points: np.ndarray, n: int, fill_pool: np.ndarray) -> np.ndarray:
    rounded = np.round(points, 12)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    out = points[np.sort(keep)]
    if len(out) >= n:
        return out[:n]
    existing = {tuple(v) for v in np.round(out, 12)}
    add = []
    for p in fill_pool:
        key = tuple(np.round(p, 12))
        if key not in existing:
            add.append(p)
            existing.add(key)
            if len(out) + len(add) == n:
                break
    if len(out) + len(add) < n:
        raise RuntimeError("Could not replenish node set after de-duplication.")
    return np.vstack([out, np.asarray(add)])[:n]


def allocation_counts(n: int, family: str, cfg: Config) -> Dict[str, int]:
    if family == "HALTON":
        return {"coverage_floor": n, "green": 0, "halton_remainder": 0, "gradient": 0}

    if family == "GHG_NO_COVERAGE_FLOOR":
        floor_n = 0
    elif family in (
        "GHG_PROTECTED",
        "GHG_NO_GREEN",
        "GHG_NO_HALTON_ENRICHMENT",
        "GHG_NO_GRADIENT",
    ):
        floor_n = int(round(cfg.coverage_floor_fraction * n))
    else:
        raise ValueError(f"Unknown allocation family {family}")

    rem = n - floor_n
    weights = np.array([cfg.alpha_green, cfg.beta_halton, cfg.gamma_gradient], float)
    if family == "GHG_NO_GREEN":
        weights[0] = 0.0
    elif family == "GHG_NO_HALTON_ENRICHMENT":
        # The protected low-discrepancy skeleton remains.  This ablation
        # removes only the additional Halton allocation from the remainder.
        weights[1] = 0.0
    elif family == "GHG_NO_GRADIENT":
        weights[2] = 0.0
    raw = rem * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    while counts.sum() < rem:
        counts[int(np.argmax(raw - counts))] += 1
    return {
        "coverage_floor": floor_n,
        "green": int(counts[0]),
        "halton_remainder": int(counts[1]),
        "gradient": int(counts[2]),
    }


def build_interior_nodes_rect(
    n: int,
    family: str,
    xmin: float,
    xmax: float,
    pilot: Dict[str, Any],
    seed: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Construct interior nodes in one physical subdomain.

    GHG_PROTECTED uses a hard low-discrepancy coverage floor FIRST.  The
    remainder is allocated by the nominal G/H/gradient mixture.  Thus the
    coverage floor is an admissibility safeguard, not a disguised retuning
    of alpha,beta,gamma.
    """
    counts = allocation_counts(n, family, cfg)
    if family == "HALTON":
        pts = map_to_rect(halton_unit(n, seed, cfg.scramble_halton, skip=2), xmin, xmax)
        return pts, np.array(["Halton"] * n, dtype=object), counts

    Gint = grid_interpolator(pilot, "G")
    Rint = grid_interpolator(pilot, "R")

    pieces = []
    labels = []
    floor_n = counts["coverage_floor"]
    if floor_n:
        p = map_to_rect(halton_unit(floor_n, seed, cfg.scramble_halton, skip=2), xmin, xmax)
        pieces.append(p)
        labels.extend(["CoverageFloor"] * len(p))

    # Remainder Halton uses a separated sequence segment.
    hrem = counts["halton_remainder"]
    if hrem:
        p = map_to_rect(halton_unit(hrem, seed + 1009, cfg.scramble_halton, skip=17), xmin, xmax)
        pieces.append(p)
        labels.extend(["HaltonRemainder"] * len(p))

    ng = counts["green"]
    if ng:
        cand = map_to_rect(
            halton_unit(cfg.candidate_multiplier * max(n, 8), seed + 2027, cfg.scramble_halton, skip=31),
            xmin, xmax
        )
        p = systematic_weighted_pick(cand, Gint(cand), ng)
        pieces.append(p)
        labels.extend(["Green"] * len(p))

    nr = counts["gradient"]
    if nr:
        cand = map_to_rect(
            halton_unit(cfg.candidate_multiplier * max(n, 8), seed + 3037, cfg.scramble_halton, skip=47),
            xmin, xmax
        )
        p = systematic_weighted_pick(cand, Rint(cand), nr)
        pieces.append(p)
        labels.extend(["Gradient"] * len(p))

    pts = np.vstack(pieces) if pieces else np.empty((0, 2))
    fill_pool = map_to_rect(
        halton_unit(6 * max(n, 8), seed + 4049, cfg.scramble_halton, skip=71), xmin, xmax
    )
    pts = unique_fill(pts, n, fill_pool)
    if len(labels) < n:
        labels.extend(["CoverageFill"] * (n - len(labels)))
    return pts[:n], np.array(labels[:n], dtype=object), counts


def boundary_interface_counts(budget: int, cfg: Config) -> Tuple[int, int]:
    root = math.sqrt(max(budget, 1))
    mb = max(cfg.min_boundary_each_edge, int(round(cfg.boundary_factor * root)))
    mi = max(cfg.min_interface, int(round(cfg.interface_factor * root)))
    return mb, mi


def outer_boundary(side: str, mb: int, cfg: Config) -> np.ndarray:
    xi = cfg.interface_x
    yv = np.linspace(0.0, 1.0, mb)
    if side == "L":
        vertical = np.column_stack([np.zeros(mb), yv])
        xt = np.linspace(0.0, xi, mb)
        # exclude x=0 corner (already vertical), include interface corner
        xb = xt[1:]
    elif side == "R":
        vertical = np.column_stack([np.ones(mb), yv])
        xt = np.linspace(xi, 1.0, mb)
        # include interface corner, exclude x=1 corner (already vertical)
        xb = xt[:-1]
    else:
        raise ValueError(side)
    bottom = np.column_stack([xb, np.zeros(len(xb))])
    top = np.column_stack([xb, np.ones(len(xb))])
    return np.vstack([vertical, bottom, top])


def interface_points(mi: int, cfg: Config) -> np.ndarray:
    # Exclude y=0,1 because those are physical Dirichlet-boundary corners.
    y = np.linspace(0.0, 1.0, mi + 2)[1:-1]
    return np.column_stack([np.full(mi, cfg.interface_x), y])


# =============================================================================
# PHS RBF-FD local differentiation
# =============================================================================

POLY_POWERS_DEG2 = [
    (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)
]


def poly_matrix(z: np.ndarray) -> np.ndarray:
    cols = []
    for px, py in POLY_POWERS_DEG2:
        cols.append((z[:, 0] ** px) * (z[:, 1] ** py))
    return np.column_stack(cols)


def poly_rhs(operator: str, h: float) -> np.ndarray:
    # At local evaluation coordinate z=(0,0).
    if operator == "lap":
        return np.array([0.0, 0.0, 0.0, 2.0 / h**2, 0.0, 2.0 / h**2])
    if operator == "dx":
        return np.array([0.0, 1.0 / h, 0.0, 0.0, 0.0, 0.0])
    if operator == "dy":
        return np.array([0.0, 0.0, 1.0 / h, 0.0, 0.0, 0.0])
    if operator == "value":
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    raise ValueError(operator)


def phs_rhs_at_center(zj: np.ndarray, operator: str, h: float) -> np.ndarray:
    r = np.sqrt(np.sum(zj * zj, axis=1))
    if operator == "lap":
        # In 2D: Δ r^m = m^2 r^(m-2); m=3 -> 9 r.
        return 9.0 * r / h**2
    if operator == "dx":
        # d/dx_eval r^3 at eval z=0: -3 r z_jx / h.
        return -3.0 * r * zj[:, 0] / h
    if operator == "dy":
        return -3.0 * r * zj[:, 1] / h
    if operator == "value":
        return r**3
    raise ValueError(operator)


def rbffd_weights(
    nodes: np.ndarray,
    center: np.ndarray,
    stencil_idx: np.ndarray,
    operator: str,
) -> np.ndarray:
    stencil = nodes[stencil_idx]
    dist = np.sqrt(np.sum((stencil - center) ** 2, axis=1))
    h = float(np.max(dist))
    if h <= 1e-14:
        raise RuntimeError("Degenerate RBF-FD stencil scale.")

    z = (stencil - center) / h
    pair = z[:, None, :] - z[None, :, :]
    rpair = np.sqrt(np.sum(pair * pair, axis=2))
    Phi = rpair**3
    P = poly_matrix(z)
    m = len(stencil_idx)
    q = P.shape[1]
    M = np.zeros((m + q, m + q), dtype=float)
    M[:m, :m] = Phi
    M[:m, m:] = P
    M[m:, :m] = P.T

    rhs = np.zeros(m + q, dtype=float)
    rhs[:m] = phs_rhs_at_center(z, operator, h)
    rhs[m:] = poly_rhs(operator, h)

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=1e-12)
    return sol[:m]


@dataclass
class Cloud:
    side: str
    family: str
    nodes: np.ndarray
    interior_idx: np.ndarray
    boundary_idx: np.ndarray
    interface_idx: np.ndarray
    labels: np.ndarray
    counts: Dict[str, int]
    tree: Any = None
    lap_stencils: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    dx_stencils: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    dy_stencils: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    build_time_s: float = 0.0


def build_cloud(
    side: str,
    family: str,
    n_interior: int,
    mb: int,
    mi: int,
    pilot: Dict[str, Any],
    seed: int,
    cfg: Config,
) -> Cloud:
    t0 = time.perf_counter()
    xmin, xmax = (0.0, cfg.interface_x) if side == "L" else (cfg.interface_x, 1.0)
    interior, ilabels, counts = build_interior_nodes_rect(
        n_interior, family, xmin, xmax, pilot, seed, cfg
    )
    bnd = outer_boundary(side, mb, cfg)
    iface = interface_points(mi, cfg)
    nodes = np.vstack([interior, bnd, iface])
    ni = len(interior)
    nb = len(bnd)
    ii = np.arange(0, ni, dtype=int)
    bi = np.arange(ni, ni + nb, dtype=int)
    gi = np.arange(ni + nb, ni + nb + len(iface), dtype=int)
    labels = np.concatenate([
        ilabels,
        np.array(["PhysicalBoundary"] * nb, dtype=object),
        np.array(["Interface"] * len(iface), dtype=object),
    ])

    cloud = Cloud(
        side=side, family=family, nodes=nodes,
        interior_idx=ii, boundary_idx=bi, interface_idx=gi,
        labels=labels, counts=counts,
    )
    cloud.tree = cKDTree(nodes)
    k = min(cfg.stencil_size, len(nodes))

    lap_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in ii:
        center = nodes[idx]
        _, sidx = cloud.tree.query(center, k=k)
        sidx = np.atleast_1d(sidx).astype(int)
        w = rbffd_weights(nodes, center, sidx, "lap")
        lap_list.append((sidx, w))

    dx_list: List[Tuple[np.ndarray, np.ndarray]] = []
    dy_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in gi:
        center = nodes[idx]
        _, sidx = cloud.tree.query(center, k=k)
        sidx = np.atleast_1d(sidx).astype(int)
        wx = rbffd_weights(nodes, center, sidx, "dx")
        wy = rbffd_weights(nodes, center, sidx, "dy")
        dx_list.append((sidx, wx))
        dy_list.append((sidx, wy))

    cloud.lap_stencils = lap_list
    cloud.dx_stencils = dx_list
    cloud.dy_stencils = dy_list
    cloud.build_time_s = time.perf_counter() - t0
    return cloud


def derivative_on_interface(cloud: Cloud, values: np.ndarray, which: str = "dx") -> np.ndarray:
    stencils = cloud.dx_stencils if which == "dx" else cloud.dy_stencils
    out = np.empty(len(cloud.interface_idx), dtype=float)
    assert stencils is not None
    for j, (idx, w) in enumerate(stencils):
        out[j] = float(np.dot(w, values[idx]))
    return out


def local_system_matrix(
    cloud: Cloud,
    cfg: Config,
    transmission: str,
    lambda_side: Optional[np.ndarray] = None,
) -> csr_matrix:
    n = len(cloud.nodes)
    A = lil_matrix((n, n), dtype=float)
    Kside = cfg.k_left if cloud.side == "L" else cfg.k_right
    normal_x = +1.0 if cloud.side == "L" else -1.0

    assert cloud.lap_stencils is not None
    for row, (idx, w) in zip(cloud.interior_idx, cloud.lap_stencils):
        A[row, idx] = -Kside * w

    for row in cloud.boundary_idx:
        A[row, row] = 1.0

    if transmission == "DIRICHLET":
        for row in cloud.interface_idx:
            A[row, row] = 1.0
    else:
        if lambda_side is None:
            raise ValueError("Robin system requires lambda_side.")
        assert cloud.dx_stencils is not None
        for j, (row, (idx, wdx)) in enumerate(zip(cloud.interface_idx, cloud.dx_stencils)):
            A[row, idx] += Kside * normal_x * wdx
            A[row, row] += float(lambda_side[j])
    return csr_matrix(A)


def local_rhs(
    cloud: Cloud,
    cfg: Config,
    interface_rhs: np.ndarray,
) -> np.ndarray:
    b = np.zeros(len(cloud.nodes), dtype=float)
    pint = cloud.nodes[cloud.interior_idx]
    b[cloud.interior_idx] = source_f(pint[:, 0], pint[:, 1], cfg)
    pb = cloud.nodes[cloud.boundary_idx]
    b[cloud.boundary_idx] = exact_u(pb[:, 0], pb[:, 1], cfg)
    b[cloud.interface_idx] = interface_rhs
    return b


def equilibrate_rows(A: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    """
    Left-scale each equation by the inverse Euclidean row norm.

    Strong-form meshfree systems mix PDE rows (O(h^-2)) with Dirichlet
    rows (O(1)) and Robin rows (O(h^-1)+lambda).  Row equilibration does
    not change the exact algebraic solution, but prevents that units/scale
    mismatch from masquerading as numerical ill-conditioning.
    """
    A = csr_matrix(A)
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    scale = 1.0 / np.maximum(norms, 1e-14)
    return csr_matrix(diags(scale) @ A), scale


def condition_proxy_dense(A: csr_matrix) -> float:
    # Screen systems are intentionally moderate.  Exact dense 2-norm condition
    # numbers are expensive but useful here; final large runs may switch to a
    # documented sparse estimator without changing the solve itself.
    n = A.shape[0]
    if n > 850:
        return float("nan")
    return float(np.linalg.cond(A.toarray()))


@dataclass
class FactoredLocalSystem:
    cloud: Cloud
    transmission: str
    lambda_side: Optional[np.ndarray]
    A: csr_matrix
    lu: Any
    condition_proxy: float
    row_scale: np.ndarray
    assembly_time_s: float


def factor_local_system(
    cloud: Cloud,
    transmission: str,
    lambda_side: Optional[np.ndarray],
    cfg: Config,
) -> FactoredLocalSystem:
    t0 = time.perf_counter()
    Araw = local_system_matrix(cloud, cfg, transmission, lambda_side)
    A, row_scale = equilibrate_rows(Araw)
    lu = splu(csc_matrix(A))
    cond = condition_proxy_dense(A)
    return FactoredLocalSystem(
        cloud=cloud, transmission=transmission, lambda_side=lambda_side,
        A=A, lu=lu, condition_proxy=cond, row_scale=row_scale,
        assembly_time_s=time.perf_counter() - t0,
    )


def solve_factored(system: FactoredLocalSystem, b: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    u = system.lu.solve(system.row_scale * b)
    return u, time.perf_counter() - t0


# =============================================================================
# Monolithic coupled static-interface solve (allocation comparison)
# =============================================================================

def build_monolithic_matrix(left: Cloud, right: Cloud, cfg: Config) -> csr_matrix:
    nL, nR = len(left.nodes), len(right.nodes)
    if len(left.interface_idx) != len(right.interface_idx):
        raise ValueError("Interface cardinalities differ.")
    mi = len(left.interface_idx)
    A = lil_matrix((nL + nR, nL + nR), dtype=float)

    # Interior and physical boundary equations for each side.
    for cloud, off, Kside in [(left, 0, cfg.k_left), (right, nL, cfg.k_right)]:
        assert cloud.lap_stencils is not None
        for row_local, (idx, w) in zip(cloud.interior_idx, cloud.lap_stencils):
            A[off + row_local, off + idx] = -Kside * w
        for row_local in cloud.boundary_idx:
            A[off + row_local, off + row_local] = 1.0

    # We intentionally use the local interface-row positions themselves as the
    # 2*mi coupled equations: left interface rows = continuity; right interface
    # rows = physical flux continuity.
    assert left.dx_stencils is not None and right.dx_stencils is not None
    for j in range(mi):
        row_cont = int(left.interface_idx[j])
        row_flux = nL + int(right.interface_idx[j])
        il = int(left.interface_idx[j])
        ir = int(right.interface_idx[j])
        A[row_cont, il] = 1.0
        A[row_cont, nL + ir] = -1.0

        idxL, wxL = left.dx_stencils[j]
        idxR, wxR = right.dx_stencils[j]
        A[row_flux, idxL] += cfg.k_left * wxL
        A[row_flux, nL + idxR] += -cfg.k_right * wxR

    return csr_matrix(A)


def build_monolithic_rhs(left: Cloud, right: Cloud, cfg: Config) -> np.ndarray:
    nL, nR = len(left.nodes), len(right.nodes)
    b = np.zeros(nL + nR, dtype=float)
    for cloud, off in [(left, 0), (right, nL)]:
        pint = cloud.nodes[cloud.interior_idx]
        b[off + cloud.interior_idx] = source_f(pint[:, 0], pint[:, 1], cfg)
        pb = cloud.nodes[cloud.boundary_idx]
        b[off + cloud.boundary_idx] = exact_u(pb[:, 0], pb[:, 1], cfg)
    # Interface continuity and flux RHS are zero.
    return b


def solve_monolithic(left: Cloud, right: Cloud, cfg: Config) -> Dict[str, Any]:
    t0 = time.perf_counter()
    Araw = build_monolithic_matrix(left, right, cfg)
    b = build_monolithic_rhs(left, right, cfg)
    A, row_scale = equilibrate_rows(Araw)
    b = row_scale * b
    assembly = time.perf_counter() - t0
    cond = condition_proxy_dense(A)
    t1 = time.perf_counter()
    u = spsolve(A, b)
    solve_time = time.perf_counter() - t1
    nL = len(left.nodes)
    return {
        "left_values": u[:nL],
        "right_values": u[nL:],
        "condition_proxy": cond,
        "assembly_time_s": assembly,
        "solve_time_s": solve_time,
        "A_shape": A.shape,
    }


# =============================================================================
# Field interpolation/evaluation and metrics
# =============================================================================

def piecewise_interpolated_grid(
    left: Cloud,
    uL: np.ndarray,
    right: Cloud,
    uR: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    cfg: Config,
    correction: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    EX, EY = np.meshgrid(ex, ey, indexing="ij")
    pts = np.column_stack([EX.ravel(), EY.ravel()])
    out = np.empty(len(pts), dtype=float)
    maskL = pts[:, 0] <= cfg.interface_x

    intL = LinearNDInterpolator(left.nodes, uL, fill_value=np.nan)
    intR = LinearNDInterpolator(right.nodes, uR, fill_value=np.nan)
    out[maskL] = intL(pts[maskL])
    out[~maskL] = intR(pts[~maskL])

    if np.any(~np.isfinite(out)):
        # A convex rectangular cloud should not need this, but explicit failure
        # is better than silently propagating NaN into paper metrics.
        bad = int(np.sum(~np.isfinite(out)))
        raise RuntimeError(f"LinearND interpolation produced {bad} non-finite evaluation values.")

    U = out.reshape(EX.shape)
    if correction is not None:
        cint = RegularGridInterpolator(
            (correction["x"], correction["y"]), correction["state"],
            bounds_error=False, fill_value=0.0
        )
        U = U + cint(pts).reshape(EX.shape)
    return U


def interface_metrics_from_values(
    left: Cloud,
    uL: np.ndarray,
    right: Cloud,
    uR: np.ndarray,
    cfg: Config,
    correction_iface: Optional[Dict[str, np.ndarray]] = None,
    lambda_left: Optional[np.ndarray] = None,
    lambda_right: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    IL = uL[left.interface_idx].copy()
    IR = uR[right.interface_idx].copy()
    uxL = derivative_on_interface(left, uL, "dx")
    uxR = derivative_on_interface(right, uR, "dx")
    if correction_iface is not None:
        e = correction_iface["u"]
        IL = IL + e
        IR = IR + e
        uxL = uxL + correction_iface["ux_left"]
        uxR = uxR + correction_iface["ux_right"]

    trace = IL - IR
    flux = cfg.k_left * uxL - cfg.k_right * uxR
    out = {
        "trace_rmse": float(np.sqrt(np.mean(trace**2))),
        "flux_rmse": float(np.sqrt(np.mean(flux**2))),
    }
    out["physical_mismatch"] = float(math.hypot(out["trace_rmse"], out["flux_rmse"]))

    if lambda_left is not None and lambda_right is not None:
        # Receiving-side Robin operators are audited separately then combined.
        rL = cfg.k_left * uxL + lambda_left * IL - (cfg.k_right * uxR + lambda_left * IR)
        rR = -cfg.k_right * uxR + lambda_right * IR - (-cfg.k_left * uxL + lambda_right * IL)
        out["robin_left_rmse"] = float(np.sqrt(np.mean(rL**2)))
        out["robin_right_rmse"] = float(np.sqrt(np.mean(rR**2)))
        out["robin_rmse"] = float(max(out["robin_left_rmse"], out["robin_right_rmse"]))
    else:
        out["robin_left_rmse"] = np.nan
        out["robin_right_rmse"] = np.nan
        out["robin_rmse"] = np.nan
    return out


def grid_field_metrics(
    U: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    cfg: Config,
    collar_width: float,
) -> Dict[str, float]:
    EX, EY = np.meshgrid(ex, ey, indexing="ij")
    Ue = exact_u(EX, EY, cfg)
    err = U - Ue
    dx = ex[1] - ex[0]
    dy = ey[1] - ey[0]
    ux = np.gradient(U, dx, axis=0, edge_order=2)
    uy = np.gradient(U, dy, axis=1, edge_order=2)
    uxe = exact_ux(EX, EY, cfg)
    uye = exact_uy(EX, EY, cfg)

    collar = np.abs(EX - cfg.interface_x) <= collar_width
    bulk = ~collar

    def rmse(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))

    # Conservation: integral f = - boundary integral K grad(u).n.
    K = permeability(EX, EY, cfg)
    src_int = trapz1(np.array([trapz1(source_f(ex, y, cfg), ex) for y in ey]), ey)
    # boundary outward flux K grad u . n
    left_flux = trapz1(-cfg.k_left * ux[0, :], ey)
    right_flux = trapz1(cfg.k_right * ux[-1, :], ey)
    bottom_flux = trapz1(-K[:, 0] * uy[:, 0], ex)
    top_flux = trapz1(K[:, -1] * uy[:, -1], ex)
    boundary_flux = left_flux + right_flux + bottom_flux + top_flux
    conservation_defect = abs((-boundary_flux) - src_int) / max(abs(src_int), 1e-12)

    umin, umax = float(np.min(U)), float(np.max(U))
    range_violation = max(0.0, -umin) + max(0.0, umax - 1.0)

    return {
        "full_rmse": rmse(err),
        "full_linf": float(np.max(np.abs(err))),
        "interface_collar_rmse": rmse(err[collar]),
        "bulk_rmse": rmse(err[bulk]),
        "gradient_rmse": rmse(np.hypot(ux - uxe, uy - uye)),
        "range_min": umin,
        "range_max": umax,
        "range_violation": float(range_violation),
        "conservation_rel_defect": float(conservation_defect),
        "source_integral": float(src_int),
        "boundary_flux_integral": float(boundary_flux),
    }


def node_metrics(points: np.ndarray, cfg: Config, collar_width: float) -> Dict[str, float]:
    g = 16
    ix = np.clip((points[:, 0] * g).astype(int), 0, g - 1)
    iy = np.clip((points[:, 1] * g).astype(int), 0, g - 1)
    occ = np.zeros((g, g), dtype=int)
    for a, b in zip(ix, iy):
        occ[a, b] += 1
    return {
        "n": int(len(points)),
        "collar_fraction": float(np.mean(np.abs(points[:, 0] - cfg.interface_x) <= collar_width)),
        "empty_cell_fraction_16x16": float(np.mean(occ == 0)),
    }


def local_residual_metrics(cloud: Cloud, values: np.ndarray, cfg: Config) -> Dict[str, float]:
    Kside = cfg.k_left if cloud.side == "L" else cfg.k_right
    residual = []
    assert cloud.lap_stencils is not None
    for row_idx, (idx, w) in zip(cloud.interior_idx, cloud.lap_stencils):
        p = cloud.nodes[row_idx]
        lhs = -Kside * float(np.dot(w, values[idx]))
        rhs = float(source_f(p[0], p[1], cfg))
        residual.append(lhs - rhs)
    residual = np.asarray(residual)
    return {
        "local_pde_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "local_value_min": float(values.min()),
        "local_value_max": float(values.max()),
        "local_range_violation": float(max(0.0, -values.min()) + max(0.0, values.max() - 1.0)),
    }


# =============================================================================
# Coarse correction
# =============================================================================

@dataclass
class CoarseCorrection:
    op: Dict[str, Any]
    lu: Any
    state: np.ndarray


def build_coarse_correction(cfg: Config) -> CoarseCorrection:
    op = build_structured_operator(cfg.coarse_nx, cfg.coarse_ny, cfg, correction_bc_zero=True)
    lu = splu(csc_matrix(op["A"]))
    state = np.zeros((cfg.coarse_nx, cfg.coarse_ny), dtype=float)
    return CoarseCorrection(op=op, lu=lu, state=state)


def piecewise_values_at_points(
    left: Cloud, uL: np.ndarray, right: Cloud, uR: np.ndarray,
    points: np.ndarray, cfg: Config
) -> np.ndarray:
    out = np.empty(len(points), float)
    maskL = points[:, 0] <= cfg.interface_x
    intL = LinearNDInterpolator(left.nodes, uL, fill_value=np.nan)
    intR = LinearNDInterpolator(right.nodes, uR, fill_value=np.nan)
    out[maskL] = intL(points[maskL])
    out[~maskL] = intR(points[~maskL])
    if np.any(~np.isfinite(out)):
        raise RuntimeError("Non-finite coarse-grid interpolation from local states.")
    return out


def update_coarse_correction(
    coarse: CoarseCorrection,
    left: Cloud,
    uL: np.ndarray,
    right: Cloud,
    uR: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    op = coarse.op
    pts = np.column_stack([op["X"].ravel(), op["Y"].ravel()])
    local_vec = piecewise_values_at_points(left, uL, right, uR, pts, cfg)
    # Enforce the actual external Dirichlet state exactly before residual.
    bm = op["boundary_mask"]
    local_vec[bm] = exact_u(pts[bm, 0], pts[bm, 1], cfg)

    # Residual of the CURRENT assembled state (local + existing coarse
    # correction), followed by a standard residual-correction increment.
    # This is an additive coarse correction, not a second copy of the full
    # solution.  Solving repeatedly for the full correction while retaining
    # an old correction can double-count low-frequency error.
    actual_op = build_structured_operator(cfg.coarse_nx, cfg.coarse_ny, cfg, correction_bc_zero=False)
    total_before = local_vec + coarse.state.ravel()
    r = actual_op["b"] - actual_op["A"] @ total_before
    r[op["boundary_mask"]] = 0.0
    pre = float(np.linalg.norm(r) / math.sqrt(len(r)))
    e_inc = coarse.lu.solve(r).reshape(cfg.coarse_nx, cfg.coarse_ny)
    coarse.state = coarse.state + cfg.coarse_relaxation * e_inc

    # Post-correction residual on the coarse grid.
    total = local_vec + coarse.state.ravel()
    rpost = actual_op["b"] - actual_op["A"] @ total
    rpost[op["boundary_mask"]] = 0.0
    post = float(np.linalg.norm(rpost) / math.sqrt(len(rpost)))

    return {
        "pre_residual_rmse": pre,
        "post_residual_rmse": post,
        "correction_l2": float(np.linalg.norm(coarse.state) / math.sqrt(coarse.state.size)),
    }


def coarse_interface_data(coarse: CoarseCorrection, iface: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """Conservative coarse trace and interface-flux representation.

    The coarse grid is chosen with an even number of x nodes so x=xi lies
    on the face between the last LEFT node and first RIGHT node.  The
    finite-volume operator already uses one harmonic face coefficient there;
    hence it has ONE conservative face flux.  For interface auditing we map
    that single face flux to equivalent one-sided derivatives q/K_L and
    q/K_R.  This preserves the discrete conservation property instead of
    differentiating a coarse piecewise field across a coefficient jump.

    The correction trace is the midpoint average of the two adjacent nodal
    values and is common to both subdomains.  The coarse correction is not
    fed into Robin transmission; it enters only the additive assembly audit.
    """
    op = coarse.op
    x, y = op["x"], op["y"]
    state = coarse.state
    xi = float(cfg.interface_x)

    left_candidates = np.where(x < xi)[0]
    right_candidates = np.where(x > xi)[0]
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise RuntimeError("Coarse grid does not bracket the declared interface.")
    iL = int(left_candidates[-1])
    iR = int(right_candidates[0])
    face_x = 0.5 * (x[iL] + x[iR])
    if abs(face_x - xi) > 0.51 * op["dx"]:
        raise RuntimeError("Coarse interface face is not aligned with declared interface.")

    tr_grid = 0.5 * (state[iL, :] + state[iR, :])
    kface = harmonic(cfg.k_left, cfg.k_right)
    q_grid = kface * (state[iR, :] - state[iL, :]) / (x[iR] - x[iL])

    u_iface = np.interp(iface[:, 1], y, tr_grid)
    q_iface = np.interp(iface[:, 1], y, q_grid)
    uxL = q_iface / cfg.k_left
    uxR = q_iface / cfg.k_right
    return {"u": u_iface, "ux_left": uxL, "ux_right": uxR, "coarse_face_flux": q_iface}

def correction_export_dict(coarse: CoarseCorrection) -> Dict[str, Any]:
    return {"x": coarse.op["x"], "y": coarse.op["y"], "state": coarse.state.copy()}


# =============================================================================
# Transmission library
# =============================================================================

def coth(x: float) -> float:
    return math.cosh(x) / math.sinh(x)


def transmission_lambdas(
    name: str,
    left: Cloud,
    right: Cloud,
    pilot: Dict[str, Any],
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    mi = len(left.interface_idx)
    if name == "DIRICHLET":
        return None, None, {"lambda_left_mean": np.nan, "lambda_right_mean": np.nan}

    if name == "ROBIN_SCALAR":
        lL = np.full(mi, cfg.robin_scalar_lambda)
        lR = np.full(mi, cfg.robin_scalar_lambda)

    elif name == "ROBIN_COEFFICIENT_SCALED":
        areaL = cfg.interface_x * 1.0
        areaR = (1.0 - cfg.interface_x) * 1.0
        hL = math.sqrt(areaL / max(len(left.interior_idx), 1))
        hR = math.sqrt(areaR / max(len(right.interior_idx), 1))
        lL = np.full(mi, cfg.k_left / hL)
        lR = np.full(mi, cfg.k_right / hR)

    elif name == "ROBIN_DTN_ASYMMETRIC":
        # Transparent / DtN-motivated receiving impedances.  For the
        # dominant tangential sin(pi y) error mode, the neighboring
        # finite subdomain has DtN scale K*pi*coth(pi*L).  The impedance
        # imposed on the LEFT solve therefore approximates the RIGHT
        # neighbor DtN, and vice versa.  This cross-side assignment is
        # deliberate; using each subdomain's own DtN scale is not the
        # transparent Robin map for the receiving solve.
        LL = cfg.interface_x
        LR = 1.0 - cfg.interface_x
        dtn_left = cfg.k_left * np.pi * coth(np.pi * LL)
        dtn_right = cfg.k_right * np.pi * coth(np.pi * LR)
        lL = np.full(mi, dtn_right)
        lR = np.full(mi, dtn_left)

    elif name == "ROBIN_GHG_LEGACY_CONTROL":
        iface = left.nodes[left.interface_idx]
        Gint = grid_interpolator(pilot, "G")
        Rint = grid_interpolator(pilot, "R")
        g = normalize01(Gint(iface))
        r = normalize01(Rint(iface))
        modifier = 1.0 + cfg.robin_ghg_strength * (
            cfg.alpha_green * g + cfg.gamma_gradient * r
        )
        lL = cfg.robin_scalar_lambda * modifier
        lR = cfg.robin_scalar_lambda * modifier

    else:
        raise ValueError(name)

    meta = {
        "lambda_left_mean": float(np.mean(lL)),
        "lambda_right_mean": float(np.mean(lR)),
        "lambda_left_min": float(np.min(lL)),
        "lambda_left_max": float(np.max(lL)),
        "lambda_right_min": float(np.min(lR)),
        "lambda_right_max": float(np.max(lR)),
    }
    return lL, lR, meta


def pilot_interface_initialization(
    iface: np.ndarray,
    pilot: Dict[str, Any],
    cfg: Config,
) -> Dict[str, np.ndarray]:
    """Non-oracle interface trace and one-sided derivatives from the pilot.

    A discontinuous coefficient makes the finite-volume node sitting exactly
    on the jump a poor place from which to form a naive forward derivative:
    the first face uses a harmonic coefficient and can mix the two material
    scales.  We therefore fit a small quadratic separately to grid points
    strictly inside each material and extrapolate both the trace and d/dx to
    x=xi.  No exact interior solution is used.

    This is intentionally simple and deterministic; it is a pilot adapter,
    not an extra tuned solver.
    """
    x, y, u = pilot["x"], pilot["y"], pilot["u"]
    i = int(np.argmin(np.abs(x - cfg.interface_x)))
    xi = float(cfg.interface_x)

    if i < 3 or i + 3 >= len(x):
        raise RuntimeError("Pilot grid too small for one-sided quadratic interface fit.")

    left_ids = np.arange(i - 3, i)
    right_ids = np.arange(i + 1, i + 4)

    def fit_side(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fit u(x,y_j)=a x^2+b x+c independently for every y_j.
        V = np.column_stack([x[ids] ** 2, x[ids], np.ones(len(ids))])
        coeff, *_ = np.linalg.lstsq(V, u[ids, :], rcond=None)
        trace = coeff[0, :] * xi**2 + coeff[1, :] * xi + coeff[2, :]
        deriv = 2.0 * coeff[0, :] * xi + coeff[1, :]
        return trace, deriv

    trL_grid, uxL_grid = fit_side(left_ids)
    trR_grid, uxR_grid = fit_side(right_ids)
    # Average the two extrapolated traces; retain derivatives separately.
    tr_grid = 0.5 * (trL_grid + trR_grid)

    u0 = np.interp(iface[:, 1], y, tr_grid)
    uxL = np.interp(iface[:, 1], y, uxL_grid)
    uxR = np.interp(iface[:, 1], y, uxR_grid)
    return {"u": u0, "ux_left": uxL, "ux_right": uxR}


# =============================================================================
# Schwarz with evolving coarse correction
# =============================================================================

def run_schwarz_branch(
    allocation_family: str,
    transmission: str,
    left: Cloud,
    right: Cloud,
    pilot: Dict[str, Any],
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    iface = left.nodes[left.interface_idx]
    if not np.allclose(iface, right.nodes[right.interface_idx]):
        raise RuntimeError("Left/right interface point sets are not identical.")

    lamL, lamR, lam_meta = transmission_lambdas(transmission, left, right, pilot, cfg)
    sysL = factor_local_system(left, transmission, lamL, cfg)
    sysR = factor_local_system(right, transmission, lamR, cfg)
    coarse = build_coarse_correction(cfg)
    init = pilot_interface_initialization(iface, pilot, cfg)

    if transmission == "DIRICHLET":
        rhsL = init["u"].copy()
        rhsR = init["u"].copy()
    else:
        assert lamL is not None and lamR is not None
        rhsL = cfg.k_right * init["ux_right"] + lamL * init["u"]
        rhsR = -cfg.k_left * init["ux_left"] + lamR * init["u"]

    hist_rows = []
    prev_totalL = prev_totalR = None
    total_solve_time = 0.0
    uL = uR = None
    t_branch = time.perf_counter()

    for it in range(1, cfg.max_schwarz_iter + 1):
        bL = local_rhs(left, cfg, rhsL)
        bR = local_rhs(right, cfg, rhsR)
        uL, stL = solve_factored(sysL, bL)
        uR, stR = solve_factored(sysR, bR)
        total_solve_time += stL + stR

        coarse_diag = update_coarse_correction(coarse, left, uL, right, uR, cfg)
        c_if = coarse_interface_data(coarse, iface, cfg)

        # Gate 3 (physical coupling) is measured on the states actually
        # exchanged by Schwarz.  The additive coarse correction is audited
        # separately as part of Gate 4 assembly; it is deliberately NOT fed
        # back into the Robin target.  This cleanly separates transmission
        # convergence from low-frequency coarse repair and avoids double
        # counting a global correction in the interface iteration.
        intm = interface_metrics_from_values(
            left, uL, right, uR, cfg, correction_iface=None,
            lambda_left=lamL, lambda_right=lamR
        )
        intm_total = interface_metrics_from_values(
            left, uL, right, uR, cfg, correction_iface=c_if,
            lambda_left=lamL, lambda_right=lamR
        )

        uL_if = uL[left.interface_idx]
        uR_if = uR[right.interface_idx]
        uxL_if = derivative_on_interface(left, uL, "dx")
        uxR_if = derivative_on_interface(right, uR, "dx")

        if prev_totalL is None:
            update_norm = float("inf")
        else:
            update_norm = float(math.sqrt(
                0.5 * np.mean((uL_if - prev_totalL) ** 2)
                + 0.5 * np.mean((uR_if - prev_totalR) ** 2)
            ))

        localL = local_residual_metrics(left, uL, cfg)
        localR = local_residual_metrics(right, uR, cfg)
        hist_rows.append({
            "allocation_family": allocation_family,
            "transmission": transmission,
            "iteration": it,
            **intm,
            "assembled_trace_rmse": intm_total["trace_rmse"],
            "assembled_flux_rmse": intm_total["flux_rmse"],
            "assembled_physical_mismatch": intm_total["physical_mismatch"],
            "assembled_robin_rmse": intm_total["robin_rmse"],
            "update_norm": update_norm,
            "coarse_pre_residual_rmse": coarse_diag["pre_residual_rmse"],
            "coarse_post_residual_rmse": coarse_diag["post_residual_rmse"],
            "coarse_correction_l2": coarse_diag["correction_l2"],
            "local_pde_residual_rmse_left": localL["local_pde_residual_rmse"],
            "local_pde_residual_rmse_right": localR["local_pde_residual_rmse"],
            "local_range_violation_left": localL["local_range_violation"],
            "local_range_violation_right": localR["local_range_violation"],
            "coarse_state_id": short_hash(np.round(coarse.state, 13).tolist()),
        })

        # Two-way update includes the evolving coarse-corrected neighbor state.
        if transmission == "DIRICHLET":
            targetL = uR_if
            targetR = uL_if
        else:
            assert lamL is not None and lamR is not None
            targetL = cfg.k_right * uxR_if + lamL * uR_if
            targetR = -cfg.k_left * uxL_if + lamR * uL_if

        omega = cfg.schwarz_relaxation
        rhsL = (1.0 - omega) * rhsL + omega * targetL
        rhsR = (1.0 - omega) * rhsR + omega * targetR
        prev_totalL = uL_if.copy()
        prev_totalR = uR_if.copy()

        if it >= 3 and intm["physical_mismatch"] < cfg.ddm_tol:
            break

    if uL is None or uR is None:
        raise RuntimeError("Schwarz loop emitted no state.")

    hist = pd.DataFrame(hist_rows)
    vals = hist["physical_mismatch"].to_numpy(float)
    ratios = vals[1:] / np.maximum(vals[:-1], 1e-300)
    ratios = ratios[np.isfinite(ratios)]
    q_tail = float(np.median(ratios[-min(5, len(ratios)):])) if len(ratios) else np.nan
    if len(vals) >= 2 and vals[0] > 0 and vals[-1] > 0:
        q_net = float((vals[-1] / vals[0]) ** (1.0 / (len(vals) - 1)))
    else:
        q_net = np.nan

    result = {
        "left_values": uL,
        "right_values": uR,
        "coarse": correction_export_dict(coarse),
        "left": left,
        "right": right,
        "lambda_left": lamL,
        "lambda_right": lamR,
    }
    summary = {
        "allocation_family": allocation_family,
        "transmission": transmission,
        "iterations": int(len(hist)),
        "final_trace_rmse": float(hist.iloc[-1]["trace_rmse"]),
        "final_flux_rmse": float(hist.iloc[-1]["flux_rmse"]),
        "final_robin_rmse": float(hist.iloc[-1]["robin_rmse"]) if np.isfinite(hist.iloc[-1]["robin_rmse"]) else np.nan,
        "final_physical_mismatch": float(hist.iloc[-1]["physical_mismatch"]),
        "q_tail_physical": q_tail,
        "q_net_physical": q_net,
        "condition_max": float(np.nanmax([sysL.condition_proxy, sysR.condition_proxy])),
        "matrix_assembly_time_s": float(sysL.assembly_time_s + sysR.assembly_time_s),
        "local_solve_time_s": float(total_solve_time),
        "branch_total_time_s": float(time.perf_counter() - t_branch),
        "explicit_evolving_coarse_correction": True,
        "coarse_correction_mode": "additive_global_residual_correction_no_interface_feedback",
        "final_assembled_physical_mismatch": float(hist.iloc[-1]["assembled_physical_mismatch"]),
        **lam_meta,
    }
    logger.info(
        "%s / %s -> q_net=%.4g q_tail=%.4g Rphys=%.4g",
        allocation_family, transmission,
        summary["q_net_physical"], summary["q_tail_physical"], summary["final_physical_mismatch"]
    )
    return result, hist, summary


# =============================================================================
# Gate adjudication
# =============================================================================

def allocation_gate_rows(node_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    h = node_df[node_df["allocation_family"] == "HALTON"].iloc[0]
    rows = []
    for _, r in node_df.iterrows():
        if r["allocation_family"] == "HALTON":
            passed = True
            reason = "reference"
        else:
            limit = cfg.ghg_coverage_empty_ratio_tol * h["empty_cell_fraction_16x16"] + 1e-12
            passed = bool(r["empty_cell_fraction_16x16"] <= limit)
            reason = f"empty_fraction <= {cfg.ghg_coverage_empty_ratio_tol:.2f}x Halton"
        rows.append({
            "allocation_family": r["allocation_family"],
            "gate": "G1_ALLOCATION_COVERAGE",
            "passed": passed,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def ddm_gate_rows(
    ddm_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
    node_df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    # Competitive error reference: Halton monolithic coupled solve.
    href = allocation_df[allocation_df["allocation_family"] == "HALTON"].iloc[0]
    allocation_pass = {
        str(r.allocation_family): bool(r.passed)
        for r in allocation_gate_rows(node_df, cfg).itertuples()
    }
    rows = []
    for _, r in ddm_df.iterrows():
        alloc_pass = allocation_pass.get(str(r["allocation_family"]), False)
        local_pass = bool(
            r["condition_max"] <= cfg.condition_proxy_tol
            and r["range_violation"] <= cfg.range_violation_tol
        )
        coupling_pass = bool(
            np.isfinite(r["q_net_physical"])
            and np.isfinite(r["q_tail_physical"])
            and np.isfinite(r["final_physical_mismatch"])
            and r["q_net_physical"] < cfg.q_margin
            and r["q_tail_physical"] < cfg.q_margin
            and r["final_physical_mismatch"] <= cfg.ddm_tol
        )
        assembly_pass = bool(
            r["conservation_rel_defect"] <= cfg.conservation_rel_tol
            and r["full_rmse"] <= cfg.competitive_error_ratio * href["full_rmse"]
            and r["interface_collar_rmse"] <= cfg.competitive_error_ratio * href["interface_collar_rmse"]
        )
        for gate, passed, reason in [
            ("G1_ALLOCATION", alloc_pass, "declared node budget and inherited coverage gate"),
            ("G2_LOCAL_APPROXIMATION", local_pass, "condition and range admissibility"),
            (
                "G3_PHYSICAL_COUPLING",
                coupling_pass,
                "q_net and q_tail below 0.95; terminal physical mismatch <= 5e-4",
            ),
            ("G4_ASSEMBLY", assembly_pass, "accuracy and conservation admissibility"),
        ]:
            rows.append({
                "allocation_family": r["allocation_family"],
                "transmission": r["transmission"],
                "gate": gate,
                "passed": bool(passed),
                "reason": reason,
            })
        rows.append({
            "allocation_family": r["allocation_family"],
            "transmission": r["transmission"],
            "gate": "ALL_FOUR_FINAL_GATES",
            "passed": bool(alloc_pass and local_pass and coupling_pass and assembly_pass),
            "reason": "frozen production/holdout adjudication",
        })
    return pd.DataFrame(rows)


# =============================================================================
# One seed/budget run
# =============================================================================

def run_one(seed: int, budget: int, cfg: Config, campaign_dir: Path) -> Dict[str, Any]:
    run_dir = campaign_dir / f"seed_{seed:04d}_budget_{budget:04d}"
    ensure_dir(run_dir)
    status_path = run_dir / "status.json"

    if status_path.exists() and not cfg.force:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "COMPLETED":
                return {"status": "SKIPPED_COMPLETED", "seed": seed, "budget": budget, "run_dir": str(run_dir)}
        except Exception:
            pass

    logger = setup_logger(run_dir)
    started = time.time()
    status = {
        "status": "RUNNING", "seed": seed, "budget": budget,
        "started_unix": started,
    }
    json_dump(status_path, status)

    run_cfg = asdict(cfg)
    run_cfg.update({"seed": seed, "budget": budget, "transmission_library": TRANSMISSION_LIBRARY})
    run_cfg["frozen_experiment_sha256"] = frozen_experiment_contract(cfg)["frozen_experiment_sha256"]
    provenance_root = short_hash(run_cfg)
    run_cfg["provenance_root"] = provenance_root
    json_dump(run_dir / "run_config.json", run_cfg)

    try:
        logger.info("=" * 76)
        logger.info("CGF static-interface %s | seed=%d budget=%d", cfg.profile, seed, budget)
        logger.info("Manuscript targets: G1-G3, R1, D1-D4, R4, F1, X1")
        logger.info("Frozen transmission library: %s", ", ".join(TRANSMISSION_LIBRARY))

        # Pilot and fingerprint.
        pilot = build_pilot(cfg)
        fingerprint = physical_fingerprint(pilot, cfg)
        fingerprint.update({"seed": seed, "budget": budget, "provenance_root": provenance_root})
        pd.DataFrame([fingerprint]).to_csv(run_dir / "fingerprint.csv", index=False)

        # Budget means TOTAL interior nodes across both physical subdomains.
        nL = budget // 2
        nR = budget - nL
        mb, mi = boundary_interface_counts(budget, cfg)
        logger.info("Interior split L/R=%d/%d, boundary m=%d, interface m=%d", nL, nR, mb, mi)

        clouds: Dict[Tuple[str, str], Cloud] = {}
        node_rows = []
        for family in ALLOCATION_FAMILIES:
            left = build_cloud("L", family, nL, mb, mi, pilot, seed, cfg)
            right = build_cloud("R", family, nR, mb, mi, pilot, seed + 17, cfg)
            clouds[(family, "L")] = left
            clouds[(family, "R")] = right
            interior_all = np.vstack([left.nodes[left.interior_idx], right.nodes[right.interior_idx]])
            nm = node_metrics(interior_all, cfg, float(pilot["collar_width"]))
            nm.update({
                "seed": seed, "budget": budget, "allocation_family": family,
                "left_cloud_build_time_s": left.build_time_s,
                "right_cloud_build_time_s": right.build_time_s,
                "coverage_floor_fraction": (
                    0.0 if family in ("HALTON", "GHG_NO_COVERAGE_FLOOR")
                    else cfg.coverage_floor_fraction
                ),
                "provenance_key": short_hash({"root": provenance_root, "family": family, "kind": "nodes"}),
            })
            node_rows.append(nm)

            # Exact node export with role labels.
            for side, cloud in [("L", left), ("R", right)]:
                ndf = pd.DataFrame(cloud.nodes, columns=["x", "y"])
                ndf["role"] = cloud.labels
                ndf["side"] = side
                ndf["allocation_family"] = family
                ndf.to_csv(run_dir / f"nodes_{family}_{side}.csv", index=False)

        node_df = pd.DataFrame(node_rows)
        node_df.to_csv(run_dir / "node_metrics.csv", index=False)

        # Evaluation grid common to all methods.
        ex = np.linspace(0.0, 1.0, cfg.eval_nx)
        ey = np.linspace(0.0, 1.0, cfg.eval_ny)
        EX, EY = np.meshgrid(ex, ey, indexing="ij")

        # Monolithic coupled interface solve isolates ALLOCATION behavior.
        allocation_rows = []
        allocation_states: Dict[str, np.ndarray] = {}
        for family in ALLOCATION_FAMILIES:
            logger.info("Monolithic coupled RBF-FD solve: %s", family)
            left, right = clouds[(family, "L")], clouds[(family, "R")]
            tgen = left.build_time_s + right.build_time_s + float(pilot["pilot_time_s"])
            sol = solve_monolithic(left, right, cfg)
            U = piecewise_interpolated_grid(
                left, sol["left_values"], right, sol["right_values"], ex, ey, cfg
            )
            metrics = grid_field_metrics(U, ex, ey, cfg, float(pilot["collar_width"]))
            intm = interface_metrics_from_values(
                left, sol["left_values"], right, sol["right_values"], cfg
            )
            row = {
                "seed": seed, "budget": budget, "allocation_family": family,
                **metrics, **{f"interface_{k}": v for k, v in intm.items()},
                "condition_proxy": sol["condition_proxy"],
                "generation_time_s": tgen,
                "assembly_time_s": sol["assembly_time_s"],
                "solve_time_s": sol["solve_time_s"],
                "total_time_s": tgen + sol["assembly_time_s"] + sol["solve_time_s"],
                "provenance_key": short_hash({"root": provenance_root, "family": family, "kind": "allocation"}),
            }
            allocation_rows.append(row)
            allocation_states[family] = U
            logger.info(
                "%s -> RMSE=%.4g interface=%.4g cond=%.4g range=%.4g",
                family, row["full_rmse"], row["interface_collar_rmse"],
                row["condition_proxy"], row["range_violation"]
            )

        allocation_df = pd.DataFrame(allocation_rows)
        allocation_df.to_csv(run_dir / "allocation_metrics.csv", index=False)

        # DDM: full 2 allocation x 5 transmission factorial.  This prevents
        # allocation/transmission confounding and gives equal opportunities.
        ddm_rows = []
        histories = []
        ddm_states: Dict[str, np.ndarray] = {}
        coarse_states: Dict[str, np.ndarray] = {}
        for family in DDM_ALLOCATION_FAMILIES:
            left, right = clouds[(family, "L")], clouds[(family, "R")]
            for transmission in TRANSMISSION_LIBRARY:
                logger.info("DDM branch: %s x %s", family, transmission)
                result, hist, summary = run_schwarz_branch(
                    family, transmission, left, right, pilot, cfg, logger
                )
                U = piecewise_interpolated_grid(
                    left, result["left_values"], right, result["right_values"],
                    ex, ey, cfg, correction=result["coarse"]
                )
                metrics = grid_field_metrics(U, ex, ey, cfg, float(pilot["collar_width"]))
                localL = local_residual_metrics(left, result["left_values"], cfg)
                localR = local_residual_metrics(right, result["right_values"], cfg)
                row = {
                    "seed": seed, "budget": budget,
                    **summary, **metrics,
                    "local_pde_residual_rmse_max": max(localL["local_pde_residual_rmse"], localR["local_pde_residual_rmse"]),
                    "local_range_violation_max": max(localL["local_range_violation"], localR["local_range_violation"]),
                    "generation_time_s": left.build_time_s + right.build_time_s + float(pilot["pilot_time_s"]),
                    "total_time_s": left.build_time_s + right.build_time_s + float(pilot["pilot_time_s"]) + summary["branch_total_time_s"],
                    "provenance_key": short_hash({
                        "root": provenance_root, "family": family,
                        "transmission": transmission, "kind": "ddm"
                    }),
                }
                ddm_rows.append(row)
                hist = hist.copy()
                hist.insert(0, "seed", seed)
                hist.insert(1, "budget", budget)
                hist["provenance_key"] = row["provenance_key"]
                histories.append(hist)
                key = f"{family}__{transmission}"
                ddm_states[key] = U
                coarse_states[key] = result["coarse"]["state"]

        ddm_df = pd.DataFrame(ddm_rows)
        ddm_df.to_csv(run_dir / "ddm_metrics.csv", index=False)
        hist_df = pd.concat(histories, ignore_index=True)
        hist_df.to_csv(run_dir / "ddm_history.csv", index=False)

        # Four-gate adjudication.
        gate_alloc = allocation_gate_rows(node_df, cfg)
        gate_alloc["seed"] = seed
        gate_alloc["budget"] = budget
        gate_ddm = ddm_gate_rows(ddm_df, allocation_df, node_df, cfg)
        gate_ddm["seed"] = seed
        gate_ddm["budget"] = budget
        gates = pd.concat([gate_alloc, gate_ddm], ignore_index=True, sort=False)
        gates.to_csv(run_dir / "gate_matrix.csv", index=False)

        # Compact exact replay arrays.
        if cfg.save_full_arrays:
            save = {
                "eval_x": ex, "eval_y": ey,
                "exact_u": exact_u(EX, EY, cfg),
                "pilot_x": pilot["x"], "pilot_y": pilot["y"],
                "pilot_u": pilot["u"], "pilot_G": pilot["G"], "pilot_R": pilot["R"],
            }
            for k, arr in allocation_states.items():
                save[f"allocation__{k}"] = arr
            for k, arr in ddm_states.items():
                save[f"ddm__{k}"] = arr
            for k, arr in coarse_states.items():
                save[f"coarse__{k}"] = arr
            np.savez_compressed(run_dir / "state_arrays.npz", **save)

        status = {
            "status": "COMPLETED", "seed": seed, "budget": budget,
            "started_unix": started, "ended_unix": time.time(),
            "elapsed_s": time.time() - started,
            "provenance_root": provenance_root,
            "warnings": [], "failure_reason": "",
        }
        json_dump(status_path, status)
        logger.info("Run completed in %.2f s", status["elapsed_s"])

        # The status and the final log record must be written before hashing.
        # This ordering repairs the provenance-only defect found in the S
        # production archive, where those two files changed after the manifest.
        for handler in logger.handlers:
            handler.flush()
        run_manifest = []
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.name != "run_manifest_sha256.csv":
                run_manifest.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        pd.DataFrame(run_manifest).to_csv(run_dir / "run_manifest_sha256.csv", index=False)

        return {"status": "COMPLETED", "seed": seed, "budget": budget, "run_dir": str(run_dir)}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Run failed: %s\n%s", exc, tb)
        status = {
            "status": "FAILED", "seed": seed, "budget": budget,
            "started_unix": started, "ended_unix": time.time(),
            "elapsed_s": time.time() - started,
            "warnings": [], "failure_reason": repr(exc), "traceback": tb,
        }
        json_dump(status_path, status)
        return {"status": "FAILED", "seed": seed, "budget": budget, "run_dir": str(run_dir), "error": repr(exc)}


# =============================================================================
# Experiment aggregation / manuscript-ready figures
# =============================================================================

def read_completed_tables(campaign_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(campaign_dir.glob("seed_*_budget_*")):
        status_path = run_dir / "status.json"
        table_path = run_dir / filename
        if not status_path.exists() or not table_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") == "COMPLETED":
            frames.append(pd.read_csv(table_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_campaign_allocation(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    metrics = [("full_rmse", "Full-field RMSE"), ("interface_collar_rmse", "Interface-collar RMSE")]
    for a, (metric, title) in zip(ax, metrics):
        for fam, sub in df.groupby("allocation_family"):
            g = sub.groupby("budget")[metric]
            med, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
            x = med.index.to_numpy(dtype=float)
            a.plot(x, med.to_numpy(), marker="o", label=fam)
            a.fill_between(x, lo.to_numpy(), hi.to_numpy(), alpha=0.13)
        a.set_xlabel("Total interior-node budget")
        a.set_ylabel(title)
        a.set_title(title + ": median and seed 10--90% interval")
        a.set_yscale("log")
        a.grid(True, alpha=0.25)
    ax[0].legend(fontsize=7, ncol=2)
    fig.savefig(out, dpi=210)
    plt.close(fig)


def plot_refinement_intervals(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for fam in ("HALTON", "GHG_PROTECTED"):
        sub = df[df["allocation_family"] == fam]
        for a, metric in zip(ax, ("full_rmse", "condition_proxy")):
            g = sub.groupby("budget")[metric]
            med, lo, hi = g.median(), g.min(), g.max()
            x = med.index.to_numpy(dtype=float)
            a.plot(x, med.to_numpy(), marker="o", label=fam)
            a.fill_between(x, lo.to_numpy(), hi.to_numpy(), alpha=0.18)
    ax[0].set_ylabel("Full-field RMSE")
    ax[0].set_title("Refinement evidence: median and full seed range")
    ax[1].set_ylabel("Equilibrated condition proxy")
    ax[1].set_title("Local-system admissibility under refinement")
    for a in ax:
        a.set_xlabel("Total interior-node budget")
        a.set_yscale("log")
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig.savefig(out, dpi=210)
    plt.close(fig)


def compute_ablation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    key = ["seed", "budget"]
    ref = df[df["allocation_family"] == "GHG_PROTECTED"][
        key + ["full_rmse", "interface_collar_rmse", "condition_proxy", "range_violation"]
    ].copy()
    ref = ref.rename(columns={c: f"protected_{c}" for c in ref.columns if c not in key})
    out = df.merge(ref, on=key, how="left")
    for metric in ("full_rmse", "interface_collar_rmse", "condition_proxy"):
        out[f"{metric}_ratio_to_protected"] = out[metric] / out[f"protected_{metric}"]
    out["range_violation_difference_from_protected"] = (
        out["range_violation"] - out["protected_range_violation"]
    )
    return out


def plot_ablation_effects(ablation: pd.DataFrame, out: Path) -> None:
    if ablation.empty:
        return
    sub = ablation[ablation["allocation_family"].str.startswith("GHG_")]
    agg = sub.groupby("allocation_family", as_index=False)[
        ["full_rmse_ratio_to_protected", "interface_collar_rmse_ratio_to_protected"]
    ].median()
    x = np.arange(len(agg))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, agg["full_rmse_ratio_to_protected"], width, label="Full field")
    ax.bar(x + width / 2, agg["interface_collar_rmse_ratio_to_protected"], width, label="Interface collar")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["allocation_family"], rotation=25, ha="right")
    ax.set_ylabel("Median error ratio to GHG_PROTECTED")
    ax.set_title("Same-engine component ablations across seeds and budgets")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out, dpi=210)
    plt.close(fig)


def plot_transmission_gates(ddm: pd.DataFrame, gates: pd.DataFrame, out: Path) -> None:
    if ddm.empty or gates.empty:
        return
    final = gates[gates["gate"] == "ALL_FOUR_FINAL_GATES"]
    pass_rate = final.groupby(["allocation_family", "transmission"])["passed"].mean()
    mismatch = ddm.groupby(["allocation_family", "transmission"])["final_physical_mismatch"].median()
    idx = sorted(set(pass_rate.index).union(set(mismatch.index)))
    labels = [f"{a}\n{t}" for a, t in idx]
    x = np.arange(len(idx))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    ax[0].bar(x, [pass_rate.get(i, 0.0) for i in idx])
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("Fraction of seed-budget rows passing all four gates")
    ax[0].set_title("Final gate replication")
    ax[1].bar(x, [mismatch.get(i, np.nan) for i in idx])
    ax[1].axhline(5e-4, color="black", linestyle="--", linewidth=1, label="5e-4 terminal tolerance")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Median terminal physical mismatch")
    ax[1].set_title("Physical interface admissibility")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=68, ha="right", fontsize=7)
        a.grid(True, axis="y", alpha=0.22)
    fig.savefig(out, dpi=210)
    plt.close(fig)


def plot_campaign_histories(hist: pd.DataFrame, out: Path) -> None:
    if hist.empty:
        return
    budget = int(hist["budget"].max())
    h = hist[(hist["budget"] == budget) & (hist["allocation_family"] == "GHG_PROTECTED")]
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    for trans, sub in h.groupby("transmission"):
        g = sub.groupby("iteration")["physical_mismatch"]
        med, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
        x = med.index.to_numpy(dtype=float)
        ax.semilogy(x, med.to_numpy(), marker="o", ms=3, label=trans)
        ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), alpha=0.15)
    ax.axhline(5e-4, color="black", linestyle="--", linewidth=1, label="terminal tolerance")
    ax.set_xlabel("Schwarz iteration")
    ax.set_ylabel("Physical trace/flux mismatch")
    ax.set_title(f"GHG_PROTECTED mismatch histories at budget {budget}: median and seed 10--90% interval")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=210)
    plt.close(fig)


def campaign_summary_text(
    cfg: Config,
    seeds: List[int],
    budgets: List[int],
    alloc: pd.DataFrame,
    ddm: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    lines = []
    lines.append("CGF PAPER-I STATIC-INTERFACE EXPERIMENT SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Profile: {cfg.profile}")
    lines.append(f"Seeds: {seeds}")
    lines.append(f"Budgets: {budgets}")
    lines.append("Manuscript gaps adjudicated: G1-G3, R1, D1-D4, R4, F1, X1")
    lines.append("")
    lines.append("Transmission library is frozen in this code as:")
    for t in TRANSMISSION_LIBRARY:
        lines.append(f"  - {t}")
    lines.append("")

    if not alloc.empty:
        agg = alloc.groupby("allocation_family")[["full_rmse", "interface_collar_rmse", "range_violation", "condition_proxy"]].mean()
        lines.append("Mean allocation metrics across completed seed-budget rows:")
        lines.append(agg.to_string())
        lines.append("")

    if not ddm.empty:
        ddm2 = ddm.copy()
        ddm2["diagnostic_sum"] = ddm2["full_rmse"] + ddm2["interface_collar_rmse"] + ddm2["final_physical_mismatch"]
        best = ddm2.sort_values("diagnostic_sum").head(8)[[
            "allocation_family", "transmission", "q_net_physical", "q_tail_physical",
            "final_physical_mismatch", "full_rmse", "interface_collar_rmse",
            "conservation_rel_defect", "range_violation", "condition_max"
        ]]
        lines.append("Lowest diagnostic-sum rows (orientation only; not a paper score):")
        lines.append(best.to_string(index=False))
        lines.append("")

    if not gates.empty:
        final_gate = gates[gates["gate"] == "ALL_FOUR_FINAL_GATES"]
        if not final_gate.empty:
            replication = final_gate.groupby(["allocation_family", "transmission"])["passed"].agg(["sum", "count"])
            lines.append("All-four-gate replication counts:")
            lines.append(replication.to_string())
            lines.append("")

            legacy = final_gate[
                (final_gate["allocation_family"] == "GHG_PROTECTED")
                & (final_gate["transmission"] == "ROBIN_GHG_LEGACY_CONTROL")
            ]
            expected = len(seeds) * len(budgets)
            if cfg.profile == "smoke":
                decision = "PENDING: smoke is a structural test and is not scientific evidence."
            elif len(legacy) < expected:
                decision = "PENDING: the declared seed-budget rectangle is incomplete."
            elif bool(legacy["passed"].all()):
                decision = "D4 PASS in this profile: the legacy GHG-conditioned branch passed every final row."
            else:
                decision = (
                    "D4 FAIL/DEFER for Paper I: at least one final legacy GHG-conditioned "
                    "row failed; do not add or retune a transmission condition."
                )
            lines.append("Publication-cutoff decision:")
            lines.append(decision)
            lines.append("")

    lines.extend([
        "INTERPRETATION DISCIPLINE",
        "-------------------------",
        "The allocation comparison, component ablations, contraction histories,",
        "accuracy, conservation, and condition proxies are separate numerical facts.",
        "No composite diagnostic sum is used as a manuscript acceptance gate.",
        "",
        "If ROBIN_GHG_LEGACY_CONTROL is inadmissible in production or holdout,",
        "Paper I invokes the hard DDM cutoff: integrated GHG-",
        "conditioned DDM is FAIL/DEFER, not an invitation to invent another operator.",
        "A conventional Schwarz branch may remain only if it independently passes",
        "replication, refinement, contraction, accuracy, conservation, and provenance.",
    ])
    return "\n".join(lines) + "\n"


def write_gap_map(path: Path) -> None:
    rows = [
        ("G1", "Deployable Green/interface indicator", "pilot coefficient-jump distance field", "Methods"),
        ("G2", "Operational Halton component", "protected low-discrepancy coverage floor", "Methods"),
        ("G3", "Finite collar rule", "delta = collar_factor * pilot_dx", "Methods"),
        ("R1", "Static-interface replay", "monolithic coupled PHS-RBF-FD allocation rows", "Results"),
        ("D1", "Restricted DDM class", "two physical subdomains + evolving coarse correction", "DDM"),
        ("D2", "Finite transmission library", ";".join(TRANSMISSION_LIBRARY), "DDM"),
        ("D3", "True two-way histories", "trace/flux/Robin/update/coarse history per run", "DDM"),
        ("D4", "GHG-conditioned DDM adjudication", "all four final gates including terminal mismatch", "DDM/Claims"),
        ("R4", "Schwarz replication", "five seeds x four budgets x finite transmission library", "Results"),
        ("F1", "Journal figure sources", "five exact PNGs + CSV source tables", "Figures"),
        ("X1", "Replay provenance", "config/environment/status/manifests/state arrays", "Data/Code"),
    ]
    pd.DataFrame(rows, columns=["gap_id", "gap", "artifact_or_test", "manuscript_destination"]).to_csv(path, index=False)


def aggregate_campaign(campaign_dir: Path, cfg: Config, seeds: List[int], budgets: List[int]) -> None:
    alloc = read_completed_tables(campaign_dir, "allocation_metrics.csv")
    ddm = read_completed_tables(campaign_dir, "ddm_metrics.csv")
    hist = read_completed_tables(campaign_dir, "ddm_history.csv")
    gates = read_completed_tables(campaign_dir, "gate_matrix.csv")
    nodes = read_completed_tables(campaign_dir, "node_metrics.csv")

    if not alloc.empty:
        alloc.to_csv(campaign_dir / "campaign_allocation_metrics.csv", index=False)
        summary = alloc.groupby(["budget", "allocation_family"]).agg(
            full_rmse_mean=("full_rmse", "mean"),
            full_rmse_std=("full_rmse", "std"),
            full_rmse_median=("full_rmse", "median"),
            interface_rmse_mean=("interface_collar_rmse", "mean"),
            interface_rmse_std=("interface_collar_rmse", "std"),
            condition_proxy_max=("condition_proxy", "max"),
            completed_seeds=("seed", "nunique"),
        ).reset_index()
        summary.to_csv(campaign_dir / "seed_refinement_summary.csv", index=False)
        ablation = compute_ablation_metrics(alloc)
        ablation.to_csv(campaign_dir / "campaign_ablation_metrics.csv", index=False)
    else:
        ablation = pd.DataFrame()
    if not ddm.empty:
        ddm.to_csv(campaign_dir / "campaign_ddm_metrics.csv", index=False)
    if not hist.empty:
        hist.to_csv(campaign_dir / "campaign_ddm_history.csv", index=False)
    if not gates.empty:
        gates.to_csv(campaign_dir / "campaign_gate_matrix.csv", index=False)
        gates.to_csv(campaign_dir / "production_gate_matrix.csv", index=False)
    if not nodes.empty:
        nodes.to_csv(campaign_dir / "campaign_node_metrics.csv", index=False)

    figure_sources = [
        ("static_interface_allocation_errors.png", "campaign_allocation_metrics.csv", "Results: static-interface accuracy"),
        ("static_interface_refinement_intervals.png", "seed_refinement_summary.csv", "Results: replication and refinement"),
        ("static_interface_ablation_effects.png", "campaign_ablation_metrics.csv", "Results: role ablation"),
        ("static_interface_transmission_gates.png", "production_gate_matrix.csv;campaign_ddm_metrics.csv", "DDM: final gate adjudication"),
        ("static_interface_physical_mismatch_histories.png", "campaign_ddm_history.csv", "DDM: physical contraction"),
    ]
    plot_campaign_allocation(alloc, campaign_dir / figure_sources[0][0])
    plot_refinement_intervals(alloc, campaign_dir / figure_sources[1][0])
    plot_ablation_effects(ablation, campaign_dir / figure_sources[2][0])
    plot_transmission_gates(ddm, gates, campaign_dir / figure_sources[3][0])
    plot_campaign_histories(hist, campaign_dir / figure_sources[4][0])

    provenance_rows = []
    for filename, source_table, destination in figure_sources:
        p = campaign_dir / filename
        provenance_rows.append({
            "figure_filename": filename,
            "producer_code": Path(__file__).name,
            "source_table_or_tables": source_table,
            "manuscript_destination": destination,
            "status": "generated" if p.exists() else "pending",
            "sha256": sha256_file(p) if p.exists() else "",
        })
    pd.DataFrame(provenance_rows).to_csv(campaign_dir / "figure_provenance.csv", index=False)

    pd.DataFrame([
        ("Methods: physical fingerprint and allocation", "fingerprint.csv;node_metrics.csv;campaign_allocation_metrics.csv", "G1-G3"),
        ("Results: static-interface accuracy", "seed_refinement_summary.csv;campaign_ablation_metrics.csv", "R1;R4"),
        ("Domain decomposition", "campaign_ddm_metrics.csv;campaign_ddm_history.csv;production_gate_matrix.csv", "D1-D4"),
        ("Figures and reproducibility", "figure_provenance.csv;campaign_manifest_sha256.csv;state_arrays.npz", "F1;X1"),
    ], columns=["paper_section", "scientific_artifacts", "gap_ids"]).to_csv(
        campaign_dir / "paper_section_output_map.csv", index=False
    )

    decision_text = campaign_summary_text(cfg, seeds, budgets, alloc, ddm, gates)
    (campaign_dir / "campaign_summary.txt").write_text(decision_text, encoding="utf-8")
    (campaign_dir / "production_decision_summary.txt").write_text(decision_text, encoding="utf-8")

    manifest = []
    for p in sorted(campaign_dir.iterdir()):
        if p.is_file() and p.name != "campaign_manifest_sha256.csv":
            manifest.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    pd.DataFrame(manifest).to_csv(campaign_dir / "campaign_manifest_sha256.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CGF Paper-I static-interface production/holdout experiment")
    ap.add_argument("--profile", choices=["smoke", "production", "holdout"], default=BASE_CONFIG.profile)
    ap.add_argument("--force", action="store_true", help="rerun completed seed/budget folders")
    ap.add_argument("--campaign-dir", default=None, help="override the profile-specific output folder")
    ap.add_argument(
        "--production-dir", default="static_interface_production_campaign",
        help="completed production folder required to authorize the holdout",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    campaign_name = args.campaign_dir or default_campaign_dir(args.profile)
    benchmark = (
        {"interface_x": 0.37, "k_left": 1.0, "k_right": 30.0}
        if args.profile == "holdout"
        else {"interface_x": 0.50, "k_left": 1.0, "k_right": 100.0}
    )
    cfg = replace(
        BASE_CONFIG,
        profile=args.profile,
        force=args.force or FORCE,
        campaign_dir=campaign_name,
        **benchmark,
    )
    cfg.validate()
    seeds, budgets = profile_seeds_budgets(cfg.profile)

    campaign_dir = Path(cfg.campaign_dir)
    contract = frozen_experiment_contract(cfg)
    if cfg.profile == "holdout":
        verify_holdout_authorization(Path(args.production_dir), contract)
    ensure_dir(campaign_dir)
    existing_contract = campaign_dir / "frozen_experiment_contract.json"
    if existing_contract.exists():
        stored = json.loads(existing_contract.read_text(encoding="utf-8"))
        if stored.get("frozen_experiment_sha256") != contract["frozen_experiment_sha256"]:
            raise RuntimeError("Existing campaign folder contains a different frozen experiment contract.")
    json_dump(existing_contract, contract)
    json_dump(campaign_dir / "campaign_config.json", {
        **asdict(cfg), "seeds": seeds, "budgets": budgets,
        "allocation_families": ALLOCATION_FAMILIES,
        "ddm_allocation_families": DDM_ALLOCATION_FAMILIES,
        "transmission_library": TRANSMISSION_LIBRARY,
        "frozen_experiment_sha256": contract["frozen_experiment_sha256"],
        "publication_boundary_note": contract["paper_I_stop_rule"],
    })
    json_dump(campaign_dir / "environment.json", environment_record())
    write_gap_map(campaign_dir / "manuscript_gap_map.csv")

    run_index = []
    for seed in seeds:
        for budget in budgets:
            run_index.append(run_one(seed, budget, cfg, campaign_dir))
            pd.DataFrame(run_index).to_csv(campaign_dir / "campaign_run_index.csv", index=False)

    aggregate_campaign(campaign_dir, cfg, seeds, budgets)

    completed = all(r.get("status") in ("COMPLETED", "SKIPPED_COMPLETED") for r in run_index)
    if completed and cfg.profile in ("production", "holdout"):
        marker_name = "production_complete.json" if cfg.profile == "production" else "holdout_complete.json"
        json_dump(campaign_dir / marker_name, {
            "status": "COMPLETED",
            "profile": cfg.profile,
            "completed_seed_budget_rows": len(seeds) * len(budgets),
            "frozen_experiment_sha256": contract["frozen_experiment_sha256"],
            "completed_unix": time.time(),
        })
        # Refresh the top-level manifest so the completion marker is covered.
        aggregate_campaign(campaign_dir, cfg, seeds, budgets)
    print(f"\nCampaign outputs: {campaign_dir.resolve()}")
    print("Read production_decision_summary.txt first, then production_gate_matrix.csv.")
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
