"""Shared numerical core for the CGF clean-replay campaigns.

The module implements one interface-aware, polynomial-augmented PHS RBF-FD
engine.  All allocation families therefore use the same differential
operator, boundary representation, interface constraints, evaluation grid,
and diagnostics.  It is intentionally dependency-light (NumPy/SciPy only)
so that the campaign entry points can be run directly in Spyder.

This is research software.  A ``screen`` profile checks the implementation;
only a completed ``production`` profile with its manifest may be cited.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree, distance_matrix
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, onenormest, splu
from scipy.stats import qmc


EPS = np.finfo(float).eps
PHS_POWER = 3
POLYNOMIAL_DEGREE = 2
DEFAULT_STENCIL = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def protocol_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def environment_record() -> dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
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


class OrderedLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{utc_now()} | {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class InterfaceProblem:
    """Manufactured two-material problem on the unit square.

    ``phi < 0`` identifies the minus material and ``phi > 0`` the plus
    material. ``normal`` points from minus to plus at the interface.
    """

    name: str
    k_minus: float
    k_plus: float
    phi_function: Callable[[np.ndarray], np.ndarray]
    exact_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    forcing_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    interface_sampler: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    metadata: dict[str, Any]

    def phi(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(self.phi_function(np.atleast_2d(points)), dtype=float)

    def side(self, points: np.ndarray) -> np.ndarray:
        return np.where(self.phi(points) <= 0.0, -1, 1)

    def conductivity(self, points: np.ndarray) -> np.ndarray:
        return np.where(self.side(points) < 0, self.k_minus, self.k_plus)

    def exact(self, points: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
        points = np.atleast_2d(points)
        if side is None:
            side = self.side(points)
        return np.asarray(self.exact_function(points, np.asarray(side)), dtype=float)

    def forcing(self, points: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
        points = np.atleast_2d(points)
        if side is None:
            side = self.side(points)
        return np.asarray(self.forcing_function(points, np.asarray(side)), dtype=float)

    def sample_interface(self, unit_parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.interface_sampler(np.asarray(unit_parameter, dtype=float))


def vertical_resistance_problem(
    name: str,
    xi: float,
    k_left: float,
    k_right: float,
) -> InterfaceProblem:
    resistance = xi / k_left + (1.0 - xi) / k_right

    def phi(points: np.ndarray) -> np.ndarray:
        return points[:, 0] - xi

    def exact(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        sx = np.where(
            side < 0,
            x / k_left,
            xi / k_left + (x - xi) / k_right,
        ) / resistance
        return sx * np.sin(np.pi * y)

    def forcing(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        k = np.where(side < 0, k_left, k_right)
        return k * np.pi**2 * exact(points, side)

    def interface(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.clip(u, 1.0e-6, 1.0 - 1.0e-6)
        points = np.column_stack((np.full_like(y, xi), y))
        normals = np.tile(np.array([[1.0, 0.0]]), (y.size, 1))
        return points, normals

    return InterfaceProblem(
        name=name,
        k_minus=float(k_left),
        k_plus=float(k_right),
        phi_function=phi,
        exact_function=exact,
        forcing_function=forcing,
        interface_sampler=interface,
        metadata={"geometry": "vertical_line", "xi": xi, "resistance": resistance},
    )


def _line_segment_in_square(center: np.ndarray, tangent: np.ndarray) -> tuple[float, float]:
    lower, upper = -np.inf, np.inf
    for coordinate in range(2):
        value = float(center[coordinate])
        direction = float(tangent[coordinate])
        if abs(direction) < 1.0e-14:
            if not (0.0 <= value <= 1.0):
                raise ValueError("Line does not cross the unit square")
            continue
        a = (0.0 - value) / direction
        b = (1.0 - value) / direction
        lower = max(lower, min(a, b))
        upper = min(upper, max(a, b))
    if not lower < upper:
        raise ValueError("Degenerate interface segment")
    return float(lower), float(upper)


def oblique_interface_problem(
    name: str,
    angle_degrees: float,
    center: tuple[float, float],
    k_minus: float,
    k_plus: float,
    normal_flux: float = 0.7,
    tangential_amplitude: float = 0.18,
) -> InterfaceProblem:
    theta = math.radians(angle_degrees)
    normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    center_array = np.asarray(center, dtype=float)
    s_min, s_max = _line_segment_in_square(center_array, tangent)
    segment_length = s_max - s_min
    wave_number = 2.0 * np.pi / segment_length

    def phi(points: np.ndarray) -> np.ndarray:
        return (points - center_array) @ normal

    def exact(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        normal_coordinate = phi(points)
        tangential_coordinate = (points - center_array) @ tangent
        k = np.where(side < 0, k_minus, k_plus)
        return (
            0.6
            + normal_flux * normal_coordinate / k
            + tangential_amplitude * np.cos(wave_number * (tangential_coordinate - s_min))
        )

    def forcing(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        tangential_coordinate = (points - center_array) @ tangent
        k = np.where(side < 0, k_minus, k_plus)
        return k * tangential_amplitude * wave_number**2 * np.cos(
            wave_number * (tangential_coordinate - s_min)
        )

    def interface(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        margin = 1.0e-5 * segment_length
        s = s_min + margin + np.clip(u, 0.0, 1.0) * (segment_length - 2.0 * margin)
        points = center_array + s[:, None] * tangent
        normals = np.tile(normal, (u.size, 1))
        return points, normals

    return InterfaceProblem(
        name=name,
        k_minus=float(k_minus),
        k_plus=float(k_plus),
        phi_function=phi,
        exact_function=exact,
        forcing_function=forcing,
        interface_sampler=interface,
        metadata={
            "geometry": "oblique_line",
            "angle_degrees": angle_degrees,
            "center": list(center),
            "segment_parameter": [s_min, s_max],
            "normal_flux": normal_flux,
            "tangential_amplitude": tangential_amplitude,
        },
    )


def circular_inclusion_problem(
    name: str,
    center: tuple[float, float],
    radius: float,
    k_inside: float,
    k_outside: float,
    outer_quadratic: float = 0.8,
) -> InterfaceProblem:
    center_array = np.asarray(center, dtype=float)
    if radius <= 0.0 or np.any(center_array - radius <= 0.0) or np.any(center_array + radius >= 1.0):
        raise ValueError("Circular interface must lie strictly inside the unit square")
    inner_quadratic = outer_quadratic * k_outside / k_inside
    outer_constant = 0.25
    inner_constant = outer_constant + (outer_quadratic - inner_quadratic) * radius**2

    def phi(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - center_array, axis=1) - radius

    def exact(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        r2 = np.sum((points - center_array) ** 2, axis=1)
        return np.where(
            side < 0,
            inner_constant + inner_quadratic * r2,
            outer_constant + outer_quadratic * r2,
        )

    def forcing(points: np.ndarray, side: np.ndarray) -> np.ndarray:
        k = np.where(side < 0, k_inside, k_outside)
        a = np.where(side < 0, inner_quadratic, outer_quadratic)
        return -4.0 * k * a * np.ones(points.shape[0])

    def interface(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        angle = 2.0 * np.pi * np.mod(u, 1.0)
        normals = np.column_stack((np.cos(angle), np.sin(angle)))
        points = center_array + radius * normals
        return points, normals

    return InterfaceProblem(
        name=name,
        k_minus=float(k_inside),
        k_plus=float(k_outside),
        phi_function=phi,
        exact_function=exact,
        forcing_function=forcing,
        interface_sampler=interface,
        metadata={
            "geometry": "circular_inclusion",
            "center": list(center),
            "radius": radius,
            "outer_quadratic": outer_quadratic,
        },
    )


def _interior_filter(points: np.ndarray, problem: InterfaceProblem, interface_gap: float) -> np.ndarray:
    margin = 1.0e-5
    keep = np.all((points > margin) & (points < 1.0 - margin), axis=1)
    keep &= np.abs(problem.phi(points)) > interface_gap
    return points[keep]


def _halton_pool(count: int, seed: int, start: int = 0, dimension: int = 2) -> np.ndarray:
    engine = qmc.Halton(d=dimension, scramble=True, seed=seed)
    if start:
        engine.fast_forward(start)
    return engine.random(count)


def global_halton(problem: InterfaceProblem, count: int, seed: int, interface_gap: float = 2.0e-5) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2))
    selected: list[np.ndarray] = []
    start = 0
    while sum(block.shape[0] for block in selected) < count:
        block_size = max(64, 2 * (count - sum(block.shape[0] for block in selected)))
        pool = _halton_pool(block_size, seed, start=start)
        start += block_size
        selected.append(_interior_filter(pool, problem, interface_gap))
    return np.vstack(selected)[:count]


def global_random(problem: InterfaceProblem, count: int, seed: int, interface_gap: float = 2.0e-5) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2))
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    while sum(block.shape[0] for block in selected) < count:
        pool = rng.random((max(64, 2 * count), 2))
        selected.append(_interior_filter(pool, problem, interface_gap))
    return np.vstack(selected)[:count]


def _band_enrichment(
    problem: InterfaceProblem,
    count: int,
    seed: int,
    half_width: float,
    random_control: bool,
    interface_gap: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2))
    selected: list[np.ndarray] = []
    generated = 0
    rng = np.random.default_rng(seed)
    while sum(block.shape[0] for block in selected) < count:
        batch = max(64, 3 * (count - sum(block.shape[0] for block in selected)))
        if random_control:
            unit = rng.random((batch, 2))
        else:
            unit = _halton_pool(batch, seed, start=generated, dimension=2)
            generated += batch
        base, normals = problem.sample_interface(unit[:, 0])
        offsets = (2.0 * unit[:, 1] - 1.0) * half_width
        offsets = np.where(
            np.abs(offsets) < interface_gap,
            np.where(offsets < 0.0, -interface_gap, interface_gap),
            offsets,
        )
        points = base + offsets[:, None] * normals
        selected.append(_interior_filter(points, problem, interface_gap * 0.95))
    return np.vstack(selected)[:count]


def protected_band_allocation(
    problem: InterfaceProblem,
    count: int,
    seed: int,
    skeleton_fraction: float,
    half_width: float,
    random_control: bool = False,
    interface_gap: float = 2.0e-5,
) -> tuple[np.ndarray, int]:
    skeleton_count = int(math.ceil(skeleton_fraction * count))
    skeleton = global_halton(problem, skeleton_count, seed, interface_gap)
    enrichment = _band_enrichment(
        problem,
        count - skeleton_count,
        seed + 104729,
        half_width,
        random_control,
        interface_gap,
    )
    return np.vstack((skeleton, enrichment)), skeleton_count


def normalized_indicator(values: np.ndarray, floor: float = 0.02) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    upper = float(np.quantile(values, 0.98)) if values.size else 0.0
    lower = float(np.quantile(values, 0.02)) if values.size else 0.0
    if upper <= lower + EPS:
        return np.ones_like(values)
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return floor + (1.0 - floor) * scaled


class PilotGradient:
    def __init__(self, problem: InterfaceProblem, minus_points: np.ndarray, minus_values: np.ndarray,
                 plus_points: np.ndarray, plus_values: np.ndarray, neighbors: int = 14):
        self.problem = problem
        self.minus_points = minus_points
        self.plus_points = plus_points
        self.minus_tree = cKDTree(minus_points)
        self.plus_tree = cKDTree(plus_points)
        self.minus_magnitude = self._least_squares_gradients(minus_points, minus_values, neighbors)
        self.plus_magnitude = self._least_squares_gradients(plus_points, plus_values, neighbors)

    @staticmethod
    def _least_squares_gradients(points: np.ndarray, values: np.ndarray, neighbors: int) -> np.ndarray:
        tree = cKDTree(points)
        k = min(max(6, neighbors), points.shape[0])
        _, indices = tree.query(points, k=k)
        if indices.ndim == 1:
            indices = indices[:, None]
        result = np.zeros(points.shape[0])
        for row, stencil in enumerate(indices):
            delta = points[stencil] - points[row]
            design = np.column_stack((np.ones(stencil.size), delta))
            coefficients, *_ = np.linalg.lstsq(design, values[stencil], rcond=None)
            result[row] = float(np.linalg.norm(coefficients[1:]))
        return normalized_indicator(result)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        side = self.problem.side(points)
        result = np.empty(points.shape[0])
        minus = side < 0
        if np.any(minus):
            _, index = self.minus_tree.query(points[minus], k=1)
            result[minus] = self.minus_magnitude[index]
        if np.any(~minus):
            _, index = self.plus_tree.query(points[~minus], k=1)
            result[~minus] = self.plus_magnitude[index]
        return result


def protected_weighted_allocation(
    problem: InterfaceProblem,
    count: int,
    seed: int,
    component_weights: tuple[float, float, float],
    pilot_gradient: Callable[[np.ndarray], np.ndarray] | None,
    skeleton_fraction: float = 0.45,
    green_length: float = 0.10,
    candidate_multiplier: int = 18,
    interface_gap: float = 2.0e-5,
) -> tuple[np.ndarray, int]:
    alpha, beta, gamma = component_weights
    if min(alpha, beta, gamma) < 0.0 or alpha + beta + gamma <= 0.0:
        raise ValueError("Component weights must be nonnegative with positive sum")
    total = alpha + beta + gamma
    alpha, beta, gamma = alpha / total, beta / total, gamma / total
    skeleton_count = int(math.ceil(skeleton_fraction * count))
    skeleton = global_halton(problem, skeleton_count, seed, interface_gap)
    enrichment_count = count - skeleton_count
    if enrichment_count <= 0:
        return skeleton, skeleton_count
    pool_count = max(candidate_multiplier * enrichment_count, enrichment_count + 1)
    raw = _halton_pool(pool_count * 2, seed + 2750159, dimension=3)
    raw_points = raw[:, :2]
    margin = 1.0e-5
    keep = np.all((raw_points > margin) & (raw_points < 1.0 - margin), axis=1)
    keep &= np.abs(problem.phi(raw_points)) > interface_gap
    candidates = raw_points[keep]
    random_key = raw[keep, 2]
    if candidates.shape[0] < pool_count:
        raise RuntimeError("Candidate pool unexpectedly small")
    candidates = candidates[:pool_count]
    random_key = random_key[:pool_count]
    green = normalized_indicator(np.exp(-np.abs(problem.phi(candidates)) / max(green_length, 1.0e-6)))
    coverage = np.ones(pool_count)
    if pilot_gradient is None:
        gradient = normalized_indicator(np.abs(problem.forcing(candidates)))
    else:
        gradient = normalized_indicator(pilot_gradient(candidates))
    density = np.maximum(alpha * green + beta * coverage + gamma * gradient, 1.0e-8)
    keys = -np.log(np.clip(random_key, np.finfo(float).tiny, 1.0)) / density
    selected = np.argpartition(keys, enrichment_count - 1)[:enrichment_count]
    selected.sort()
    return np.vstack((skeleton, candidates[selected])), skeleton_count


def coverage_metrics(points: np.ndarray, evaluation_resolution: int = 45) -> dict[str, float]:
    tree = cKDTree(points)
    grid = (np.arange(evaluation_resolution) + 0.5) / evaluation_resolution
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    evaluation = np.column_stack((xx.ravel(), yy.ravel()))
    fill, _ = tree.query(evaluation, k=1)
    near, _ = tree.query(points, k=min(2, points.shape[0]))
    separation = near[:, -1] if near.ndim == 2 else near
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=12, range=((0, 1), (0, 1)))
    return {
        "coverage_fill_max": float(np.max(fill)),
        "coverage_fill_rms": float(np.sqrt(np.mean(fill**2))),
        "coverage_min_separation": float(np.min(separation)),
        "coverage_empty_fraction_12x12": float(np.mean(counts == 0)),
    }


def _outer_boundary_points(per_edge: int) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, per_edge + 1)
    bottom = np.column_stack((coordinate, np.zeros_like(coordinate)))
    right = np.column_stack((np.ones_like(coordinate[1:]), coordinate[1:]))
    top = np.column_stack((coordinate[-2::-1], np.ones_like(coordinate[-2::-1])))
    left_coordinate = coordinate[-2:0:-1]
    left = np.column_stack((np.zeros_like(left_coordinate), left_coordinate))
    return np.vstack((bottom, right, top, left))


@dataclass
class SideCloud:
    side: int
    conductivity: float
    nodes: np.ndarray
    interior: np.ndarray
    boundary: np.ndarray
    interface: np.ndarray
    interface_points: np.ndarray
    interface_normals_out: np.ndarray
    tree: cKDTree


def build_side_clouds(
    problem: InterfaceProblem,
    interior_points: np.ndarray,
    boundary_per_edge: int,
    interface_count: int,
) -> tuple[SideCloud, SideCloud]:
    side = problem.side(interior_points)
    boundary_points = _outer_boundary_points(boundary_per_edge)
    boundary_phi = problem.phi(boundary_points)
    boundary_points = boundary_points[np.abs(boundary_phi) > 1.0e-10]
    boundary_side = problem.side(boundary_points)
    if problem.metadata["geometry"] == "circular_inclusion":
        interface_parameter = (np.arange(interface_count) + 0.5) / interface_count
    else:
        interface_parameter = (np.arange(interface_count) + 1.0) / (interface_count + 1.0)
    interface_points, normals = problem.sample_interface(interface_parameter)

    clouds: list[SideCloud] = []
    for sign, conductivity, normal_sign in ((-1, problem.k_minus, 1.0), (1, problem.k_plus, -1.0)):
        interior_local = interior_points[side == sign]
        boundary_local = boundary_points[boundary_side == sign]
        nodes = np.vstack((interior_local, boundary_local, interface_points))
        n_i = interior_local.shape[0]
        n_b = boundary_local.shape[0]
        interior_index = np.arange(n_i, dtype=int)
        boundary_index = np.arange(n_i, n_i + n_b, dtype=int)
        interface_index = np.arange(n_i + n_b, nodes.shape[0], dtype=int)
        if n_i < 10 or nodes.shape[0] < 10:
            raise RuntimeError(f"Insufficient nodes in side {sign}: interior={n_i}, total={nodes.shape[0]}")
        clouds.append(
            SideCloud(
                side=sign,
                conductivity=float(conductivity),
                nodes=nodes,
                interior=interior_index,
                boundary=boundary_index,
                interface=interface_index,
                interface_points=interface_points,
                interface_normals_out=normal_sign * normals,
                tree=cKDTree(nodes),
            )
        )
    return clouds[0], clouds[1]


def _polynomial_matrix(z: np.ndarray) -> np.ndarray:
    x, y = z[:, 0], z[:, 1]
    return np.column_stack((np.ones(z.shape[0]), x, y, x * x, x * y, y * y))


def local_phs_weights(
    cloud: SideCloud,
    center: np.ndarray,
    operator: str,
    direction: np.ndarray | None = None,
    stencil_size: int = DEFAULT_STENCIL,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(max(10, stencil_size), cloud.nodes.shape[0])
    _, stencil = cloud.tree.query(center, k=k)
    stencil = np.atleast_1d(stencil).astype(int)
    points = cloud.nodes[stencil]
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    if not np.isfinite(scale) or scale <= 1.0e-14:
        raise RuntimeError("Degenerate local RBF-FD stencil")
    z = (points - center) / scale
    radial = distance_matrix(z, z) ** PHS_POWER
    polynomial = _polynomial_matrix(z)
    augmented = np.block(
        [[radial, polynomial], [polynomial.T, np.zeros((polynomial.shape[1], polynomial.shape[1]))]]
    )
    r = np.linalg.norm(z, axis=1)
    if operator == "laplacian":
        radial_rhs = 9.0 * r / scale**2
        polynomial_rhs = np.array([0.0, 0.0, 0.0, 2.0 / scale**2, 0.0, 2.0 / scale**2])
    elif operator == "directional":
        if direction is None:
            raise ValueError("A direction is required")
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        radial_rhs = -3.0 * r * (z @ direction) / scale
        polynomial_rhs = np.array([0.0, direction[0] / scale, direction[1] / scale, 0.0, 0.0, 0.0])
    elif operator == "value":
        radial_rhs = r**PHS_POWER
        polynomial_rhs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown operator: {operator}")
    rhs = np.concatenate((radial_rhs, polynomial_rhs))
    try:
        solution = np.linalg.solve(augmented, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(augmented, rhs, rcond=1.0e-12)
    weights = solution[:stencil.size]
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("Non-finite local RBF-FD weights")
    return stencil, weights


@dataclass
class MonolithicResult:
    problem: InterfaceProblem
    minus: SideCloud
    plus: SideCloud
    values_minus: np.ndarray
    values_plus: np.ndarray
    matrix: csr_matrix
    matrix_raw: csr_matrix
    rhs: np.ndarray
    condition_proxy: float
    solve_seconds: float
    metrics: dict[str, Any]


def _equilibrated_condition_proxy(matrix: csr_matrix, lu: Any) -> float:
    norm_a = float(onenormest(matrix))

    def matvec(vector: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(vector, dtype=float))

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(vector, dtype=float), trans="T")

    inverse = LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
    return norm_a * float(onenormest(inverse))


def _append_weight_row(rows: list[int], columns: list[int], data: list[float],
                       row_index: int, offset: int, stencil: np.ndarray, weights: np.ndarray) -> None:
    rows.extend([row_index] * stencil.size)
    columns.extend((offset + stencil).tolist())
    data.extend(weights.tolist())


def assemble_monolithic(
    problem: InterfaceProblem,
    interior_points: np.ndarray,
    boundary_per_edge: int,
    interface_count: int,
    stencil_size: int = DEFAULT_STENCIL,
) -> tuple[SideCloud, SideCloud, csr_matrix, np.ndarray]:
    minus, plus = build_side_clouds(problem, interior_points, boundary_per_edge, interface_count)
    offsets = {-1: 0, 1: minus.nodes.shape[0]}
    total = minus.nodes.shape[0] + plus.nodes.shape[0]
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    rhs = np.zeros(total)
    row = 0

    for cloud in (minus, plus):
        offset = offsets[cloud.side]
        for local_index in cloud.interior:
            center = cloud.nodes[local_index]
            stencil, weights = local_phs_weights(cloud, center, "laplacian", stencil_size=stencil_size)
            _append_weight_row(rows, columns, data, row, offset, stencil, -cloud.conductivity * weights)
            rhs[row] = problem.forcing(center[None, :], np.array([cloud.side]))[0]
            row += 1
        for local_index in cloud.boundary:
            rows.append(row)
            columns.append(offset + int(local_index))
            data.append(1.0)
            point = cloud.nodes[local_index]
            rhs[row] = problem.exact(point[None, :], np.array([cloud.side]))[0]
            row += 1
        if cloud.side == -1:
            for interface_position, local_index in enumerate(cloud.interface):
                rows.extend((row, row))
                columns.extend((offset + int(local_index), offsets[1] + int(plus.interface[interface_position])))
                data.extend((1.0, -1.0))
                row += 1
        else:
            for interface_position, local_index in enumerate(cloud.interface):
                point = cloud.nodes[local_index]
                normal = -cloud.interface_normals_out[interface_position]  # minus-to-plus normal
                stencil_minus, weights_minus = local_phs_weights(
                    minus, point, "directional", direction=normal, stencil_size=stencil_size
                )
                stencil_plus, weights_plus = local_phs_weights(
                    plus, point, "directional", direction=normal, stencil_size=stencil_size
                )
                _append_weight_row(
                    rows, columns, data, row, offsets[-1], stencil_minus, problem.k_minus * weights_minus
                )
                _append_weight_row(
                    rows, columns, data, row, offsets[1], stencil_plus, -problem.k_plus * weights_plus
                )
                row += 1
    if row != total:
        raise RuntimeError(f"Row-count defect: assembled {row}, expected {total}")
    matrix = coo_matrix((data, (rows, columns)), shape=(total, total)).tocsr()
    return minus, plus, matrix, rhs


def _piecewise_interpolators(cloud: SideCloud, values: np.ndarray) -> tuple[Any, Any]:
    linear = LinearNDInterpolator(cloud.nodes, values, fill_value=np.nan)
    nearest = NearestNDInterpolator(cloud.nodes, values)
    return linear, nearest


def evaluate_piecewise(
    problem: InterfaceProblem,
    minus: SideCloud,
    plus: SideCloud,
    values_minus: np.ndarray,
    values_plus: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    points = np.atleast_2d(points)
    side = problem.side(points)
    result = np.empty(points.shape[0])
    for sign, cloud, values in ((-1, minus, values_minus), (1, plus, values_plus)):
        mask = side == sign
        if not np.any(mask):
            continue
        linear, nearest = _piecewise_interpolators(cloud, values)
        local = np.asarray(linear(points[mask]), dtype=float).reshape(-1)
        missing = ~np.isfinite(local)
        if np.any(missing):
            local[missing] = np.asarray(nearest(points[mask][missing]), dtype=float).reshape(-1)
        result[mask] = local
    return result


def interface_diagnostics(
    problem: InterfaceProblem,
    minus: SideCloud,
    plus: SideCloud,
    values_minus: np.ndarray,
    values_plus: np.ndarray,
    stencil_size: int,
) -> dict[str, float]:
    trace = values_minus[minus.interface] - values_plus[plus.interface]
    flux = np.zeros(minus.interface.size)
    for position, point in enumerate(minus.interface_points):
        normal = minus.interface_normals_out[position]
        stencil_minus, weights_minus = local_phs_weights(
            minus, point, "directional", direction=normal, stencil_size=stencil_size
        )
        stencil_plus, weights_plus = local_phs_weights(
            plus, point, "directional", direction=normal, stencil_size=stencil_size
        )
        flux[position] = (
            problem.k_minus * float(weights_minus @ values_minus[stencil_minus])
            - problem.k_plus * float(weights_plus @ values_plus[stencil_plus])
        )
    return {
        "interface_trace_rms": float(np.sqrt(np.mean(trace**2))),
        "interface_flux_rms": float(np.sqrt(np.mean(flux**2))),
        "interface_trace_max": float(np.max(np.abs(trace))),
        "interface_flux_max": float(np.max(np.abs(flux))),
    }


def conservation_defect(
    problem: InterfaceProblem,
    minus: SideCloud,
    plus: SideCloud,
    values_minus: np.ndarray,
    values_plus: np.ndarray,
    stencil_size: int,
    quadrature_per_edge: int = 30,
) -> float:
    coordinate = (np.arange(quadrature_per_edge) + 0.5) / quadrature_per_edge
    boundary_sets = (
        (np.column_stack((coordinate, np.zeros_like(coordinate))), np.array([0.0, -1.0])),
        (np.column_stack((np.ones_like(coordinate), coordinate)), np.array([1.0, 0.0])),
        (np.column_stack((coordinate, np.ones_like(coordinate))), np.array([0.0, 1.0])),
        (np.column_stack((np.zeros_like(coordinate), coordinate)), np.array([-1.0, 0.0])),
    )
    boundary_integral = 0.0
    for points, outward in boundary_sets:
        side = problem.side(points)
        flux_values = np.zeros(points.shape[0])
        for sign, cloud, values in ((-1, minus, values_minus), (1, plus, values_plus)):
            positions = np.where(side == sign)[0]
            for position in positions:
                stencil, weights = local_phs_weights(
                    cloud, points[position], "directional", direction=outward, stencil_size=stencil_size
                )
                flux_values[position] = cloud.conductivity * float(weights @ values[stencil])
        boundary_integral += float(np.mean(flux_values))
    q = max(24, quadrature_per_edge)
    coordinate2 = (np.arange(q) + 0.5) / q
    xx, yy = np.meshgrid(coordinate2, coordinate2, indexing="xy")
    volume_points = np.column_stack((xx.ravel(), yy.ravel()))
    forcing_integral = float(np.mean(problem.forcing(volume_points)))
    numerator = abs(forcing_integral + boundary_integral)
    denominator = max(abs(forcing_integral), abs(boundary_integral), 1.0e-12)
    return float(numerator / denominator)


def solve_monolithic(
    problem: InterfaceProblem,
    interior_points: np.ndarray,
    boundary_per_edge: int,
    interface_count: int,
    evaluation_resolution: int,
    collar_width: float,
    stencil_size: int = DEFAULT_STENCIL,
    compute_condition: bool = True,
) -> MonolithicResult:
    start = time.perf_counter()
    minus, plus, raw_matrix, raw_rhs = assemble_monolithic(
        problem, interior_points, boundary_per_edge, interface_count, stencil_size
    )
    row_norm = np.sqrt(np.asarray(raw_matrix.multiply(raw_matrix).sum(axis=1)).reshape(-1))
    if np.any(row_norm <= 0.0) or not np.all(np.isfinite(row_norm)):
        raise RuntimeError("Zero or non-finite matrix row norm")
    scale = 1.0 / row_norm
    matrix = (diags(scale) @ raw_matrix).tocsc()
    rhs = scale * raw_rhs
    lu = splu(matrix)
    solution = lu.solve(rhs)
    residual = matrix @ solution - rhs
    condition = _equilibrated_condition_proxy(matrix.tocsr(), lu) if compute_condition else float("nan")
    n_minus = minus.nodes.shape[0]
    values_minus = solution[:n_minus]
    values_plus = solution[n_minus:]

    coordinate = np.linspace(0.0, 1.0, evaluation_resolution)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    evaluation = np.column_stack((xx.ravel(), yy.ravel()))
    numerical = evaluate_piecewise(problem, minus, plus, values_minus, values_plus, evaluation)
    exact = problem.exact(evaluation)
    error = numerical - exact
    collar = np.abs(problem.phi(evaluation)) <= collar_width
    exact_range = float(np.max(exact) - np.min(exact))
    overshoot = max(
        0.0,
        float(np.max(numerical) - np.max(exact)),
        float(np.min(exact) - np.min(numerical)),
    ) / max(exact_range, 1.0e-12)

    diagnostics = interface_diagnostics(
        problem, minus, plus, values_minus, values_plus, stencil_size
    )
    diagnostics.update(
        {
            "full_rmse": float(np.sqrt(np.mean(error**2))),
            "full_linf": float(np.max(np.abs(error))),
            "collar_rmse": float(np.sqrt(np.mean(error[collar] ** 2))),
            "collar_linf": float(np.max(np.abs(error[collar]))),
            "relative_overshoot": float(overshoot),
            "conservation_defect": conservation_defect(
                problem, minus, plus, values_minus, values_plus, stencil_size
            ),
            "algebraic_residual_relative": float(
                np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-14)
            ),
            "condition_proxy_equilibrated_1norm": float(condition),
            "degrees_of_freedom": int(solution.size),
            "n_minus": int(minus.nodes.shape[0]),
            "n_plus": int(plus.nodes.shape[0]),
            "n_interface_each_side": int(interface_count),
            "solve_seconds": float(time.perf_counter() - start),
        }
    )
    return MonolithicResult(
        problem=problem,
        minus=minus,
        plus=plus,
        values_minus=values_minus,
        values_plus=values_plus,
        matrix=matrix.tocsr(),
        matrix_raw=raw_matrix,
        rhs=rhs,
        condition_proxy=float(condition),
        solve_seconds=float(time.perf_counter() - start),
        metrics=diagnostics,
    )


def pilot_gradient_from_result(result: MonolithicResult) -> PilotGradient:
    return PilotGradient(
        result.problem,
        result.minus.nodes,
        result.values_minus,
        result.plus.nodes,
        result.values_plus,
    )


@dataclass
class LocalRobinSystem:
    """One subdomain factorization for a fixed Robin impedance."""

    cloud: SideCloud
    matrix: csr_matrix
    lu: Any
    rhs_base: np.ndarray
    row_scale: np.ndarray
    interface_rows: np.ndarray
    flux_operator: csr_matrix
    robin_lambda: np.ndarray
    condition_proxy: float

    def solve(self, interface_target: np.ndarray) -> np.ndarray:
        if interface_target.shape != self.robin_lambda.shape:
            raise ValueError("Interface target has the wrong size")
        rhs = self.rhs_base.copy()
        rhs[self.interface_rows] = self.row_scale[self.interface_rows] * interface_target
        value = self.lu.solve(rhs)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("Non-finite local Robin solution")
        return value

    def flux(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(self.flux_operator @ value, dtype=float).reshape(-1)

    def trace(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value[self.cloud.interface], dtype=float)


def assemble_local_robin_system(
    problem: InterfaceProblem,
    cloud: SideCloud,
    robin_lambda: np.ndarray,
    stencil_size: int = DEFAULT_STENCIL,
    compute_condition: bool = True,
) -> LocalRobinSystem:
    robin_lambda = np.asarray(robin_lambda, dtype=float).reshape(-1)
    if robin_lambda.size != cloud.interface.size or np.any(robin_lambda <= 0.0):
        raise ValueError("Robin impedance must be positive at every interface point")
    total = cloud.nodes.shape[0]
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    flux_rows: list[int] = []
    flux_columns: list[int] = []
    flux_data: list[float] = []
    rhs = np.zeros(total)
    row = 0
    for local_index in cloud.interior:
        center = cloud.nodes[local_index]
        stencil, weights = local_phs_weights(cloud, center, "laplacian", stencil_size=stencil_size)
        _append_weight_row(rows, columns, data, row, 0, stencil, -cloud.conductivity * weights)
        rhs[row] = problem.forcing(center[None, :], np.array([cloud.side]))[0]
        row += 1
    for local_index in cloud.boundary:
        rows.append(row)
        columns.append(int(local_index))
        data.append(1.0)
        point = cloud.nodes[local_index]
        rhs[row] = problem.exact(point[None, :], np.array([cloud.side]))[0]
        row += 1
    interface_rows = np.arange(row, row + cloud.interface.size, dtype=int)
    for position, local_index in enumerate(cloud.interface):
        point = cloud.nodes[local_index]
        outward = cloud.interface_normals_out[position]
        stencil, derivative = local_phs_weights(
            cloud, point, "directional", direction=outward, stencil_size=stencil_size
        )
        flux_weights = cloud.conductivity * derivative
        _append_weight_row(rows, columns, data, row, 0, stencil, flux_weights)
        rows.append(row)
        columns.append(int(local_index))
        data.append(float(robin_lambda[position]))
        flux_rows.extend([position] * stencil.size)
        flux_columns.extend(stencil.tolist())
        flux_data.extend(flux_weights.tolist())
        row += 1
    if row != total:
        raise RuntimeError(f"Local row-count defect: {row} != {total}")
    raw = coo_matrix((data, (rows, columns)), shape=(total, total)).tocsr()
    row_norm = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).reshape(-1))
    if np.any(row_norm <= 0.0):
        raise RuntimeError("Zero local Robin row norm")
    row_scale = 1.0 / row_norm
    matrix = (diags(row_scale) @ raw).tocsc()
    rhs_scaled = row_scale * rhs
    lu = splu(matrix)
    condition = _equilibrated_condition_proxy(matrix.tocsr(), lu) if compute_condition else float("nan")
    flux_operator = coo_matrix(
        (flux_data, (flux_rows, flux_columns)),
        shape=(cloud.interface.size, total),
    ).tocsr()
    return LocalRobinSystem(
        cloud=cloud,
        matrix=matrix.tocsr(),
        lu=lu,
        rhs_base=rhs_scaled,
        row_scale=row_scale,
        interface_rows=interface_rows,
        flux_operator=flux_operator,
        robin_lambda=robin_lambda,
        condition_proxy=float(condition),
    )


@dataclass
class SchwarzResult:
    minus_system: LocalRobinSystem
    plus_system: LocalRobinSystem
    values_minus: np.ndarray
    values_plus: np.ndarray
    history: list[dict[str, float]]
    metrics: dict[str, Any]


def run_two_way_robin_schwarz(
    problem: InterfaceProblem,
    interior_points: np.ndarray,
    boundary_per_edge: int,
    interface_count: int,
    lambda_minus: np.ndarray,
    lambda_plus: np.ndarray,
    relaxation: float,
    maximum_iterations: int,
    evaluation_resolution: int,
    collar_width: float,
    stencil_size: int = DEFAULT_STENCIL,
    compute_condition: bool = True,
) -> SchwarzResult:
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("Relaxation must lie in (0,1]")
    minus, plus = build_side_clouds(problem, interior_points, boundary_per_edge, interface_count)
    minus_system = assemble_local_robin_system(
        problem, minus, lambda_minus, stencil_size, compute_condition
    )
    plus_system = assemble_local_robin_system(
        problem, plus, lambda_plus, stencil_size, compute_condition
    )
    zero = np.zeros(interface_count)
    values_minus = minus_system.solve(zero)
    values_plus = plus_system.solve(zero)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration in range(maximum_iterations + 1):
        trace_minus = minus_system.trace(values_minus)
        trace_plus = plus_system.trace(values_plus)
        flux_minus = minus_system.flux(values_minus)
        flux_plus = plus_system.flux(values_plus)
        trace_rms = float(np.sqrt(np.mean((trace_minus - trace_plus) ** 2)))
        flux_rms = float(np.sqrt(np.mean((flux_minus + flux_plus) ** 2)))
        physical = float(math.hypot(trace_rms, flux_rms))
        update_rms = 0.0
        if iteration > 0:
            update_rms = float(
                math.sqrt(
                    np.mean((values_minus - previous_minus) ** 2)
                    + np.mean((values_plus - previous_plus) ** 2)
                )
            )
        history.append(
            {
                "iteration": float(iteration),
                "trace_rms": trace_rms,
                "flux_rms": flux_rms,
                "physical_mismatch": physical,
                "update_rms": update_rms,
            }
        )
        if iteration == maximum_iterations:
            break
        target_minus = -flux_plus + lambda_minus * trace_plus
        target_plus = -flux_minus + lambda_plus * trace_minus
        raw_minus = minus_system.solve(target_minus)
        raw_plus = plus_system.solve(target_plus)
        previous_minus = values_minus.copy()
        previous_plus = values_plus.copy()
        values_minus = (1.0 - relaxation) * values_minus + relaxation * raw_minus
        values_plus = (1.0 - relaxation) * values_plus + relaxation * raw_plus

    mismatch = np.asarray([row["physical_mismatch"] for row in history], dtype=float)
    safe = np.maximum(mismatch, 1.0e-300)
    ratios = safe[1:] / safe[:-1]
    q_net = float((safe[-1] / safe[0]) ** (1.0 / max(maximum_iterations, 1)))
    q_tail = float(np.median(ratios[-min(5, ratios.size):])) if ratios.size else float("inf")
    coordinate = np.linspace(0.0, 1.0, evaluation_resolution)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    evaluation = np.column_stack((xx.ravel(), yy.ravel()))
    numerical = evaluate_piecewise(
        problem, minus, plus, values_minus, values_plus, evaluation
    )
    exact = problem.exact(evaluation)
    error = numerical - exact
    collar = np.abs(problem.phi(evaluation)) <= collar_width
    exact_range = float(np.max(exact) - np.min(exact))
    overshoot = max(
        0.0,
        float(np.max(numerical) - np.max(exact)),
        float(np.min(exact) - np.min(numerical)),
    ) / max(exact_range, 1.0e-12)
    metrics = {
        "full_rmse": float(np.sqrt(np.mean(error**2))),
        "collar_rmse": float(np.sqrt(np.mean(error[collar] ** 2))),
        "full_linf": float(np.max(np.abs(error))),
        "collar_linf": float(np.max(np.abs(error[collar]))),
        "relative_overshoot": float(overshoot),
        "trace_rms_terminal": float(history[-1]["trace_rms"]),
        "flux_rms_terminal": float(history[-1]["flux_rms"]),
        "physical_mismatch_initial": float(history[0]["physical_mismatch"]),
        "physical_mismatch_terminal": float(history[-1]["physical_mismatch"]),
        "q_net": q_net,
        "q_tail": q_tail,
        "condition_proxy_minus": minus_system.condition_proxy,
        "condition_proxy_plus": plus_system.condition_proxy,
        "condition_proxy_max": max(minus_system.condition_proxy, plus_system.condition_proxy),
        "conservation_defect": conservation_defect(
            problem, minus, plus, values_minus, values_plus, stencil_size
        ),
        "degrees_of_freedom": int(minus.nodes.shape[0] + plus.nodes.shape[0]),
        "iterations": int(maximum_iterations),
        "iteration_seconds": float(time.perf_counter() - started),
    }
    return SchwarzResult(
        minus_system=minus_system,
        plus_system=plus_system,
        values_minus=values_minus,
        values_plus=values_plus,
        history=history,
        metrics=metrics,
    )


def scientific_manifest(root: Path, exclusions: Iterable[str] = ("scientific_manifest_sha256.csv",)) -> list[dict[str, Any]]:
    excluded = set(exclusions)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded and not path.name.endswith(".tmp"):
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def median(values: Iterable[float]) -> float:
    return float(np.median(np.asarray(list(values), dtype=float)))


def case_stem(problem: str, method: str, budget: int, seed: int) -> str:
    safe_problem = problem.lower().replace("_", "-")
    safe_method = method.lower().replace("_", "-")
    return f"{safe_problem}__{safe_method}__N{budget:05d}__seed{seed:03d}"
