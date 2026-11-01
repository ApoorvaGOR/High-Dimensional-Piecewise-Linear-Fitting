"""Numerically stable adapter around the supplied polynomial-time algorithms."""
import time
import numpy as np
from run_optimization_nonsparse import exact_multidimensional_slope_change
from run_optimization_sparse import exact_sparse_multidimensional_slope_change


def solve_exact(x, Y, k=None):
    """Preserve unweighted raw-unit SSE, returning coefficients in original units.

    Affine time rescaling and per-stream response translation leave the fitting
    problem unchanged. They avoid cancellation for years near 2000 and large
    constant sensor offsets. No per-stream variance scaling is performed.
    Interval polynomial diagnostics remain in the documented internal coordinates.
    """
    started = time.perf_counter()
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if x.ndim != 1 or Y.ndim != 2 or Y.shape[1] != len(x) or len(x) < 6:
        raise ValueError("Require x=(T,), Y=(J,T), T>=6")
    if not np.isfinite(x).all() or not np.isfinite(Y).all() or not np.all(np.diff(x) >= 0) or x[1] >= x[-2]:
        raise ValueError("Require finite observations, nondecreasing time, and nonempty interior domain")
    if k is not None and (int(k) != k or not 0 <= k <= len(Y)):
        raise ValueError("k must be an integer from 0 through J")
    origin, scale = float(x[0]), float(x[-1] - x[0])
    u = np.ascontiguousarray((x - origin) / scale)
    offset = Y.mean(axis=1)
    centered = np.ascontiguousarray(Y - offset[:, None])
    prep = time.perf_counter() - started
    t0 = time.perf_counter()
    if k is None:
        result = exact_multidimensional_slope_change(u, centered)
    else:
        result = exact_sparse_multidimensional_slope_change(u, centered, k)
    call_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    result["internal_breakpoint"] = result["shared_breakpoint"]
    result["shared_breakpoint"] = origin + scale * result["shared_breakpoint"]
    result["interval_best_breakpoints"] = origin + scale * result["interval_best_breakpoints"]
    for key in ("left_slope", "right_slope", "delta"):
        result[key] = result[key] / scale
    result["alpha"] = result["alpha"] + offset - result["left_slope"] * origin
    result["left_intercept"] = result["alpha"].copy()
    result["right_intercept"] = result["alpha"] - result["delta"] * result["shared_breakpoint"]
    result["continuity_value"] += offset
    result["selected_mask"] = result.get("selected_mask", np.ones(len(Y), dtype=bool))
    result["time_origin"] = origin
    result["time_scale"] = scale
    result["coordinate_preparation_seconds"] = prep
    result["coordinate_recovery_seconds"] = time.perf_counter() - t0
    result["algorithm_call_seconds"] = call_time
    result["end_to_end_seconds"] = time.perf_counter() - started
    return result


def fitted_values(x, result):
    return (result["alpha"][:, None] + result["left_slope"][:, None] * x
            + result["delta"][:, None] * np.maximum(0, x - result["shared_breakpoint"]))
