import argparse
import time
from pathlib import Path
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
MIN_SIDE_POINTS_DEFAULT = 2
POLY_TOL_DEFAULT = 1e-12
DENOM_TOL_DEFAULT = 1e-13

# =========================================================================
# MATHEMATICAL GUIDE: build_stream_statistics_numba
# Global sufficient statistics per stream j:
# sy[j]=sum_i y[j,i], sxy[j]=sum_i x[i]*y[j,i],
# syy[j]=sum_i y[j,i]^2, plus suffix sums of y and x*y.
# These evaluate C_j=y_j.T M_A y_j and nu_j(z) without M_A.
# =========================================================================
@njit(parallel=True)
def build_stream_statistics_numba(x, Y):
    """Compute stream-wide sums and right-side suffix sums in compiled code."""
    D_streams, T = Y.shape

    sy = np.zeros(D_streams, dtype=np.float64)
    sxy = np.zeros(D_streams, dtype=np.float64)
    syy = np.zeros(D_streams, dtype=np.float64)

    # suffix_y[d,k]   = sum_{i=k}^{T-1} y[d,i]
    # suffix_xy[d,k]  = sum_{i=k}^{T-1} x[i] y[d,i]
    suffix_y = np.zeros((D_streams, T + 1), dtype=np.float64)
    suffix_xy = np.zeros((D_streams, T + 1), dtype=np.float64)

    for d in prange(D_streams):
        a = 0.0
        b = 0.0
        c = 0.0
        for i in range(T):
            yi = Y[d, i]
            xi = x[i]
            a += yi
            b += xi * yi
            c += yi * yi
        sy[d] = a
        sxy[d] = b
        syy[d] = c

        run_y = 0.0
        run_xy = 0.0
        for i in range(T - 1, -1, -1):
            yi = Y[d, i]
            run_y += yi
            run_xy += x[i] * yi
            suffix_y[d, i] = run_y
            suffix_xy[d, i] = run_xy

    return sy, sxy, syy, suffix_y, suffix_xy


# =========================================================================
# MATHEMATICAL GUIDE: build_x_suffix_numba
# Construct sum_{i>=t} x_i and sum_{i>=t} x_i^2.
# For split t, p_i=x_i*1{i>t}, q_i=1{i>t}.
# =========================================================================
@njit
def build_x_suffix_numba(x):
    T = x.shape[0]
    suffix_x = np.zeros(T + 1, dtype=np.float64)
    suffix_xx = np.zeros(T + 1, dtype=np.float64)
    run_x = 0.0
    run_xx = 0.0
    for i in range(T - 1, -1, -1):
        xi = x[i]
        run_x += xi
        run_xx += xi * xi
        suffix_x[i] = run_x
        suffix_xx[i] = run_xx
    return suffix_x, suffix_xx


# =========================================================================
# MATHEMATICAL GUIDE: ma_inner_numba
# v.T M_A w=v.T w-[sum(v),sum(x*v)] G^{-1}
# [sum(w),sum(x*w)]^T, G=A.T A, A=[1,x].
# =========================================================================
@njit(inline="always")
def ma_inner_numba(vv, v1, vx, w1, wx, g00, g01, g11):
    """v^T M_A w where A=[1,x] and G^{-1}=[[g00,g01],[g01,g11]]."""
    return vv - (
        v1 * (g00 * w1 + g01 * wx)
        + vx * (g01 * w1 + g11 * wx)
    )


# =========================================================================
# MATHEMATICAL GUIDE: objective_from_coefficients_numba
# Dense objective F(z)=C-N(z)/D(z) (paper Eq. (13)),
# N(z)=sum_j nu_j(z)^2; D(z)=h(z).T M_A h(z).
# =========================================================================
@njit(inline="always")
def objective_from_coefficients_numba(
    z, C, n0, n1, n2, d0, d1, d2, denom_tol
):
    N = n0 + n1 * z + n2 * z * z
    Dval = d0 + d1 * z + d2 * z * z

    scale = 1.0
    a = abs(d0)
    if a > scale:
        scale = a
    a = abs(d1 * z)
    if a > scale:
        scale = a
    a = abs(d2 * z * z)
    if a > scale:
        scale = a

    # Exact theory: if D(z)=0, the objective at that point is C.
    # Here denom_tol is only a floating-point zero classifier.
    if Dval <= denom_tol * scale:
        return C

    value = C - N / Dval
    if value < 0.0 and value > -1e-8 * max(1.0, abs(C)):
        value = 0.0
    return value


# =========================================================================
# MATHEMATICAL GUIDE: consider_root_numba
# Evaluate an admissible root of R(z)=N'(z)D(z)-N(z)D'(z)
# inside the current interval and compare its objective.
# =========================================================================
@njit(inline="always")
def consider_root_numba(
    z,
    l,
    u,
    C,
    n0,
    n1,
    n2,
    d0,
    d1,
    d2,
    denom_tol,
    inside_tol,
    best_obj,
    best_z,
    count,
):
    """Evaluate one stationary root when it lies in/very near [l,u]."""
    if z < l - inside_tol or z > u + inside_tol:
        return best_obj, best_z, count

    if z < l:
        z = l
    elif z > u:
        z = u

    # Endpoints were already counted; do not count a root numerically equal
    # to either endpoint as an extra distinct candidate.
    if abs(z - l) <= inside_tol or abs(z - u) <= inside_tol:
        return best_obj, best_z, count

    obj = objective_from_coefficients_numba(
        z, C, n0, n1, n2, d0, d1, d2, denom_tol
    )
    count += 1
    if obj < best_obj:
        best_obj = obj
        best_z = z
    return best_obj, best_z, count


# =========================================================================
# MATHEMATICAL GUIDE: solve_all_intervals_exact_numba
# DENSE CORE (paper Section 3.1, Lemmas 1-4, Theorem 2).
# For each interval [x[t],x[t+1]], construct
# h(z)=p-z*q, p_i=x_i*1{i>t}, q_i=1{i>t}.
# D(z)=h(z).T M_A h(z)=d0+d1*z+d2*z^2,
# nu_j(z)=y_j.T M_A h(z)=s0[j]+s1[j]*z,
# N(z)=sum_j nu_j(z)^2=n0+n1*z+n2*z^2.
# Minimize C-N(z)/D(z) over endpoints and roots of
# R(z)=N'(z)D(z)-N(z)D'(z), then take best interval.
# Uses float64 candidate calculations, NOT exact symbolic arithmetic.
# =========================================================================
@njit(parallel=True)
def solve_all_intervals_exact_numba(
    x,
    sy,
    sxy,
    suffix_y,
    suffix_xy,
    suffix_x,
    suffix_xx,
    valid_t,
    C,
    g00,
    g01,
    g11,
    poly_tol,
    denom_tol,
):
    """
    Solve every original sampling interval exactly in the finite-candidate
    sense of the paper. Parallelism is across intervals.
    """
    D_streams = sy.shape[0]
    T = x.shape[0]
    M = valid_t.shape[0]

    interval_best_z = np.empty(M, dtype=np.float64)
    interval_best_obj = np.empty(M, dtype=np.float64)
    interval_num_candidates = np.empty(M, dtype=np.int64)
    interval_D_coeffs = np.empty((M, 3), dtype=np.float64)
    interval_N_coeffs = np.empty((M, 3), dtype=np.float64)
    interval_R_coeffs = np.empty((M, 3), dtype=np.float64)

    for pos in prange(M):
        t = valid_t[pos]
        right = t + 1
        l = x[t]
        u = x[t + 1]
        nr = float(T - right)

        # p_i = x_i 1{i>t}; q_i = 1{i>t}
        p1 = suffix_x[right]
        pxx = suffix_xx[right]
        q1 = nr
        qx = p1

        # D(z)=d0+d1 z+d2 z^2.
        d0 = ma_inner_numba(
            pxx, p1, pxx, p1, pxx, g00, g01, g11
        )
        pmq = ma_inner_numba(
            p1, p1, pxx, q1, qx, g00, g01, g11
        )
        qmq = ma_inner_numba(
            nr, q1, qx, q1, qx, g00, g01, g11
        )
        d1 = -2.0 * pmq
        d2 = qmq

        # N(z)=sum_j (s_{j,0}+s_{j,1}z)^2.
        n0 = 0.0
        n1_half = 0.0
        n2 = 0.0

        gp0 = g00 * p1 + g01 * pxx
        gp1 = g01 * p1 + g11 * pxx
        gq0 = g00 * q1 + g01 * qx
        gq1 = g01 * q1 + g11 * qx

        for d in range(D_streams):
            ytp = suffix_xy[d, right]
            ytq = suffix_y[d, right]

            s0 = ytp - (sy[d] * gp0 + sxy[d] * gp1)
            ymq = ytq - (sy[d] * gq0 + sxy[d] * gq1)
            s1 = -ymq

            n0 += s0 * s0
            n1_half += s0 * s1
            n2 += s1 * s1

        n1 = 2.0 * n1_half

        interval_D_coeffs[pos, 0] = d0
        interval_D_coeffs[pos, 1] = d1
        interval_D_coeffs[pos, 2] = d2
        interval_N_coeffs[pos, 0] = n0
        interval_N_coeffs[pos, 1] = n1
        interval_N_coeffs[pos, 2] = n2

        # R(z)=N'(z)D(z)-N(z)D'(z) = r0+r1 z+r2 z^2.
        r0 = n1 * d0 - n0 * d1
        r1 = 2.0 * (n2 * d0 - n0 * d2)
        r2 = n2 * d1 - n1 * d2

        interval_R_coeffs[pos, 0] = r0
        interval_R_coeffs[pos, 1] = r1
        interval_R_coeffs[pos, 2] = r2

        # Endpoints are always candidates.
        best_z = l
        best_obj = objective_from_coefficients_numba(
            l, C, n0, n1, n2, d0, d1, d2, denom_tol
        )
        obj_u = objective_from_coefficients_numba(
            u, C, n0, n1, n2, d0, d1, d2, denom_tol
        )
        if obj_u < best_obj:
            best_obj = obj_u
            best_z = u
        count = 2

        interval_scale = max(1.0, abs(l), abs(u))
        inside_tol = 100.0 * poly_tol * interval_scale

        coeff_scale = max(1.0, abs(r0), abs(r1), abs(r2))
        eps = poly_tol * coeff_scale

        if abs(r2) <= eps:
            if abs(r1) > eps:
                root = -r0 / r1
                best_obj, best_z, count = consider_root_numba(
                    root, l, u, C, n0, n1, n2, d0, d1, d2,
                    denom_tol, inside_tol, best_obj, best_z, count
                )
        else:
            disc = r1 * r1 - 4.0 * r2 * r0
            disc_scale = max(1.0, r1 * r1, abs(4.0 * r2 * r0))
            disc_eps = poly_tol * disc_scale

            if disc >= -disc_eps:
                if abs(disc) <= disc_eps:
                    root = -r1 / (2.0 * r2)
                    best_obj, best_z, count = consider_root_numba(
                        root, l, u, C, n0, n1, n2, d0, d1, d2,
                        denom_tol, inside_tol, best_obj, best_z, count
                    )
                else:
                    sqrt_disc = np.sqrt(disc)
                    # Stable quadratic formula.
                    if r1 >= 0.0:
                        q = -0.5 * (r1 + sqrt_disc)
                    else:
                        q = -0.5 * (r1 - sqrt_disc)

                    root1 = q / r2
                    if q != 0.0:
                        root2 = r0 / q
                    else:
                        root2 = (-r1 - sqrt_disc) / (2.0 * r2)

                    best_obj, best_z, count = consider_root_numba(
                        root1, l, u, C, n0, n1, n2, d0, d1, d2,
                        denom_tol, inside_tol, best_obj, best_z, count
                    )
                    # Avoid double-counting a repeated/root-near-root case.
                    if abs(root2 - root1) > inside_tol:
                        best_obj, best_z, count = consider_root_numba(
                            root2, l, u, C, n0, n1, n2, d0, d1, d2,
                            denom_tol, inside_tol, best_obj, best_z, count
                        )

        interval_best_z[pos] = best_z
        interval_best_obj[pos] = best_obj
        interval_num_candidates[pos] = count

    return (
        interval_best_z,
        interval_best_obj,
        interval_num_candidates,
        interval_D_coeffs,
        interval_N_coeffs,
        interval_R_coeffs,
    )


# =========================================================================
# MATHEMATICAL GUIDE: recover_solution_numba
# Dense recovery: delta_j=nu_j(z*)/D(z*),
# [alpha_j,beta_j]^T=G^{-1} A.T(y_j-delta_j*h(z*)).
# Fitted value alpha_j+beta_j*x+delta_j*max(0,x-z*).
# Pre-slope beta_j; post-slope beta_j+delta_j;
# post-intercept alpha_j-delta_j*z*.
# =========================================================================
@njit(parallel=True)
def recover_solution_numba(
    x,
    Y,
    sy,
    sxy,
    syy,
    suffix_y,
    suffix_xy,
    suffix_x,
    suffix_xx,
    best_t,
    best_z,
    d0,
    d1,
    d2,
    g00,
    g01,
    g11,
    denom_tol,
):
    """Recover all stream coefficients at the selected global breakpoint."""
    D_streams, T = Y.shape
    right = best_t + 1
    p1 = suffix_x[right]
    pxx = suffix_xx[right]
    q1 = float(T - right)
    qx = p1

    D_best = d0 + d1 * best_z + d2 * best_z * best_z
    D_scale = max(1.0, abs(d0), abs(d1 * best_z), abs(d2 * best_z * best_z))
    use_hinge = D_best > denom_tol * D_scale

    gp0 = g00 * p1 + g01 * pxx
    gp1 = g01 * p1 + g11 * pxx
    gq0 = g00 * q1 + g01 * qx
    gq1 = g01 * q1 + g11 * qx

    sum_h = p1 - best_z * q1
    sum_xh = pxx - best_z * qx

    Cj = np.empty(D_streams, dtype=np.float64)
    losses = np.empty(D_streams, dtype=np.float64)
    alpha = np.empty(D_streams, dtype=np.float64)
    delta = np.empty(D_streams, dtype=np.float64)
    left_slope = np.empty(D_streams, dtype=np.float64)
    right_slope = np.empty(D_streams, dtype=np.float64)
    left_intercept = np.empty(D_streams, dtype=np.float64)
    right_intercept = np.empty(D_streams, dtype=np.float64)
    continuity_value = np.empty(D_streams, dtype=np.float64)

    for d in prange(D_streams):
        # C_j = y_j^T M_A y_j.
        cj = syy[d] - (
            sy[d] * (g00 * sy[d] + g01 * sxy[d])
            + sxy[d] * (g01 * sy[d] + g11 * sxy[d])
        )
        if cj < 0.0 and cj > -1e-8 * max(1.0, syy[d]):
            cj = 0.0
        Cj[d] = cj

        ytp = suffix_xy[d, right]
        ytq = suffix_y[d, right]
        s0 = ytp - (sy[d] * gp0 + sxy[d] * gp1)
        ymq = ytq - (sy[d] * gq0 + sxy[d] * gq1)
        s1 = -ymq
        nu = s0 + s1 * best_z

        if use_hinge:
            dj = nu / D_best
            loss = cj - (nu * nu) / D_best
        else:
            dj = 0.0
            loss = cj

        if loss < 0.0 and loss > -1e-8 * max(1.0, cj):
            loss = 0.0

        rhs0 = sy[d] - dj * sum_h
        rhs1 = sxy[d] - dj * sum_xh
        a = g00 * rhs0 + g01 * rhs1
        b = g01 * rhs0 + g11 * rhs1

        losses[d] = loss
        alpha[d] = a
        delta[d] = dj
        left_slope[d] = b
        right_slope[d] = b + dj
        left_intercept[d] = a
        right_intercept[d] = a - dj * best_z
        continuity_value[d] = a + b * best_z

    return (
        Cj,
        losses,
        alpha,
        delta,
        left_slope,
        right_slope,
        left_intercept,
        right_intercept,
        continuity_value,
    )


# =========================================================================
# MATHEMATICAL GUIDE: warm_up_exact_numba
# Compile Numba kernels once, separately from timed optimization.
# =========================================================================
def warm_up_exact_numba():
    """Compile kernels before timing the actual experiment."""
    x = np.arange(8, dtype=np.float64)
    Y = np.ascontiguousarray(
        np.vstack((2.0 + x, 4.0 - 0.5 * x)), dtype=np.float64
    )
    sy, sxy, syy, suffix_y, suffix_xy = build_stream_statistics_numba(x, Y)
    suffix_x, suffix_xx = build_x_suffix_numba(x)

    T = len(x)
    sx = float(np.sum(x))
    sxx = float(np.dot(x, x))
    det = T * sxx - sx * sx
    g00 = sxx / det
    g01 = -sx / det
    g11 = T / det
    C = 0.0
    for d in range(Y.shape[0]):
        C += syy[d] - (
            sy[d] * (g00 * sy[d] + g01 * sxy[d])
            + sxy[d] * (g01 * sy[d] + g11 * sxy[d])
        )

    valid_t = np.arange(1, T - 2, dtype=np.int64)
    ans = solve_all_intervals_exact_numba(
        x, sy, sxy, suffix_y, suffix_xy, suffix_x, suffix_xx,
        valid_t, C, g00, g01, g11, 1e-12, 1e-13
    )
    pos = int(np.argmin(ans[1]))
    t = int(valid_t[pos])
    z = float(ans[0][pos])
    d0, d1, d2 = ans[3][pos]
    recover_solution_numba(
        x, Y, sy, sxy, syy, suffix_y, suffix_xy, suffix_x, suffix_xx,
        t, z, d0, d1, d2, g00, g01, g11, 1e-13
    )


# =========================================================================
# MATHEMATICAL GUIDE: exact_multidimensional_slope_change
# Solve the non-sparse case k=J of paper Eq. (1), under squared
# l2 loss. Every stream may have nonzero hinge coefficient delta_j.
# Return global breakpoint, objective, recovered coefficients, diagnostics.
# =========================================================================
def exact_multidimensional_slope_change(
    x,
    Y,
    min_side_points=MIN_SIDE_POINTS_DEFAULT,
    poly_tol=POLY_TOL_DEFAULT,
    denom_tol=DENOM_TOL_DEFAULT,
):
    """
    Globally solve the NON-SPARSE shared-breakpoint continuous PWL problem.

    No discretization, recursive bisection, or target optimality gap is used.
    For each original interval [x_t,x_{t+1}], only its endpoints and the
    at-most-two real stationary roots of the quadratic R_t are checked.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    D_streams, T = Y.shape

    if x.ndim != 1 or len(x) != T:
        raise ValueError("x must be one-dimensional with len(x) == Y.shape[1].")
    if not np.isfinite(x).all() or not np.isfinite(Y).all() or not np.all(np.diff(x) >= 0.0):
        raise ValueError("x must be finite and nondecreasing; Y must be finite.")
    if min_side_points < 1 or 2 * min_side_points > T:
        raise ValueError("Invalid min_side_points.")

    # G^{-1} for A=[1,x], written explicitly to avoid a tiny np.linalg call.
    sx = float(np.sum(x))
    sxx = float(np.dot(x, x))
    det = float(T) * sxx - sx * sx
    if det <= 0.0:
        raise ValueError("A=[1,x] is rank deficient.")
    g00 = sxx / det
    g01 = -sx / det
    g11 = float(T) / det

    stats_start = time.perf_counter()
    sy, sxy, syy, suffix_y, suffix_xy = build_stream_statistics_numba(x, Y)
    suffix_x, suffix_xx = build_x_suffix_numba(x)
    statistics_runtime = time.perf_counter() - stats_start

    # C = sum_j y_j^T M_A y_j.
    projected = (
        g00 * sy * sy
        + 2.0 * g01 * sy * sxy
        + g11 * sxy * sxy
    )
    Cj_python = syy - projected
    tiny = (Cj_python < 0.0) & (Cj_python > -1e-8 * np.maximum(1.0, syy))
    Cj_python[tiny] = 0.0
    C = float(np.sum(Cj_python))

    first_t = min_side_points - 1
    last_t = T - min_side_points - 1
    valid_t = np.arange(first_t, last_t + 1, dtype=np.int64)
    valid_t = valid_t[x[valid_t + 1] > x[valid_t]]
    if not len(valid_t):
        raise ValueError("No positive-width admissible breakpoint intervals.")

    solve_start = time.perf_counter()
    (
        interval_best_z,
        interval_best_obj,
        interval_num_candidates,
        interval_D_coeffs,
        interval_N_coeffs,
        interval_R_coeffs,
    ) = solve_all_intervals_exact_numba(
        x=x,
        sy=sy,
        sxy=sxy,
        suffix_y=suffix_y,
        suffix_xy=suffix_xy,
        suffix_x=suffix_x,
        suffix_xx=suffix_xx,
        valid_t=valid_t,
        C=C,
        g00=g00,
        g01=g01,
        g11=g11,
        poly_tol=poly_tol,
        denom_tol=denom_tol,
    )
    solve_runtime = time.perf_counter() - solve_start

    best_pos = int(np.argmin(interval_best_obj))
    best_t = int(valid_t[best_pos])
    best_z = float(interval_best_z[best_pos])
    best_obj = float(interval_best_obj[best_pos])

    d0, d1, d2 = interval_D_coeffs[best_pos]
    recover_start = time.perf_counter()
    (
        Cj,
        losses_by_dimension,
        alpha,
        delta,
        left_slope,
        right_slope,
        left_intercept,
        right_intercept,
        continuity_value,
    ) = recover_solution_numba(
        x=x,
        Y=Y,
        sy=sy,
        sxy=sxy,
        syy=syy,
        suffix_y=suffix_y,
        suffix_xy=suffix_xy,
        suffix_x=suffix_x,
        suffix_xx=suffix_xx,
        best_t=best_t,
        best_z=best_z,
        d0=d0,
        d1=d1,
        d2=d2,
        g00=g00,
        g01=g01,
        g11=g11,
        denom_tol=denom_tol,
    )
    recovery_runtime = time.perf_counter() - recover_start

    return {
        "valid_t": valid_t,
        "interval_best_breakpoints": interval_best_z,
        "interval_optimal_values": interval_best_obj,
        "interval_num_candidates": interval_num_candidates,
        "interval_D_coefficients": interval_D_coeffs,
        "interval_N_coefficients": interval_N_coeffs,
        "interval_R_coefficients": interval_R_coeffs,
        "C": C,
        "Cj": Cj,
        "best_t": best_t,
        "best_position": best_pos,
        "shared_breakpoint": best_z,
        "optimal_loss": best_obj,
        "losses_by_dimension": losses_by_dimension,
        "alpha": alpha,
        "delta": delta,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "left_intercept": left_intercept,
        "right_intercept": right_intercept,
        "continuity_value": continuity_value,
        "statistics_runtime": statistics_runtime,
        "interval_solve_runtime": solve_runtime,
        "recovery_runtime": recovery_runtime,
        "solve_runtime": statistics_runtime + solve_runtime + recovery_runtime,
    }


# =========================================================================
# MATHEMATICAL GUIDE: direct_continuous_fit_loss
# Cross-check at fixed breakpoint z using explicit least squares with
# columns [1,x,max(0,x-z)] for each stream.
# =========================================================================
def direct_continuous_fit_loss(x, Y, r):
    """Independent least-squares check using the 3-column hinge design."""
    left = np.minimum(x - r, 0.0)
    right = np.maximum(x - r, 0.0)
    X = np.column_stack([np.ones_like(x), left, right])
    theta, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)
    residual = Y.T - X @ theta
    losses = np.sum(residual * residual, axis=0)
    return float(np.sum(losses)), losses, theta.T



# =========================================================================
# MATHEMATICAL GUIDE: write_excel
# Export model results and numerical checks; does not alter the solution.
# =========================================================================
def write_excel(path, x, Y, dates, tickers, result, warmup, validation_time, validation):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    summary = {
        "D": int(Y.shape[0]), "T": int(Y.shape[1]),
        "best_t_zero_based": result["best_t"],
        "shared_breakpoint": result["shared_breakpoint"],
        "optimal_SSE": result["optimal_loss"],
        "numba_warmup_seconds_EXCLUDED": warmup,
        "statistics_seconds": result["statistics_runtime"],
        "interval_solve_seconds": result["interval_solve_runtime"],
        "coefficient_recovery_seconds": result["recovery_runtime"],
        "total_numerical_seconds_EXCLUDES_warmup": result["solve_runtime"],
        "validation_seconds": validation_time,
        "validation_direct_SSE": validation["direct_loss"] if validation else None,
        "validation_absolute_difference": validation["absolute_difference"] if validation else None,
        "validation_relative_difference": validation["relative_difference"] if validation else None,
    }
    ws.append(["Metric", "Value"])
    for key, value in summary.items():
        ws.append([key, value])
    wi = wb.create_sheet("Intervals")
    wi.append(["t_zero_based", "left_x", "right_x", "left_date", "right_date",
               "best_breakpoint", "best_SSE", "candidates_checked",
               "d0", "d1", "d2", "n0", "n1", "n2", "r0", "r1", "r2"])
    for pos, t in enumerate(result["valid_t"]):
        wi.append([int(t), float(x[t]), float(x[t+1]), str(dates[t]), str(dates[t+1]),
                   float(result["interval_best_breakpoints"][pos]),
                   float(result["interval_optimal_values"][pos]),
                   int(result["interval_num_candidates"][pos]),
                   *[float(v) for v in result["interval_D_coefficients"][pos]],
                   *[float(v) for v in result["interval_N_coefficients"][pos]],
                   *[float(v) for v in result["interval_R_coefficients"][pos]]])
    wf = wb.create_sheet("Fitted streams")
    wf.append(["ticker", "SSE", "left_slope", "right_slope", "delta_slope",
               "left_intercept", "right_intercept", "continuity_value"])
    for j, ticker in enumerate(tickers):
        wf.append([str(ticker), float(result["losses_by_dimension"][j]),
                   float(result["left_slope"][j]), float(result["right_slope"][j]),
                   float(result["delta"][j]), float(result["left_intercept"][j]),
                   float(result["right_intercept"][j]), float(result["continuity_value"][j])])
    for sheet in wb:
        sheet.freeze_panes = "B2"
        sheet.sheet_view.showGridLines = False
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="183153")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True)
        sheet.row_dimensions[1].height = 30
        for col in sheet.columns:
            letter = get_column_letter(col[0].column)
            sheet.column_dimensions[letter].width = min(42, max(14, max(len(str(c.value or "")) for c in list(col)[:150]) + 2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000000000'
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# =========================================================================
# MATHEMATICAL GUIDE: main
# Load simulated data, warm up, optimize dense model, validate and export.
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run non-sparse exact optimization on saved synthetic data")
    parser.add_argument("--input", default="synthetic_stock_data.npz")
    parser.add_argument("--output", default="exact_nonsparse_results.xlsx")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.threads is not None:
        set_num_threads(args.threads)
    with np.load(args.input, allow_pickle=False) as data:
        x, Y = data["x"].copy(), data["Y"].copy()
        dates, tickers = data["dates"].copy(), data["tickers"].copy()
    t0 = time.perf_counter()
    warm_up_exact_numba()
    warmup = time.perf_counter() - t0
    result = exact_multidimensional_slope_change(x, Y)
    validation = None
    validation_time = 0.0
    if not args.skip_validation:
        t0 = time.perf_counter()
        loss, _, _ = direct_continuous_fit_loss(x, Y, result["shared_breakpoint"])
        validation_time = time.perf_counter() - t0
        validation = {"direct_loss": loss, "absolute_difference": abs(loss-result["optimal_loss"]),
                      "relative_difference": abs(loss-result["optimal_loss"])/max(1.0, abs(result["optimal_loss"]))}
    write_excel(Path(args.output), x, Y, dates, tickers, result, warmup, validation_time, validation)
    print(f"D={Y.shape[0]}, T={Y.shape[1]}; breakpoint={result['shared_breakpoint']:.12g}; SSE={result['optimal_loss']:.12g}")
    print(f"Numerical runtime={result['solve_runtime']:.4f}s; Numba warmup={warmup:.4f}s (excluded)")
    print(f"Excel results saved to {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
