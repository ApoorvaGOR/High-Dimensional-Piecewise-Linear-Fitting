import argparse
import os
import json
import time
from pathlib import Path
# Apple Accelerate is not controlled by threadpoolctl. Set before NumPy loads.
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from numba import njit, get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

POLY_TOL_DEFAULT = 1e-12
DENOM_TOL_DEFAULT = 1e-13
MIN_SIDE_POINTS_DEFAULT = 2

# =========================================================================
# MATHEMATICAL GUIDE: _ma_inner_from_summaries
# Projector identity (paper Section 3.1):
# A=[1,x], G=A.T@A, M_A=I-A G^{-1} A.T.
# v.T M_A w = v.T w - [sum(v),sum(x*v)] G^{-1}
#                             [sum(w),sum(x*w)].
# Avoids allocating the T-by-T projector M_A.
# =========================================================================
def _ma_inner_from_summaries(vv, v1, vx, ww, w1, wx, G_inv):
    """Return v^T M_A w from sufficient statistics."""
    left = np.asarray([v1, vx], dtype=np.float64)
    right = np.asarray([w1, wx], dtype=np.float64)
    return float(vv - left @ G_inv @ right)


# =========================================================================
# MATHEMATICAL GUIDE: _quadratic_real_roots
# Numerically solve a*z^2+b*z+c=0, including linear/degenerate cases.
# Uses float64 tolerance and a cancellation-resistant quadratic formula;
# this is NOT exact algebraic root isolation as in the theoretical proof.
# =========================================================================
def _quadratic_real_roots(a, b, c, tol=POLY_TOL_DEFAULT):
    """Robust real roots of a*z^2 + b*z + c = 0 (float64)."""
    scale = max(1.0, abs(a), abs(b), abs(c))
    eps = tol * scale

    if abs(a) <= eps:
        if abs(b) <= eps:
            return []
        return [-c / b]

    disc = b * b - 4.0 * a * c
    disc_scale = max(1.0, b * b, abs(4.0 * a * c))
    disc_eps = tol * disc_scale
    if disc < -disc_eps:
        return []
    if abs(disc) <= disc_eps:
        return [-b / (2.0 * a)]

    sqrt_disc = np.sqrt(max(0.0, disc))

    # Stable quadratic formula.
    q = -0.5 * (b + np.copysign(sqrt_disc, b))
    r1 = q / a
    r2 = c / q if q != 0.0 else (-b - sqrt_disc) / (2.0 * a)
    return [r1, r2]


# =========================================================================
# MATHEMATICAL GUIDE: _objective_from_coefficients
# Fixed-support least-squares objective (paper Eq. (13)/(21)):
# F_S(z) = C - N_S(z)/D(z),
# N_S(z)=n0+n1*z+n2*z^2, D(z)=d0+d1*z+d2*z^2.
# C=sum_j ||M_A y_j||^2 is the one-line baseline SSE.
# At D(z)=0 the theoretical fixed-z gain is zero; this implementation
# uses a scale-relative tolerance to treat small denominators as zero.
# =========================================================================
def _objective_from_coefficients(z, C, n0, n1, n2, d0, d1, d2, denom_tol):
    N = n0 + n1 * z + n2 * z * z
    D = d0 + d1 * z + d2 * z * z

    scale = max(1.0, abs(d0), abs(d1 * z), abs(d2 * z * z))
    if D <= denom_tol * scale:
        # At an exact zero of D, Lemma 1 gives value C.
        if D < -100.0 * denom_tol * scale:
            raise FloatingPointError(f"D(z) became materially negative: {D}")
        return float(C)

    value = C - N / D
    if value < 0.0 and value > -1e-8 * max(1.0, abs(C)):
        value = 0.0
    return float(value)


# =========================================================================
# MATHEMATICAL GUIDE: _interval_candidates
# Stationarity (paper Lemma 3): minimize C-N_S(z)/D(z) on [l,u].
# Derivative = -R(z)/D(z)^2, R=N_S' D-N_S D'.
# R(z)=r0+r1*z+r2*z^2 with
# r0=n1*d0-n0*d1, r1=2*(n2*d0-n0*d2),
# r2=n2*d1-n1*d2. Check both endpoints and interior real roots.
# =========================================================================
def _interval_candidates(l, u, n0, n1, n2, d0, d1, d2, poly_tol):
    """Endpoints plus all stationary roots of N/D inside (l,u)."""
    # R = N' D - N D', degree <= 2.
    r0 = n1 * d0 - n0 * d1
    r1 = 2.0 * (n2 * d0 - n0 * d2)
    r2 = n2 * d1 - n1 * d2

    candidates = [float(l), float(u)]
    roots = _quadratic_real_roots(r2, r1, r0, tol=poly_tol)

    interval_scale = max(1.0, abs(l), abs(u))
    inside_tol = 100.0 * poly_tol * interval_scale
    for z in roots:
        if l + inside_tol < z < u - inside_tol:
            candidates.append(float(z))
        elif l - inside_tol <= z <= u + inside_tol:
            # Root numerically coincident with an endpoint.
            candidates.append(float(np.clip(z, l, u)))

    # Small fixed candidate set; deduplicate numerically.
    candidates.sort()
    unique = []
    for z in candidates:
        if not unique or abs(z - unique[-1]) > inside_tol:
            unique.append(z)
    return unique, (r0, r1, r2)


def _unique_sorted_points(points, tol):
    """Sort points and remove numerical duplicates."""
    points = sorted(float(v) for v in points)
    if not points:
        return []
    out = [points[0]]
    for value in points[1:]:
        if abs(value - out[-1]) > tol:
            out.append(value)
    return out


# =========================================================================
# MATHEMATICAL GUIDE: _gain_crossings_on_interval
# Support ranking (paper Eq. (22), Lemma 13):
# nu_j(z)=s0[j]+s1[j]*z; gain_j(z)=nu_j(z)^2/D(z).
# Common D(z)>0 means rankings change only where nu_i=+/-nu_j.
# These two affine equations yield at most two crossings per pair.
# =========================================================================
def _gain_crossings_on_interval(l, u, s0, s1, poly_tol):
    """
    Return all pairwise crossing points in (l,u) at which

        (s0[i] + s1[i] z)^2 = (s0[j] + s1[j] z)^2.

    Equivalently, crossings are roots of either

        nu_i(z) - nu_j(z) = 0
    or  nu_i(z) + nu_j(z) = 0.

    This is the O(J^2) partition from Lemma 6 of the note.
    """
    J = len(s0)
    interval_scale = max(1.0, abs(l), abs(u))
    inside_tol = 100.0 * poly_tol * interval_scale

    crossings = []

    for i in range(J - 1):
        a0_i = float(s0[i])
        a1_i = float(s1[i])

        for j in range(i + 1, J):
            a0_j = float(s0[j])
            a1_j = float(s1[j])

            # nu_i - nu_j = 0
            b0 = a0_i - a0_j
            b1 = a1_i - a1_j
            scale = max(1.0, abs(b0), abs(b1))
            if abs(b1) > poly_tol * scale:
                z = -b0 / b1
                if l + inside_tol < z < u - inside_tol:
                    crossings.append(float(z))

            # nu_i + nu_j = 0
            b0 = a0_i + a0_j
            b1 = a1_i + a1_j
            scale = max(1.0, abs(b0), abs(b1))
            if abs(b1) > poly_tol * scale:
                z = -b0 / b1
                if l + inside_tol < z < u - inside_tol:
                    crossings.append(float(z))

    return _unique_sorted_points(crossings, inside_tol)


# =========================================================================
# MATHEMATICAL GUIDE: _top_k_support_at_z
# For fixed z with D(z)>0, select the k largest nu_j(z)^2.
# This maximizes sum_{j in S} gain_j(z) without enumerating subsets.
# =========================================================================
def _top_k_support_at_z(s0, s1, z, k):
    """Indices of the k largest squared affine numerators nu_j(z)^2."""
    if k == 0:
        return np.empty(0, dtype=np.int64)

    nu = s0 + s1 * z
    scores = nu * nu

    if k == len(scores):
        # Deterministic ordering is useful for reproducible output.
        return np.arange(len(scores), dtype=np.int64)

    # argpartition avoids a full O(J log J) sort when only the top k are needed.
    support = np.argpartition(scores, -k)[-k:]

    # Deterministic tie/order handling for saved output.
    support = support[np.lexsort((support, -scores[support]))]
    return np.asarray(support, dtype=np.int64)


# =========================================================================
# MATHEMATICAL GUIDE: _numerator_coefficients_for_support
# N_S(z)=sum_{j in S}(s0[j]+s1[j]*z)^2.
# Thus n0=sum s0^2, n1=2 sum s0*s1, n2=sum s1^2.
# =========================================================================
def _numerator_coefficients_for_support(s0, s1, support):
    """Coefficients of N_S(z)=sum_{j in S}(s0_j+s1_j z)^2."""
    if len(support) == 0:
        return 0.0, 0.0, 0.0

    a = s0[support]
    b = s1[support]
    n0 = float(np.dot(a, a))
    n1 = float(2.0 * np.dot(a, b))
    n2 = float(np.dot(b, b))
    return n0, n1, n2


# =========================================================================
# MATHEMATICAL GUIDE: _objective_numba
# Numba implementation of F_S(z)=C-N_S(z)/D(z); see paper Eq. (21).
# Near-zero denominator handling is a floating-point convention.
# =========================================================================
@njit(inline="always")
def _objective_numba(z, C, n0, n1, n2, d0, d1, d2, denom_tol):
    N = n0 + n1 * z + n2 * z * z
    D = d0 + d1 * z + d2 * z * z
    scale = max(1.0, abs(d0), abs(d1 * z), abs(d2 * z * z))
    if D <= denom_tol * scale:
        return C
    value = C - N / D
    if value < 0.0 and value > -1e-8 * max(1.0, abs(C)):
        value = 0.0
    return value


# =========================================================================
# MATHEMATICAL GUIDE: _best_ratio_candidate_numba
# Solve the ratio-of-quadratics problem on a fixed support region:
# endpoints plus real roots of N_S' D-N_S D'=0 (paper Lemma 3).
# =========================================================================
@njit(inline="always")
def _best_ratio_candidate_numba(a, b, C, n0, n1, n2, d0, d1, d2, poly_tol, denom_tol):
    """Return the best endpoint/stationary candidate on [a,b]."""
    best_z = a
    best_obj = _objective_numba(a, C, n0, n1, n2, d0, d1, d2, denom_tol)

    obj_b = _objective_numba(b, C, n0, n1, n2, d0, d1, d2, denom_tol)
    if obj_b < best_obj:
        best_obj = obj_b
        best_z = b

    # R(z)=N'(z)D(z)-N(z)D'(z)=r2*z^2+r1*z+r0.
    r0 = n1 * d0 - n0 * d1
    r1 = 2.0 * (n2 * d0 - n0 * d2)
    r2 = n2 * d1 - n1 * d2

    scale = max(1.0, abs(r0), abs(r1), abs(r2))
    eps = poly_tol * scale
    interval_scale = max(1.0, abs(a), abs(b))
    inside_tol = 100.0 * poly_tol * interval_scale

    if abs(r2) <= eps:
        if abs(r1) > eps:
            z = -r0 / r1
            if a - inside_tol <= z <= b + inside_tol:
                z = min(max(z, a), b)
                obj = _objective_numba(z, C, n0, n1, n2, d0, d1, d2, denom_tol)
                if obj < best_obj:
                    best_obj = obj
                    best_z = z
        return best_z, best_obj

    disc = r1 * r1 - 4.0 * r2 * r0
    disc_scale = max(1.0, r1 * r1, abs(4.0 * r2 * r0))
    disc_eps = poly_tol * disc_scale
    if disc < -disc_eps:
        return best_z, best_obj

    if abs(disc) <= disc_eps:
        z = -r1 / (2.0 * r2)
        if a - inside_tol <= z <= b + inside_tol:
            z = min(max(z, a), b)
            obj = _objective_numba(z, C, n0, n1, n2, d0, d1, d2, denom_tol)
            if obj < best_obj:
                best_obj = obj
                best_z = z
        return best_z, best_obj

    sqrt_disc = np.sqrt(max(0.0, disc))
    q = -0.5 * (r1 + np.copysign(sqrt_disc, r1))
    z1 = q / r2
    if q != 0.0:
        z2 = r0 / q
    else:
        z2 = (-r1 - sqrt_disc) / (2.0 * r2)

    if a - inside_tol <= z1 <= b + inside_tol:
        z = min(max(z1, a), b)
        obj = _objective_numba(z, C, n0, n1, n2, d0, d1, d2, denom_tol)
        if obj < best_obj:
            best_obj = obj
            best_z = z
    if a - inside_tol <= z2 <= b + inside_tol:
        z = min(max(z2, a), b)
        obj = _objective_numba(z, C, n0, n1, n2, d0, d1, d2, denom_tol)
        if obj < best_obj:
            best_obj = obj
            best_z = z

    return best_z, best_obj


# =========================================================================
# MATHEMATICAL GUIDE: _ma_inner_numba
# Compute v.T M_A w from scalar sufficient statistics, no matrix M_A.
# =========================================================================
@njit(inline="always")
def _ma_inner_numba(vv, v1, vx, w1, wx, g00, g01, g10, g11):
    return vv - (
        v1 * (g00 * w1 + g01 * wx)
        + vx * (g10 * w1 + g11 * wx)
    )


# =========================================================================
# MATHEMATICAL GUIDE: _solve_sparse_intervals_numba
# SPARSE CORE (paper Section 4.1, Lemmas 12-13, Theorem 3).
# For each original interval [x[t],x[t+1]], form h(z)=p-z*q with
# p_i=x_i*1{i>t}, q_i=1{i>t}.
# Compute D(z)=h(z).T M_A h(z)=d0+d1*z+d2*z^2 and
# nu_j(z)=y_j.T M_A h(z)=s0[j]+s1[j]*z.
# Partition interval at nu_i(z)=+/-nu_j(z). Within each open region,
# the k largest nu_j(z)^2 have fixed support S; minimize
# C-[sum_{j in S}nu_j(z)^2]/D(z) via endpoints/stationary roots.
# Track changes of S as crossing groups are processed, then choose the
# smallest objective over all regions and original intervals.
# The proof uses exact rational/algebraic arithmetic; this code uses
# float64, sorting and tolerances, so 'exact' means finite-candidate
# optimization numerically, not a formal exact-arithmetic guarantee.
# =========================================================================
@njit
def _solve_sparse_intervals_numba(
    x,
    valid_t,
    sy,
    sxy,
    suffix_x,
    suffix_xx,
    suffix_y,
    suffix_xy,
    g00,
    g01,
    g10,
    g11,
    C,
    k,
    poly_tol,
    denom_tol,
):
    """
    Exact sparse solve using the sweep in Theorem 2 / Lemma 6.

    Key implementation points
    -------------------------
    * Pairwise crossings are generated only once per original interval.
    * We do NOT recompute a top-k ordering on every support region.
    * One full ordering is computed at the leftmost open region.
    * At a crossing, only streams participating in that crossing can change
      relative order.  Their ranks are updated using a probe point immediately
      to the right of the whole coincident crossing group.
    * The coefficients of N_S(z) are maintained incrementally: when a stream
      enters/leaves the top-k support, its quadratic contribution is added or
      removed in O(1).
    * Crossing storage is allocated with an exact two-pass count rather than
      J*(J-1)+1 worst-case storage per thread.

    Thus, after crossing generation/sorting, the region sweep no longer carries
    an O(J log J) top-k recomputation at every region.
    """
    J = sy.shape[0]
    T = x.shape[0]
    M = valid_t.shape[0]

    interval_best_z = np.empty(M, dtype=np.float64)
    interval_best_obj = np.empty(M, dtype=np.float64)
    interval_num_crossings = np.zeros(M, dtype=np.int64)
    interval_num_support_regions = np.ones(M, dtype=np.int64)
    interval_best_support = np.full((M, k), -1, dtype=np.int64)
    interval_D_coeffs = np.empty((M, 3), dtype=np.float64)

    for pos in range(M):
        t = int(valid_t[pos])
        right = t + 1
        l = x[t]
        u = x[t + 1]
        nr = float(T - right)

        # ------------------------------------------------------------
        # D_t(z) = d0 + d1 z + d2 z^2
        # ------------------------------------------------------------
        p1 = suffix_x[right]
        px_inner = suffix_xx[right]
        q1 = nr
        qx = p1

        d0 = _ma_inner_numba(
            px_inner, p1, px_inner, p1, px_inner,
            g00, g01, g10, g11
        )
        pmq = _ma_inner_numba(
            p1, p1, px_inner, q1, qx,
            g00, g01, g10, g11
        )
        qmq = _ma_inner_numba(
            nr, q1, qx, q1, qx,
            g00, g01, g10, g11
        )
        d1 = -2.0 * pmq
        d2 = qmq

        interval_D_coeffs[pos, 0] = d0
        interval_D_coeffs[pos, 1] = d1
        interval_D_coeffs[pos, 2] = d2

        # ------------------------------------------------------------
        # nu_j(z) = s0_j + s1_j z
        # ------------------------------------------------------------
        s0 = np.empty(J, dtype=np.float64)
        s1 = np.empty(J, dtype=np.float64)

        cp0 = g00 * p1 + g01 * px_inner
        cp1 = g10 * p1 + g11 * px_inner
        cq0 = g00 * q1 + g01 * qx
        cq1 = g10 * q1 + g11 * qx

        for j in range(J):
            s0[j] = suffix_xy[j, right] - (
                sy[j] * cp0 + sxy[j] * cp1
            )
            ymq = suffix_y[j, right] - (
                sy[j] * cq0 + sxy[j] * cq1
            )
            s1[j] = -ymq

        # ------------------------------------------------------------
        # Trivial sparse limits
        # ------------------------------------------------------------
        if k == 0:
            interval_best_z[pos] = l
            interval_best_obj[pos] = C
            continue

        if k == J:
            n0 = 0.0
            n1h = 0.0
            n2 = 0.0
            for j in range(J):
                n0 += s0[j] * s0[j]
                n1h += s0[j] * s1[j]
                n2 += s1[j] * s1[j]
                interval_best_support[pos, j] = j

            bz, bo = _best_ratio_candidate_numba(
                l, u, C,
                n0, 2.0 * n1h, n2,
                d0, d1, d2,
                poly_tol, denom_tol,
            )
            interval_best_z[pos] = bz
            interval_best_obj[pos] = bo
            continue

        inside_tol = 100.0 * poly_tol * max(1.0, abs(l), abs(u))

        # ============================================================
        # PASS 1: count only crossings that actually lie in (l,u).
        # This avoids allocating O(J^2) storage blindly per thread.
        # ============================================================
        event_count = 0

        for i in range(J - 1):
            s0i = s0[i]
            s1i = s1[i]
            for j in range(i + 1, J):
                # nu_i - nu_j = 0
                a0 = s0i - s0[j]
                a1 = s1i - s1[j]
                scale = max(1.0, abs(a0), abs(a1))
                if abs(a1) > poly_tol * scale:
                    z = -a0 / a1
                    if l + inside_tol < z < u - inside_tol:
                        event_count += 1

                # nu_i + nu_j = 0
                a0 = s0i + s0[j]
                a1 = s1i + s1[j]
                scale = max(1.0, abs(a0), abs(a1))
                if abs(a1) > poly_tol * scale:
                    z = -a0 / a1
                    if l + inside_tol < z < u - inside_tol:
                        event_count += 1

        # D can have at most one real root because it is nonnegative.
        # It is a region boundary but does not correspond to a pair swap.
        has_d_root = False
        mu = 0.0
        d_scale = max(1.0, abs(d0), abs(d1), abs(d2))
        if abs(d2) > poly_tol * d_scale:
            discD = d1 * d1 - 4.0 * d2 * d0
            disc_scale = max(1.0, d1 * d1, abs(4.0 * d2 * d0))
            if abs(discD) <= poly_tol * disc_scale:
                mu = -d1 / (2.0 * d2)
                if l + inside_tol < mu < u - inside_tol:
                    has_d_root = True
                    event_count += 1

        # No crossing: one top-k computation and one exact ratio solve.
        if event_count == 0:
            midpoint = 0.5 * (l + u)
            scores = np.empty(J, dtype=np.float64)
            for j in range(J):
                nu = s0[j] + s1[j] * midpoint
                scores[j] = nu * nu

            order = np.argsort(scores)
            threshold = J - k
            n0 = 0.0
            n1h = 0.0
            n2 = 0.0
            for q in range(threshold, J):
                idx = order[q]
                n0 += s0[idx] * s0[idx]
                n1h += s0[idx] * s1[idx]
                n2 += s1[idx] * s1[idx]

            bz, bo = _best_ratio_candidate_numba(
                l, u, C,
                n0, 2.0 * n1h, n2,
                d0, d1, d2,
                poly_tol, denom_tol,
            )
            interval_best_z[pos] = bz
            interval_best_obj[pos] = bo
            for q in range(k):
                interval_best_support[pos, q] = order[threshold + q]
            continue

        # ============================================================
        # PASS 2: fill exact-size event arrays.
        # type 0 = pair crossing; type 1 = D-root boundary.
        # ============================================================
        event_z = np.empty(event_count, dtype=np.float64)
        event_i = np.empty(event_count, dtype=np.int64)
        event_j = np.empty(event_count, dtype=np.int64)
        fill = 0

        for i in range(J - 1):
            s0i = s0[i]
            s1i = s1[i]
            for j in range(i + 1, J):
                a0 = s0i - s0[j]
                a1 = s1i - s1[j]
                scale = max(1.0, abs(a0), abs(a1))
                if abs(a1) > poly_tol * scale:
                    z = -a0 / a1
                    if l + inside_tol < z < u - inside_tol:
                        event_z[fill] = z
                        event_i[fill] = i
                        event_j[fill] = j
                        fill += 1

                a0 = s0i + s0[j]
                a1 = s1i + s1[j]
                scale = max(1.0, abs(a0), abs(a1))
                if abs(a1) > poly_tol * scale:
                    z = -a0 / a1
                    if l + inside_tol < z < u - inside_tol:
                        event_z[fill] = z
                        event_i[fill] = i
                        event_j[fill] = j
                        fill += 1

        if has_d_root:
            event_z[fill] = mu
            event_i[fill] = -1
            event_j[fill] = -1
            fill += 1

        # Sort once.  All subsequent work is a sweep.
        permutation = np.argsort(event_z)
        sorted_z = np.empty(event_count, dtype=np.float64)
        sorted_i = np.empty(event_count, dtype=np.int64)
        sorted_j = np.empty(event_count, dtype=np.int64)
        for e in range(event_count):
            src = permutation[e]
            sorted_z[e] = event_z[src]
            sorted_i[e] = event_i[src]
            sorted_j[e] = event_j[src]

        # Count unique event coordinates for reporting.
        unique_count = 1
        for e in range(1, event_count):
            if abs(sorted_z[e] - sorted_z[e - 1]) > inside_tol:
                unique_count += 1
        interval_num_crossings[pos] = unique_count
        interval_num_support_regions[pos] = unique_count + 1

        # ------------------------------------------------------------
        # Initial ordering: exactly ONE O(J log J) sort for this
        # original interval, using an interior point of first region.
        # order is ascending by gain; rank is its inverse.
        # ------------------------------------------------------------
        first_boundary = sorted_z[0]
        probe = 0.5 * (l + first_boundary)

        scores = np.empty(J, dtype=np.float64)
        for j in range(J):
            nu = s0[j] + s1[j] * probe
            scores[j] = nu * nu

        order = np.argsort(scores)
        rank = np.empty(J, dtype=np.int64)
        for r in range(J):
            rank[order[r]] = r

        threshold = J - k

        # Current N_S coefficients.
        n0 = 0.0
        n1h = 0.0
        n2 = 0.0
        for r in range(threshold, J):
            idx = order[r]
            n0 += s0[idx] * s0[idx]
            n1h += s0[idx] * s1[idx]
            n2 += s1[idx] * s1[idx]

        # Best solution seen in this original interval.
        local_best_obj = np.inf
        local_best_z = l
        local_best_support = np.empty(k, dtype=np.int64)

        # Scratch storage reused by every coincident-event group.
        touched = np.zeros(J, dtype=np.uint8)
        affected = np.empty(J, dtype=np.int64)
        old_ranks = np.empty(J, dtype=np.int64)
        affected_scores = np.empty(J, dtype=np.float64)

        prev_boundary = l
        e0 = 0

        while e0 < event_count:
            z_event = sorted_z[e0]

            # Solve the current constant-support region exactly.
            if z_event - prev_boundary > inside_tol:
                bz, bo = _best_ratio_candidate_numba(
                    prev_boundary,
                    z_event,
                    C,
                    n0,
                    2.0 * n1h,
                    n2,
                    d0,
                    d1,
                    d2,
                    poly_tol,
                    denom_tol,
                )
                if bo < local_best_obj:
                    local_best_obj = bo
                    local_best_z = bz
                    for q in range(k):
                        local_best_support[q] = order[threshold + q]

            # Gather all events at this same numerical coordinate.
            e1 = e0 + 1
            while e1 < event_count and abs(sorted_z[e1] - z_event) <= inside_tol:
                e1 += 1

            # Find the next distinct boundary so that ordering just to the
            # right of this event can be determined without epsilon hacks.
            if e1 < event_count:
                right_probe = 0.5 * (z_event + sorted_z[e1])
            else:
                right_probe = 0.5 * (z_event + u)

            # Collect unique stream indices participating in the event group.
            naff = 0
            for e in range(e0, e1):
                ii = sorted_i[e]
                jj = sorted_j[e]

                # (-1,-1) is a D-root boundary; no ranking update.
                if ii >= 0:
                    if touched[ii] == 0:
                        touched[ii] = 1
                        affected[naff] = ii
                        naff += 1
                    if touched[jj] == 0:
                        touched[jj] = 1
                        affected[naff] = jj
                        naff += 1

            if naff > 0:
                # Save the rank slots occupied by affected streams and compute
                # their post-crossing scores.  No unaffected stream can pass
                # through these streams at z_event unless it also participates
                # in a crossing at z_event.
                for aidx in range(naff):
                    idx = affected[aidx]
                    old_ranks[aidx] = rank[idx]
                    nu = s0[idx] + s1[idx] * right_probe
                    affected_scores[aidx] = nu * nu

                slots_perm = np.argsort(old_ranks[:naff])
                score_perm = np.argsort(affected_scores[:naff])

                # Record old support membership before ranks change.
                old_in = np.empty(naff, dtype=np.uint8)
                for aidx in range(naff):
                    idx = affected[aidx]
                    if rank[idx] >= threshold:
                        old_in[aidx] = 1
                    else:
                        old_in[aidx] = 0

                # Reassign the affected streams to the same set of rank slots,
                # ordered by their score immediately to the right.
                for q in range(naff):
                    slot = old_ranks[slots_perm[q]]
                    idx = affected[score_perm[q]]
                    order[slot] = idx
                    rank[idx] = slot

                # Incrementally update N_S only for streams crossing the
                # top-k boundary.  This is the O(1)-per-enter/leave update
                # that the proof's sweep is meant to exploit.
                for aidx in range(naff):
                    idx = affected[aidx]
                    new_in = 1 if rank[idx] >= threshold else 0
                    if old_in[aidx] == 1 and new_in == 0:
                        n0 -= s0[idx] * s0[idx]
                        n1h -= s0[idx] * s1[idx]
                        n2 -= s1[idx] * s1[idx]
                    elif old_in[aidx] == 0 and new_in == 1:
                        n0 += s0[idx] * s0[idx]
                        n1h += s0[idx] * s1[idx]
                        n2 += s1[idx] * s1[idx]

                # Reset touched flags only for entries used in this group.
                for aidx in range(naff):
                    touched[affected[aidx]] = 0

            prev_boundary = z_event
            e0 = e1

        # Final region after the last event.
        if u - prev_boundary > inside_tol:
            bz, bo = _best_ratio_candidate_numba(
                prev_boundary,
                u,
                C,
                n0,
                2.0 * n1h,
                n2,
                d0,
                d1,
                d2,
                poly_tol,
                denom_tol,
            )
            if bo < local_best_obj:
                local_best_obj = bo
                local_best_z = bz
                for q in range(k):
                    local_best_support[q] = order[threshold + q]

        # Extremely degenerate fallback (all regions numerically zero width).
        if not np.isfinite(local_best_obj):
            midpoint = 0.5 * (l + u)
            for j in range(J):
                nu = s0[j] + s1[j] * midpoint
                scores[j] = nu * nu
            order = np.argsort(scores)
            n0 = 0.0
            n1h = 0.0
            n2 = 0.0
            for r in range(threshold, J):
                idx = order[r]
                n0 += s0[idx] * s0[idx]
                n1h += s0[idx] * s1[idx]
                n2 += s1[idx] * s1[idx]
            local_best_z, local_best_obj = _best_ratio_candidate_numba(
                l, u, C,
                n0, 2.0 * n1h, n2,
                d0, d1, d2,
                poly_tol, denom_tol,
            )
            for q in range(k):
                local_best_support[q] = order[threshold + q]

        interval_best_z[pos] = local_best_z
        interval_best_obj[pos] = local_best_obj
        for q in range(k):
            interval_best_support[pos, q] = local_best_support[q]

    return (
        interval_best_z,
        interval_best_obj,
        interval_num_crossings,
        interval_num_support_regions,
        interval_best_support,
        interval_D_coeffs,
    )


# =========================================================================
# MATHEMATICAL GUIDE: _recover_sparse_solution_numba
# Coefficient recovery (paper Lemma 4, Step 4): for selected j,
# delta_j=nu_j(z*)/D(z*), and [alpha_j,beta_j]^T
# =G^{-1} A.T (y_j-delta_j*h(z*)).
# Fit is alpha_j+beta_j*x+delta_j*max(0,x-z*).
# For unselected j set delta_j=0, giving a single OLS line.
# =========================================================================
@njit
def _recover_sparse_solution_numba(
    best_z,
    best_t,
    best_support,
    sy,
    sxy,
    Cj,
    suffix_x,
    suffix_xx,
    suffix_y,
    suffix_xy,
    g00,
    g01,
    g10,
    g11,
    T,
    denom_tol,
):
    J = sy.shape[0]
    right = best_t + 1
    p1 = suffix_x[right]
    px_inner = suffix_xx[right]
    q1 = float(T - right)
    qx = p1

    d0 = _ma_inner_numba(px_inner, p1, px_inner, p1, px_inner, g00, g01, g10, g11)
    pmq = _ma_inner_numba(p1, p1, px_inner, q1, qx, g00, g01, g10, g11)
    qmq = _ma_inner_numba(q1, q1, qx, q1, qx, g00, g01, g10, g11)
    d1 = -2.0 * pmq
    d2 = qmq
    Dbest = d0 + d1 * best_z + d2 * best_z * best_z
    Dscale = max(1.0, abs(d0), abs(d1 * best_z), abs(d2 * best_z * best_z))

    s0 = np.empty(J, dtype=np.float64)
    s1 = np.empty(J, dtype=np.float64)
    cp0 = g00 * p1 + g01 * px_inner
    cp1 = g10 * p1 + g11 * px_inner
    cq0 = g00 * q1 + g01 * qx
    cq1 = g10 * q1 + g11 * qx
    for j in range(J):
        s0[j] = suffix_xy[j, right] - (sy[j] * cp0 + sxy[j] * cp1)
        ymq = suffix_y[j, right] - (sy[j] * cq0 + sxy[j] * cq1)
        s1[j] = -ymq

    selected_mask = np.zeros(J, dtype=np.bool_)
    for q in range(best_support.shape[0]):
        selected_mask[best_support[q]] = True

    delta = np.zeros(J, dtype=np.float64)
    if Dbest > denom_tol * Dscale:
        for q in range(best_support.shape[0]):
            j = best_support[q]
            delta[j] = (s0[j] + s1[j] * best_z) / Dbest

    sum_h = p1 - best_z * q1
    sum_xh = px_inner - best_z * qx
    alpha = np.empty(J, dtype=np.float64)
    beta = np.empty(J, dtype=np.float64)
    left_slope = np.empty(J, dtype=np.float64)
    right_slope = np.empty(J, dtype=np.float64)
    left_intercept = np.empty(J, dtype=np.float64)
    right_intercept = np.empty(J, dtype=np.float64)
    continuity_value = np.empty(J, dtype=np.float64)
    losses = Cj.copy()

    for j in range(J):
        rhs0 = sy[j] - delta[j] * sum_h
        rhs1 = sxy[j] - delta[j] * sum_xh
        a = g00 * rhs0 + g01 * rhs1
        b = g10 * rhs0 + g11 * rhs1
        alpha[j] = a
        beta[j] = b
        left_slope[j] = b
        right_slope[j] = b + delta[j]
        left_intercept[j] = a
        right_intercept[j] = a - delta[j] * best_z
        continuity_value[j] = a + b * best_z

    if Dbest > denom_tol * Dscale:
        for q in range(best_support.shape[0]):
            j = best_support[q]
            nu = s0[j] + s1[j] * best_z
            losses[j] = Cj[j] - (nu * nu) / Dbest
            if losses[j] < 0.0 and losses[j] > -1e-8 * max(1.0, Cj[j]):
                losses[j] = 0.0

    return (
        selected_mask, losses, alpha, delta, left_slope, right_slope,
        left_intercept, right_intercept, continuity_value
    )


# =========================================================================
# MATHEMATICAL GUIDE: exact_sparse_multidimensional_slope_change
# Outer orchestration for paper Eq. (21):
# min_{z,S:|S|=k} sum_j ||y_j-alpha_j-beta_j*x
#                              -delta_j*max(0,x-z)||^2,
# with delta_j=0 for j outside S.
# C_j=||M_A y_j||^2, C=sum_j C_j; suffix statistics permit
# O(J) construction of nu_j and D per original interval.
# The numerical sweep handles O(J^2) pairwise crossings per interval.
# =========================================================================
def exact_sparse_multidimensional_slope_change(
    x,
    Y,
    k,
    min_side_points=MIN_SIDE_POINTS_DEFAULT,
    poly_tol=POLY_TOL_DEFAULT,
    denom_tol=DENOM_TOL_DEFAULT,
):
    """Exact sparse solver with the O(T J^2 log J) interval work in Numba."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    J, T = Y.shape

    if x.ndim != 1 or len(x) != T:
        raise ValueError("x must be one-dimensional with len(x) == Y.shape[1].")
    if not np.isfinite(x).all() or not np.isfinite(Y).all() or not np.all(np.diff(x) >= 0.0):
        raise ValueError("x must be finite and nondecreasing; Y must be finite.")
    if min_side_points < 1 or 2 * min_side_points > T:
        raise ValueError("Invalid min_side_points.")
    if int(k) != k or not (0 <= int(k) <= J):
        raise ValueError(f"k must satisfy 0 <= k <= J={J}.")
    k = int(k)

    # --------------------------------------------------------
    # Straight-line OLS baseline / global sufficient statistics
    # --------------------------------------------------------
    baseline_start = time.perf_counter()

    sx = float(np.sum(x))
    sxx = float(np.dot(x, x))
    det = float(T) * sxx - sx * sx
    g00 = sxx / det
    g01 = -sx / det
    g10 = -sx / det
    g11 = float(T) / det

    sy = np.ascontiguousarray(np.sum(Y, axis=1), dtype=np.float64)
    sxy = np.ascontiguousarray(Y @ x, dtype=np.float64)
    syy = np.ascontiguousarray(np.einsum("ij,ij->i", Y, Y), dtype=np.float64)
    projected_norm = g00 * sy * sy + 2.0 * g01 * sy * sxy + g11 * sxy * sxy
    Cj = np.ascontiguousarray(syy - projected_norm, dtype=np.float64)
    mask = (Cj < 0.0) & (Cj > -1e-8 * np.maximum(1.0, syy))
    Cj[mask] = 0.0
    C = float(np.sum(Cj))

    baseline_runtime = time.perf_counter() - baseline_start

    # --------------------------------------------------------
    # Prefix/suffix construction
    # --------------------------------------------------------
    prefix_start = time.perf_counter()

    suffix_x = np.zeros(T + 1, dtype=np.float64)
    suffix_xx = np.zeros(T + 1, dtype=np.float64)
    suffix_y = np.zeros((J, T + 1), dtype=np.float64)
    suffix_xy = np.zeros((J, T + 1), dtype=np.float64)
    suffix_x[:T] = np.cumsum(x[::-1])[::-1]
    suffix_xx[:T] = np.cumsum((x * x)[::-1])[::-1]
    suffix_y[:, :T] = np.cumsum(Y[:, ::-1], axis=1)[:, ::-1]
    suffix_xy[:, :T] = np.cumsum((Y * x[None, :])[:, ::-1], axis=1)[:, ::-1]
    suffix_x = np.ascontiguousarray(suffix_x)
    suffix_xx = np.ascontiguousarray(suffix_xx)
    suffix_y = np.ascontiguousarray(suffix_y)
    suffix_xy = np.ascontiguousarray(suffix_xy)

    prefix_runtime = time.perf_counter() - prefix_start

    first_t = min_side_points - 1
    last_t = T - min_side_points - 1
    valid_t = np.arange(first_t, last_t + 1, dtype=np.int64)
    valid_t = valid_t[x[valid_t + 1] > x[valid_t]]
    if not len(valid_t):
        raise ValueError("No positive-width admissible breakpoint intervals.")

    # --------------------------------------------------------
    # Exact sparse interval kernel
    # Includes: interval D/nu setup, pairwise crossing generation,
    # event sorting, initial ranking, sweep updates, and exact
    # quadratic-ratio candidate solves on all support regions.
    # --------------------------------------------------------
    interval_kernel_start = time.perf_counter()
    (
        interval_best_z,
        interval_best_obj,
        interval_num_crossings,
        interval_num_support_regions,
        interval_best_support,
        interval_D_coeffs,
    ) = _solve_sparse_intervals_numba(
        x, valid_t, sy, sxy, suffix_x, suffix_xx, suffix_y, suffix_xy,
        g00, g01, g10, g11, C, k, poly_tol, denom_tol
    )
    interval_kernel_runtime = time.perf_counter() - interval_kernel_start

    # --------------------------------------------------------
    # Global sparse dimension / interval selection
    # --------------------------------------------------------
    selection_start = time.perf_counter()

    best_pos = int(np.argmin(interval_best_obj))
    best_t = int(valid_t[best_pos])
    best_z = float(interval_best_z[best_pos])
    best_obj = float(interval_best_obj[best_pos])
    best_support = np.asarray(interval_best_support[best_pos], dtype=np.int64).copy()
    if k == 0:
        best_support = np.empty(0, dtype=np.int64)

    selection_runtime = time.perf_counter() - selection_start

    # --------------------------------------------------------
    # Recover fitted coefficients at the globally best solution
    # --------------------------------------------------------
    recovery_start = time.perf_counter()

    (
        selected_mask,
        losses_by_dimension,
        alpha,
        delta,
        left_slope,
        right_slope,
        left_intercept,
        right_intercept,
        continuity_value,
    ) = _recover_sparse_solution_numba(
        best_z, best_t, best_support, sy, sxy, Cj,
        suffix_x, suffix_xx, suffix_y, suffix_xy,
        g00, g01, g10, g11, T, denom_tol
    )
    recovery_runtime = time.perf_counter() - recovery_start

    numerical_runtime = (
        baseline_runtime
        + prefix_runtime
        + interval_kernel_runtime
        + selection_runtime
        + recovery_runtime
    )
    solve_runtime = numerical_runtime

    return {
        "k": k,
        "valid_t": valid_t,
        "interval_best_breakpoints": interval_best_z,
        "interval_optimal_values": interval_best_obj,
        "interval_num_crossings": interval_num_crossings,
        "interval_num_support_regions": interval_num_support_regions,
        "interval_best_support": interval_best_support,
        "interval_D_coefficients": interval_D_coeffs,
        "C": C,
        "Cj": Cj,
        "best_t": best_t,
        "best_position": best_pos,
        "shared_breakpoint": best_z,
        "optimal_loss": best_obj,
        "selected_support": best_support,
        "selected_mask": selected_mask,
        "losses_by_dimension": losses_by_dimension,
        "alpha": alpha,
        "delta": delta,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "left_intercept": left_intercept,
        "right_intercept": right_intercept,
        "continuity_value": continuity_value,
        "baseline_runtime": baseline_runtime,
        "prefix_runtime": prefix_runtime,
        "interval_kernel_runtime": interval_kernel_runtime,
        "selection_runtime": selection_runtime,
        "recovery_runtime": recovery_runtime,
        "numerical_runtime": numerical_runtime,
        "solve_runtime": solve_runtime,
    }


# =========================================================================
# MATHEMATICAL GUIDE: warm_up_numba_sparse_exact
# JIT compile on a small instance before measuring runtime; not mathematics.
# =========================================================================
def warm_up_numba_sparse_exact():
    """Compile all Numba kernels on a tiny instance.

    This time is reported separately and deliberately excluded from both
    total numerical runtime and overall runtime.
    """
    x_test = np.arange(8, dtype=np.float64)
    Y_test = np.ascontiguousarray(
        np.vstack([
            1.0 + 0.2 * x_test + 0.03 * x_test * x_test,
            2.0 - 0.1 * x_test + 0.02 * x_test * x_test,
            3.0 + 0.4 * x_test,
            4.0 - 0.3 * x_test,
        ]),
        dtype=np.float64,
    )
    exact_sparse_multidimensional_slope_change(
        x=x_test,
        Y=Y_test,
        k=2,
        min_side_points=2,
        poly_tol=POLY_TOL_DEFAULT,
        denom_tol=DENOM_TOL_DEFAULT,
    )


# =========================================================================
# MATHEMATICAL GUIDE: direct_sparse_continuous_fit_loss
# Independent fixed-z check: solve least squares directly using design
# [1,x,max(0,x-z)] for selected streams and [1,x] otherwise.
# Compare its SSE to the closed-form/sweep objective to catch mistakes.
# =========================================================================
def direct_sparse_continuous_fit_loss(x, Y, r, support):
    """
    Independent fixed-breakpoint validation.

    Selected streams are fit with a continuous two-piece hinge regression;
    unselected streams are fit by a single ordinary least-squares line.
    """
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    J, T = Y.shape

    selected = np.zeros(J, dtype=bool)
    selected[np.asarray(support, dtype=np.int64)] = True

    losses = np.empty(J, dtype=np.float64)
    theta = np.full((J, 3), np.nan, dtype=np.float64)

    # Single-line design for unchanged streams.
    X_line = np.column_stack([np.ones_like(x), x])

    # Continuous two-piece design for selected streams.
    left = np.minimum(x - r, 0.0)
    right = np.maximum(x - r, 0.0)
    X_hinge = np.column_stack([np.ones_like(x), left, right])

    for j in range(J):
        if selected[j]:
            coef, _, _, _ = np.linalg.lstsq(X_hinge, Y[j], rcond=None)
            residual = Y[j] - X_hinge @ coef
            losses[j] = float(np.dot(residual, residual))
            theta[j, :] = coef
        else:
            coef, _, _, _ = np.linalg.lstsq(X_line, Y[j], rcond=None)
            residual = Y[j] - X_line @ coef
            losses[j] = float(np.dot(residual, residual))
            theta[j, 0] = coef[0] + coef[1] * r
            theta[j, 1] = coef[1]
            theta[j, 2] = coef[1]

    return float(np.sum(losses)), losses, theta


# =========================================================================
# MATHEMATICAL GUIDE: write_excel
# Report objective, breakpoint, selected streams, interval diagnostics,
# recovered coefficients, and timing. No optimization takes place here.
# =========================================================================
def write_excel(path, x, Y, dates, tickers, result, timings, validation):
    """Write compact results; never place the full D x T matrix in Excel."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    summary = {
        "D": Y.shape[0], "T": Y.shape[1], "k": result["k"],
        "best_interval_index_zero_based": result["best_t"],
        "breakpoint": result["shared_breakpoint"],
        "optimal_SSE": result["optimal_loss"],
        "baseline_SSE": result["C"],
        "selected_count": int(result["selected_mask"].sum()),
        "numba_warmup_seconds_excluded": timings["warmup"],
        "numerical_solver_seconds": result["numerical_runtime"],
        "validation_seconds": timings["validation"],
        "direct_validation_SSE": validation["direct_loss"] if validation else None,
        "validation_absolute_difference": validation["absolute_difference"] if validation else None,
        "validation_relative_difference": validation["relative_difference"] if validation else None,
    }
    summary.update({key + "_seconds": result[key] for key in
                    ("baseline_runtime", "prefix_runtime", "interval_kernel_runtime", "selection_runtime", "recovery_runtime")})
    ws.append(["Metric", "Value"])
    for key, value in summary.items():
        ws.append([key, value])

    wi = wb.create_sheet("Intervals")
    wi.append(["t (zero-based)", "left_x", "right_x", "left_date", "right_date",
               "best_breakpoint", "best_SSE", "pairwise_crossings", "support_regions", "d0", "d1", "d2"])
    for pos, t in enumerate(result["valid_t"]):
        wi.append([int(t), float(x[t]), float(x[t+1]), str(dates[t]), str(dates[t+1]),
                   float(result["interval_best_breakpoints"][pos]), float(result["interval_optimal_values"][pos]),
                   int(result["interval_num_crossings"][pos]), int(result["interval_num_support_regions"][pos]),
                   *[float(v) for v in result["interval_D_coefficients"][pos]]])

    wf = wb.create_sheet("Fitted streams")
    wf.append(["ticker", "selected", "SSE", "left_slope", "right_slope", "delta_slope",
               "left_intercept", "right_intercept", "continuity_value"])
    for j, ticker in enumerate(tickers):
        wf.append([str(ticker), bool(result["selected_mask"][j]), float(result["losses_by_dimension"][j]),
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
            sheet.column_dimensions[letter].width = min(36, max(14, max(len(str(c.value or "")) for c in list(col)[:150]) + 2))
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = '0.0000000000'
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# =========================================================================
# MATHEMATICAL GUIDE: main
# Read saved (x,Y), run JIT warmup, optimize, optionally validate, export.
# =========================================================================
@threadpool_limits.wrap(limits=1)
def main():
    parser = argparse.ArgumentParser(description="Optimize a saved simulated dataset and export Excel")
    parser.add_argument("--input", default="synthetic_stock_data.npz")
    parser.add_argument("--output", default="exact_sparse_results.xlsx")
    parser.add_argument("--k", type=int, default=None, help="Default: 20 percent of D")
    parser.add_argument("--threads", type=int, choices=(1,), default=1,
                        help="Single-thread comparison; only 1 is supported")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    set_num_threads(args.threads)
    with np.load(args.input, allow_pickle=False) as data:
        x = data["x"].copy()
        Y = data["Y"].copy()
        dates = data["dates"].copy()
        tickers = data["tickers"].copy()
    D, T = Y.shape
    k = args.k if args.k is not None else int(0.2 * D)
    if D > 2000 and 0 < k < D:
        print("WARNING: The original pairwise-crossing algorithm is O(T*D^2); this may be extremely slow.")
    start = time.perf_counter()
    warm_up_numba_sparse_exact()
    warmup = time.perf_counter() - start
    result = exact_sparse_multidimensional_slope_change(x, Y, k)
    validation = None
    validation_time = 0.0
    if not args.skip_validation:
        t0 = time.perf_counter()
        loss, _, _ = direct_sparse_continuous_fit_loss(x, Y, result["shared_breakpoint"], result["selected_support"])
        validation_time = time.perf_counter() - t0
        validation = {"direct_loss": loss, "absolute_difference": abs(loss-result["optimal_loss"]),
                      "relative_difference": abs(loss-result["optimal_loss"])/max(1.0, abs(result["optimal_loss"]))}
    write_excel(Path(args.output), x, Y, dates, tickers, result,
                {"warmup": warmup, "validation": validation_time}, validation)
    print(f"D={D}, T={T}, k={k}; breakpoint={result['shared_breakpoint']:.12g}; SSE={result['optimal_loss']:.12g}")
    print(f"Solver runtime={result['numerical_runtime']:.4f}s; warmup={warmup:.4f}s (excluded)")
    print(f"Excel results saved to {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
