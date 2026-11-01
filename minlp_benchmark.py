"""Manuscript Appendix C bounded Gurobi nonconvex MIQCP benchmark."""
import time
import numpy as np


def parameter_bounds(x, Y):
    """Appendix C.1--C.4 in original input coordinates.

    Every pairwise secant is a positive weighted average of consecutive
    secants. Their minimum and maximum therefore equal the extrema of the
    consecutive secants, computing exactly (C.1)--(C.2) in O(JT) work.
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if x.ndim != 1 or Y.ndim != 2 or Y.shape[1] != len(x) or len(x) < 6:
        raise ValueError('Require x=(T,), Y=(J,T), T>=6')
    if not np.isfinite(x).all() or not np.isfinite(Y).all() or not np.all(np.diff(x) >= 0) or x[1] >= x[-2]:
        raise ValueError('Require finite data, nondecreasing x, and nonempty interior domain')
    if np.all(np.diff(x) > 0):
        slopes = np.diff(Y, axis=1) / np.diff(x)
        lm, um = slopes.min(axis=1), slopes.max(axis=1)
    else:
        # Repeated observation times: all secants between distinct time values.
        ii, jj = np.triu_indices(len(x), 1)
        positive = x[jj] > x[ii]
        ii, jj = ii[positive], jj[positive]
        lm, um = np.empty(len(Y)), np.empty(len(Y))
        for a in range(0, len(Y), 32):
            slopes = (Y[a:a+32, jj] - Y[a:a+32, ii]) / (x[jj]-x[ii])
            lm[a:a+32], um[a:a+32] = slopes.min(axis=1), slopes.max(axis=1)
    # Intercepts from Appendix C.1; formula valid for arbitrary x signs.
    low_aff = np.minimum(lm[:, None] * x, um[:, None] * x)
    high_aff = np.maximum(lm[:, None] * x, um[:, None] * x)
    lc = np.min(Y - high_aff, axis=1)
    uc = np.max(Y - low_aff, axis=1)
    fit_lb = low_aff + lc[:, None]
    fit_ub = high_aff + uc[:, None]
    M1 = fit_ub - fit_lb
    M2 = np.maximum(0, x - x[1])
    M3 = np.maximum(0, x[-2] - x)
    return dict(lm=lm, um=um, lc=lc, uc=uc, fit_lb=fit_lb, fit_ub=fit_ub,
                M1=M1, M2=M2, M3=M3)


def exact_bound_diagnostics(result, bounds):
    """Whether the recovered unconstrained solution is in the MINLP box."""
    violations = []
    for key, lower, upper in [('left_slope', 'lm', 'um'), ('right_slope', 'lm', 'um'),
                              ('left_intercept', 'lc', 'uc'), ('right_intercept', 'lc', 'uc')]:
        value = result[key]
        tol = 1e-7 * np.maximum(1, np.maximum(np.abs(bounds[lower]), np.abs(bounds[upper])))
        violations.append((value < bounds[lower]-tol) | (value > bounds[upper]+tol))
    count = int(np.any(violations, axis=0).sum())
    return dict(exact_within_minlp_bounds=(count == 0), exact_bound_violating_dimensions=count)


def solve_minlp(x, Y, k=None, time_limit=2000.0, threads=1, output_flag=0,
                solution_path=None, log_path=None, random_seed=0, bounds=None, names=None):
    import gurobipy as gp
    from gurobipy import GRB
    x, Y = np.asarray(x, dtype=float), np.asarray(Y, dtype=float)
    J, T = Y.shape
    k = J if k is None else k
    if int(k) != k or not 0 <= k <= J:
        raise ValueError('k must be an integer between 0 and J')
    if time_limit <= 0 or threads < 1:
        raise ValueError('Invalid time limit or thread count')
    start = time.perf_counter()
    bounds = parameter_bounds(x, Y) if bounds is None else bounds
    lm, um, lc, uc, fit_lb, fit_ub, M1, M2, M3 = (bounds[key] for key in
        ('lm', 'um', 'lc', 'uc', 'fit_lb', 'fit_ub', 'M1', 'M2', 'M3'))
    model = gp.Model('shared_breakpoint_paper_minlp')
    try:
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = int(output_flag)
        model.Params.NonConvex = 2
        model.Params.TimeLimit = float(time_limit)
        model.Params.Threads = int(threads)
        model.Params.Seed = int(random_seed)
        if log_path is not None:
            from pathlib import Path
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            model.Params.LogFile = str(log_path)
        model.Params.OutputFlag = int(bool(output_flag) or log_path is not None)
        z = model.addVar(lb=float(x[1]), ub=float(x[-2]), name='z')
        X = model.addMVar(T, vtype=GRB.BINARY, name='X')
        m1 = model.addMVar(J, lb=lm, ub=um, name='m1')
        m2 = model.addMVar(J, lb=lm, ub=um, name='m2')
        c1 = model.addMVar(J, lb=lc, ub=uc, name='c1')
        c2 = model.addMVar(J, lb=lc, ub=uc, name='c2')
        yh = model.addMVar((J, T), lb=fit_lb, ub=fit_ub, name='yhat')
        model.addConstr(X[:-1] >= X[1:], name='monotone')
        model.addConstr(x <= z + M2 * (1-X), name='left_location')
        model.addConstr(x >= z - M3 * X, name='right_location')
        model.addConstr(m1 * z + c1 == m2 * z + c2, name='continuity')
        a1 = m1[:, None] * x[None, :] + c1[:, None]
        a2 = m2[:, None] * x[None, :] + c2[:, None]
        model.addConstr(a1 - yh <= M1 * (1-X[None, :]), name='left_upper')
        model.addConstr(yh - a1 <= M1 * (1-X[None, :]), name='left_lower')
        model.addConstr(a2 - yh <= M1 * X[None, :], name='right_upper')
        model.addConstr(yh - a2 <= M1 * X[None, :], name='right_lower')
        if k < J:
            xi = model.addMVar(J, vtype=GRB.BINARY, name='selected')
            model.addConstr(xi.sum() <= k, name='cardinality')
            model.addConstr(m2-m1 <= (um-lm)*xi, name='slope_up')
            model.addConstr(m1-m2 <= (um-lm)*xi, name='slope_down')
            model.addConstr(c2-c1 <= (uc-lc)*xi, name='intercept_up')
            model.addConstr(c1-c2 <= (uc-lc)*xi, name='intercept_down')
        model.setObjective(((yh-Y)*(yh-Y)).sum(), GRB.MINIMIZE)
        build_seconds = time.perf_counter() - start
        model.optimize()
        status = int(model.Status)
        status_name = {GRB.OPTIMAL:'OPTIMAL', GRB.TIME_LIMIT:'TIME_LIMIT', GRB.INFEASIBLE:'INFEASIBLE',
                       GRB.INF_OR_UNBD:'INF_OR_UNBD', GRB.INTERRUPTED:'INTERRUPTED',
                       GRB.NUMERIC:'NUMERIC', GRB.SUBOPTIMAL:'SUBOPTIMAL'}.get(status, str(status))
        incumbent = float(model.ObjVal) if model.SolCount else None
        try:
            bound = float(model.ObjBound)
            if not np.isfinite(bound): bound = None
        except (AttributeError, gp.GurobiError):
            bound = None
        gap = float(model.MIPGap)*100 if model.SolCount else None
        if gap is not None and not np.isfinite(gap):
            gap = None
        result = dict(status=status_name, status_code=status, incumbent_SSE=incumbent,
                    best_bound_SSE=bound, mip_gap_percent=gap,
                    solver_runtime_seconds=float(model.Runtime), build_runtime_seconds=build_seconds,
                    total_runtime_seconds=time.perf_counter()-start,
                    solution_count=int(model.SolCount), node_count=float(model.NodeCount),
                    breakpoint=float(z.X) if model.SolCount else None,
                    selected_count=int(round(sum(xi.X))) if model.SolCount and k<J else (J if model.SolCount else None),
                    gurobi_version='.'.join(map(str, gp.gurobi.version())),
                    formulation='paper_appendix_C_bounded_MINLP')
        if solution_path is not None and model.SolCount:
            from pathlib import Path
            path = Path(solution_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, x=x, names=np.asarray(names if names is not None else [], dtype=str),
                                breakpoint=float(z.X), left_slope=m1.X, right_slope=m2.X,
                                left_intercept=c1.X, right_intercept=c2.X,
                                selected_mask=xi.X > 0.5 if k < J else np.ones(J, dtype=bool),
                                fitted=yh.X, objective=incumbent)
        result['total_runtime_seconds'] = time.perf_counter() - start
        return result
    finally:
        model.dispose()
