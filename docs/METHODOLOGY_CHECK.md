# Methodology check against the supplied manuscript

Reference: **Polynomial-Time Solvability of the Multi-Dimensional Slope Change Problem**, Apoorva Narula and Santanu S. Dey, supplied as `Sparse_and_non_sparse_Slopde_CPD.pdf`. The supplied document is the methodological authority. This package implements its squared L2 experiments; the manuscript's L1 and L-infinity results are outside this implementation.

## Non-sparse algorithm

`run_optimization_nonsparse.py` uses the continuous hinge representation `alpha_j + beta_j*x + delta_j*max(0,x-z)` and sufficient statistics for projection off `[1,x]`. Its interval objective is `C - N(z)/D(z)`, as in (7)-(13). `N` and `D` are quadratic regardless of J. The stationary polynomial `R=N'D-ND'` has degree at most two, as in (15). Interval endpoints, admissible stationary roots and degenerate denominator cases are considered before global interval selection. Suffix statistics implement the O(JT) arithmetic-count construction described in Section 3.1.

## Sparse algorithm

`run_optimization_sparse.py` implements Section 4.1, (20)-(22). With a fixed z, the common denominator means the k largest squared affine numerators identify the optimal support. The implementation enumerates the roots of both `nu_i-nu_j` and `nu_i+nu_j`, sorts crossings, sweeps support regions, and solves the quadratic-ratio problem in each region. Coincident events are grouped numerically and affected ranks/support coefficients are updated. The resulting arithmetic bound follows O(T J² log J); floating-point event merging uses the supplied tolerances. k=0 and k=J have dedicated branches. `|S|=k` means k dimensions are *permitted* to change; some selected dimensions can have zero fitted slope change. This is equivalent to allowing at most k actual changes.

Under tied gains or flat optimal objectives, multiple supports or breakpoint locations can attain the same SSE. Agreement of objective and recovered residuals is the primary numerical comparison; identical supports or breakpoint coordinates are not required when solutions are nonunique.

## Breakpoint domain and sparsity

Both algorithms and the MINLP use the manuscript's admissible domain `[x_2,x_{T-1}]` (Python `[x[1],x[-2]]`), including both endpoints. Sparse experiments use `floor(0.20 J)` throughout. This gives k=8080 for ERA5, k=218 for HoloAssist and k=4 for NASA.

## Repeated HoloAssist timestamps

The pinned workbook contains 240 paired hand readings but only 232 distinct `clip_time_s` values, including eight adjacent repeats. The manuscript assumes strictly increasing time coordinates. Silently changing these timestamps would change the requested dataset, so this implementation explicitly extends input handling to nondecreasing time coordinates while keeping **every original observation with weight one**.

For the polynomial-time methods, zero-width intervals are discarded. The positive-width intervals still cover the entire admissible domain. For each such interval the hinge representation, projection, common denominator and quadratic-ratio formulas remain valid. The design `[1,x]` retains rank two; repeated observations contribute separately to every sufficient statistic and residual. At a timestamp equal to the breakpoint the hinge is zero, so assigning such an observation to either continuous segment produces the same fitted value. No jitter, averaging, resampling or weighting is introduced.

The MINLP uses the same raw x and Y, including repeated times. Secant bounds are computed over pairs with **different time coordinates**; equal-time pairs have undefined slopes and are excluded. All finite slopes, intercept bounds and Big-M constants then follow Appendix C. The location binaries may assign equal-time observations to either side at their shared breakpoint without changing fitted values because of continuity. This data accommodation is an implementation extension, not a claim that the manuscript had distinct-time data here. Independent exhaustive-support tests include repeated times with different response values.

## Appendix C bounds and Big-M constants

`parameter_bounds` implements:

| Manuscript | Implementation |
|---|---|
| (C.1)-(C.2) | `lm`, `um`: minimum/maximum observed secant slopes per dimension |
| (C.3)-(C.5) | `lc`, `uc`: intercept extrema from both slope bounds at every x, valid for positive and negative x |
| (C.6)-(C.7) | `x[1] <= z <= x[-2]`, slope/intercept box on both segments |
| (C.8)-(C.13) | `fit_lb`, `fit_ub`, `M1=fit_ub-fit_lb` |
| (C.14)-(C.18) | `M2=max(0,x-x[1])`, `M3=max(0,x[-2]-x)`, monotone assignment binaries |
| (C.19)-(C.25) | slope/intercept difference bounds `um-lm`, `uc-lc` times binary selections; `sum(selected)<=k` |

For strictly increasing x, a long secant is a positive weighted average of consecutive secants, with weights equal to consecutive time gaps divided by the full time gap. Thus the extrema over *all* secants are exactly the extrema over consecutive secants. The code uses this equivalent O(JT) evaluation of (C.1)-(C.2). For repeated times, it explicitly evaluates all distinct-time pairs in dimension batches. Tests independently compare the returned bounds with all-pairs enumeration.

The optimization model imposes both sides of each inactive-segment Big-M inequality, explicit fitted-value bounds, shared bilinear continuity, the SSE objective, `NonConvex=2`, and a default 2000-second `TimeLimit`. k=1 is accepted (the original benchmark's validation incorrectly rejected it). Missing/infinite bounds and gaps are blank, rather than represented as a false zero. The Gurobi model is disposed even if an error occurs.

## Parameter-domain comparability

The polynomial-time fit permits unrestricted regression parameters, while the prescribed MINLP has a data-driven parameter box. The package retains both formulations. It exports the actual box and tests whether the **recovered** exact solution satisfies it, within a relative 1e-7 feasibility tolerance. A true flag provides a feasible exact candidate for the bounded problem; a false flag does not by itself prove that every unrestricted optimum violates the box. A MINLP lower bound is a bound on the bounded problem. The exact SSE, MINLP incumbent, MINLP bound and reported MIP gap should be interpreted with this distinction.

## Numerical arithmetic and coordinate stability

The supplied solver kernels use float64 arithmetic, cancellation-resistant quadratic roots and scale-dependent zero tests. They enumerate the exact method's candidate set numerically; they do not return symbolic rational/quadratic-field certificates as in Remark 1 of the manuscript. Consequently their reported optimum is subject to floating-point tolerances.

`fitting.solve_exact` maps time to `u=(x-x[0])/(x[-1]-x[0])` and centers each response by its temporal mean. These invertible transformations leave the unweighted raw-unit SSE and the admissible fits unchanged. They avoid cancellation from years near 2000 and large sensor offsets. Slopes, intercepts and all reported breakpoint locations are transformed back to original units. The interval polynomial diagnostic arrays remain in the internal u/centered-response coordinates, documented by `time_origin` and `time_scale` in NPZ. No sensor variance normalization is performed. Independent tests verify affine-coordinate invariance and direct residual reconstruction.

## Runtime and output definitions

- `numerical_runtime_seconds`: sum of the supplied numerical kernel phase timings; startup JIT compilation is measured separately in runtime workbooks and summary.
- `exact_end_to_end_seconds`: complete stable adapter call, including coordinate conversion, solver orchestration and coefficient conversion, excluding JIT warmup, input acquisition, validation and writing.
- `data_generation_seconds`: synthetic generation for the case. Real workbooks are acquired/parsed once per dataset before its cases; that shared acquisition is excluded from numerical runtime.
- `data_loading_seconds`: shared real-dataset acquisition, checksum and parsing cost; the same measured value is repeated in both method rows and must not be added twice when accounting for a dataset load.
- `validation_seconds`: independently reconstructing residual SSE in chunks, outside fitting timing. Every completed exact case is checked.
- `bound_construction_seconds`: actual Appendix C array construction before the MINLP call. The same arrays are passed to the model and saved.
- `minlp_solver_runtime_seconds`: Gurobi `Runtime`, with default `TimeLimit=2000`. The solver can return just after the limit due to termination overhead.
- `minlp_build_runtime_seconds`: model construction within the MINLP call; separately recorded.
- `minlp_total_runtime_seconds`: bounds plus elapsed MINLP worker process, including interpreter startup, input loading, model building/solving and solution serialization. Bounds-file writing is outside this field. The worker's own model/solver timings and result JSON are retained separately. A child-process failure is recorded as a MINLP error, allowing the driver to continue.
- `excel_writing_seconds`, `data_npz_writing_seconds`: per-case detail and exact NPZ writing. Summary checkpoint writing is excluded from these fields.
- `wall_clock_seconds`: elapsed case work through completion, including per-case outputs and the pre-MINLP checkpoint; excludes shared dataset parsing, startup compilation and the final checkpoint.

One Numba thread, one BLAS thread and one Gurobi thread are requested by default. Configuration, software versions, CPU/platform metadata and input-array SHA-256 fingerprints are recorded. Repeated simulation seeds are independent generated inputs; real datasets are run once rather than treated as independent replications. Sample standard deviations use ddof=1 and remain blank with one observation. Data are identical for exact/MINLP comparison and for sparse/non-sparse runs sharing the same simulated J/seed.

The default simulator is intentionally preserved: correlated stock-price paths with stream-specific drift regimes, rather than a well-specified single-shared-breakpoint signal model. A fitted shared breakpoint is not a recovery estimate for a known common synthetic change point.

## Validation evidence

See `validation/README.md` and `tests/test_methodology.py`. These checks support the implementation; they do not replace the paper's proofs or provide a formal symbolic certificate for every floating-point run.
