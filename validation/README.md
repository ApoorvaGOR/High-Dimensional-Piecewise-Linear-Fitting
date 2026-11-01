# Validation record

These are implementation checks, **not the completed 2,000-second manuscript benchmark matrix**. Some checks ran concurrently with other validation work, so their recorded timings should not be used as publication performance measurements. Run the default experiment driver in a controlled environment to collect those measurements.

## Completed checks

- Small problems: exhaustive support enumeration with independent least-squares residuals, dense breakpoint sampling and local scalar refinement. Random irregular time coordinates, all tested cardinalities, tied/duplicate gains, constants, noiseless non-grid breakpoints, coordinate invariance and repeated times are covered.
- Bounds: Appendix C formulas checked against all distinct-time secants and affine box corners.
- Source data: all three workbook byte identities match pinned GitHub blobs and SHA-256 hashes. Dataset shapes, both hands, all grid cells, full horizons and NASA RUL-derived indices are checked.
- Non-sparse exact scalability: all requested simulated J values through 100,000 at T=100, plus ERA5 J=40,401 and T=26, completed with independent recovered-residual SSE checks.
- Sparse exact scalability: simulated J=10,20,100,1000 at T=100, k=floor(0.2J), completed.
- Real-data integration: both settings completed for HoloAssist (1092×240) and NASA (21×192). MINLP integration used explicitly shortened **2-second** limits; TIME_LIMIT with no incumbent is recorded as such.
- Gurobi smoke: both settings at J=5,T=10 solved with OPTIMAL status (within Gurobi's default tolerances). The sparse smoke has k=1, exercising the corrected cardinality validation.
- Isolated MINLP execution: both smoke cases solved successfully; an intentional missing-input test verifies per-process error propagation. The current driver uses this execution route.
- Resume: completed smoke cases were skipped without creating duplicate result rows.
- PDF review: four single-page PDFs were rendered and visually inspected. A NASA subplot-spacing issue was corrected. All four final PDFs have **zero embedded raster images**, checked with pypdf.

The large ERA5 sparse exact check is recorded separately in `runs/era5_sparse/` while running. Its result is included in the consolidated CSV once completed; absence from that CSV means it has not completed in this validation record. Full-size MINLPs with the default 2,000-second limits have **not** been run as part of these installation/integration checks.

## Files

| File | Meaning |
|---|---|
| `methodology_tests.txt` | Automated numerical, data-adapter and worker tests |
| `completed_checks.csv` | Actual completed scalability/integration cases, retaining skip flags and shortened MINLP limits via the run labels/configurations |
| `check_configurations.json` | Actual command configurations for the consolidated validation rows |
| `example_summary.xlsx` | Example of the requested summary workbook layout from the isolated smoke run |
| `tested_environment.json` | Software/platform metadata from the tested environment |
| `tested-requirements.txt` | Exact installed Python package versions for this validation |
| `pdf_checks.json` | Final PDF page and raster-image counts |

Large raw/fitted arrays, per-case detailed workbooks and solver logs live in ignored `validation/runs/` in the working package; source ZIPs include the compact evidence above. Existing author-supplied results under `experiment_results_detailed/` are preserved and are not mixed into these validation rows.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python run_experiments_detailed.py --smoke --output-dir validation/runs/new_smoke
python run_experiments_detailed.py --datasets simulated era5 --settings nonsparse --skip-minlp --skip-figures --offline --output-dir validation/runs/new_dense
python run_experiments_detailed.py --datasets simulated --settings sparse --skip-minlp --skip-figures --output-dir validation/runs/new_sparse
python run_experiments_detailed.py --datasets cmapss holoassist --minlp-time-limit 2 --skip-figures --offline --output-dir validation/runs/new_real
python run_experiments_detailed.py --datasets era5 --settings sparse --skip-minlp --skip-figures --offline --output-dir validation/runs/new_era5_sparse
```

Download/cache workbooks first with `python datasets.py` if using a source-only checkout. Dataset tests skip when those optional local files are unavailable; numerical/worker tests do not require downloads or a Gurobi license. Integration MINLP runs require a working Gurobi license.
