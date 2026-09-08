# Pi0.5 precision-first low-rank mechanisms (shared step300 checkpoint)

All comparisons use the matched `baseline_resume` branch. With `start_iter=2`, updates 301–302 are dense, update 303 is first compressed, and loss step304 is the first potentially affected value.

The error-budget branch is a precision oracle: it uses a dense all-reduce to choose rank and must not be reported as a speed result. A800 timing is diagnostic only; Pro6K remains the speed-validation target.

| Variant | First divergence | Affected MAE | Affected MAPE | Mean step time |
|---|---:|---:|---:|---:|
| standard_powersgd_plus | 305 | 0.003206 | 2.062% | 3.142s |
| oracle_error_budget | 305 | 0.000295 | 0.188% | 5.708s |
| accordion_powersgd_plus | 305 | 0.006630 | 4.292% | 3.284s |

## Baseline reproducibility guard

| Comparison | MAE | MAPE | First divergence |
|---|---:|---:|---:|
| baseline_builder_vs_uninterrupted_reference | 0.000000 | 0.000% | None |
| baseline_resume_vs_builder | 0.000274 | 0.174% | 302 |
| baseline_resume_vs_uninterrupted_reference | 0.000274 | 0.174% | 302 |
