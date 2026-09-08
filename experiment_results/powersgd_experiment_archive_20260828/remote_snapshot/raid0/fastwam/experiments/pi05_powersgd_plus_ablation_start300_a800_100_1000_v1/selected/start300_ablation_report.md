# Pi0.5 PowerSGD+ isolated ablations (compression starts at update step 300)

Loss at step 300 is computed before update 300; the first potentially affected logged loss is step 301.

| Variant | First divergence | Affected MAE | Affected MAPE | Step time | Time change | Throughput change |
|---|---:|---:|---:|---:|---:|---:|
| exact_flush_start300 | 303 | 0.003358 | 2.157% | 3.134s | +221.30% | -66.96% |
| lr_rank2_start300 | 302 | 0.006291 | 4.084% | 3.178s | +225.75% | -67.36% |
| dynamic_period_start300 | 302 | 0.003233 | 2.078% | 3.167s | +224.69% | -67.21% |
| standard_powersgd_plus_start300 | 302 | 0.003247 | 2.088% | 3.120s | +219.86% | -66.68% |
