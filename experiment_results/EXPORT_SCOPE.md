# Experiment-results export

This directory contains the tracked PowerSGD/PowerSGD+ experiment archive and the Pi0.5 FP8 granularity precision run.

Included: JSON/CSV/TSV/JSONL metrics, manifests, summaries, experiment configurations, source launchers, analysis scripts, and bounded training logs.

Intentionally excluded: model checkpoints, optimizer shards, TensorBoard event caches, core dumps, generated caches, images, and individual raw log files larger than 5 MiB. The local archive and A800 remote paths remain the source of truth for those large artifacts; no successful result was overwritten.
