# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP gradient communication hooks."""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook,
    bf16_compress_hook,
    fp16_compress_hook,
)
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
    PowerSGDState,
    _orthogonalize,
    powerSGD_hook,
)

from ..utils import is_rank_zero

logger = logging.getLogger(__name__)


class PowerSGDPlusState(PowerSGDState):
    """PowerSGD state with periodic top-r subspace restarts.

    The restart follows PowerSGD+ (Xie et al., 2025): normal iterations use
    PyTorch's PowerSGD implementation, while every ``restart_period``
    compressed iterations all-reduce the full corrected gradient, compute a
    rank-r left singular subspace, and use it to reset the warm-start factors.
    ``restart_method="exact"`` uses a full SVD; ``"approximate"`` uses
    ``torch.svd_lowrank`` to avoid the prohibitive workspace of a full SVD on
    large-model gradients.
    """

    def __init__(
        self,
        *args,
        restart_period: int = 50,
        restart_method: str = "approximate",
        **kwargs,
    ):
        if restart_period <= 0:
            raise ValueError("PowerSGD+ restart_period must be greater than 0")
        if restart_method not in {"approximate", "exact"}:
            raise ValueError(
                "PowerSGD+ restart_method must be 'approximate' or 'exact'"
            )
        super().__init__(*args, **kwargs)
        self.restart_period = restart_period
        self.restart_method = restart_method
        # The optimizer can use this per-iteration, parameter-level signal to
        # freeze Adam's second moment only for tensors that actually went
        # through PowerSGD.  It is metadata (parameter ids), not a tensor
        # buffer, so the experiment does not add model-sized memory.
        self.compressed_param_ids: set[int] = set()
        self._compression_signal_iter = -1


class PowerSGDPlusErrorAveragingState(PowerSGDPlusState):
    """PowerSGD+ with infrequent cross-worker residual averaging.

    Local error feedback can leave each worker with a different residual.  The
    error-averaging variant periodically all-reduces that already-existing
    residual buffer and replaces it with the worker average.  This follows the
    low-frequency error-averaging safeguard from Step-Ahead Error Feedback,
    without allocating another model-sized buffer.
    """

    def __init__(self, *args, error_averaging_period: int = 20, **kwargs):
        if error_averaging_period <= 0:
            raise ValueError("error_averaging_period must be greater than 0")
        super().__init__(*args, **kwargs)
        self.error_averaging_period = error_averaging_period


def _record_powersgd_compression_signal(
    state: PowerSGDPlusState, bucket: dist.GradBucket, *, compressed: bool
) -> None:
    """Expose the current bucket's actual compressed parameters to the optimizer.

    DDP invokes one hook per bucket while ``state.iter`` is constant for an
    optimizer step.  The set is reset when that iteration changes (after the
    last bucket completes), then populated only for matrix-shaped tensors that
    satisfy PowerSGD's compression-rate test.  This keeps ``freeze-v`` exact
    for the communication hook without copying optimizer moments.
    """
    if state._compression_signal_iter != state.iter:
        state._compression_signal_iter = state.iter
        state.compressed_param_ids.clear()
    if not compressed or not hasattr(bucket, "parameters"):
        return
    parameters = list(bucket.parameters())
    gradients = list(bucket.gradients())
    if len(parameters) != len(gradients):
        return
    for parameter, tensor in zip(parameters, gradients):
        if tensor.ndim <= 1:
            continue
        rows, cols = tensor.shape[0], tensor.numel() // tensor.shape[0]
        rank = min(rows, cols, state.matrix_approximation_rank)
        if _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        ):
            state.compressed_param_ids.add(id(parameter))


class LossTriggeredPowerSGDPlusState(PowerSGDPlusState):
    """PowerSGD+ state whose full restart is triggered by loss drift.

    The hook reads the latest completed action-loss from the current run and
    the aligned dense-baseline metrics from shared JSONL files.  A restart is
    scheduled for the next hook iteration when the relative loss drift meets
    ``loss_threshold``.  This is an experiment-only oracle: it measures the
    relationship between model loss, learning rate, and the restart cadence;
    it is not intended as a production distributed policy.
    """

    def __init__(
        self,
        *args,
        loss_reference_path: str = "",
        loss_current_path: str = "",
        loss_threshold: float = 0.01,
        loss_event_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 < loss_threshold < 1:
            raise ValueError("loss_threshold must be in (0, 1)")
        self.loss_reference_path = loss_reference_path or os.getenv(
            "POWER_SGD_LOSS_REFERENCE", ""
        )
        self.loss_current_path = loss_current_path or os.getenv(
            "POWER_SGD_LOSS_CURRENT", ""
        )
        self.loss_threshold = loss_threshold
        self.loss_event_path = loss_event_path or os.getenv(
            "POWER_SGD_LOSS_EVENTS", ""
        )
        self._loss_last_observed_step = -1
        self._restart_this_iter = False
        self.last_loss_relative_error: float | None = None

    @staticmethod
    def _latest_metric(path: str) -> dict[str, float] | None:
        if not path:
            return None
        try:
            with open(path, "rb") as stream:
                lines = stream.readlines()[-64:]
        except OSError:
            return None
        for raw in reversed(lines):
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if "step" in record and "action_loss" in record:
                return record
        return None

    @staticmethod
    def _metric_at_step(path: str, step: int) -> dict[str, float] | None:
        """Return the metric aligned to ``step`` from a completed reference."""
        if not path:
            return None
        try:
            with open(path, "rb") as stream:
                lines = stream.readlines()
        except OSError:
            return None
        for raw in reversed(lines):
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if int(record.get("step", -1)) == step and "action_loss" in record:
                return record
        return None

    def prepare_loss_trigger(self) -> None:
        """Update the restart decision once at the first bucket of an iter."""
        self._restart_this_iter = False
        current = self._latest_metric(self.loss_current_path)
        if current is None:
            return
        observed_step = int(current["step"])
        reference = self._metric_at_step(self.loss_reference_path, observed_step)
        if reference is None:
            return
        if observed_step <= self._loss_last_observed_step:
            return
        self._loss_last_observed_step = observed_step
        baseline_loss = float(reference["action_loss"])
        current_loss = float(current["action_loss"])
        relative_error = abs(current_loss - baseline_loss) / max(
            abs(baseline_loss), 1e-12
        )
        self.last_loss_relative_error = relative_error
        if self.iter < self.start_powerSGD_iter:
            return
        if relative_error >= self.loss_threshold:
            self._restart_this_iter = True
            if self.loss_event_path and is_rank_zero():
                event = {
                    "hook_iter": self.iter,
                    "observed_step": observed_step,
                    "baseline_action_loss": baseline_loss,
                    "current_action_loss": current_loss,
                    "relative_error": relative_error,
                    "threshold": self.loss_threshold,
                    "lr": current.get("lr", reference.get("lr")),
                    "event": "loss_threshold_restart",
                }
                try:
                    with open(self.loss_event_path, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                except OSError:
                    logger.warning("Unable to write loss-trigger event", exc_info=True)



class DenseVPowerSGDPlusState(PowerSGDPlusState):
    """PowerSGD+ state that feeds Adam ``v`` from a pre-EF dense gradient."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_v_optimizer = None
        self.dense_v_allreduce_count = 0
        self.dense_v_bytes = 0

    def attach_optimizer(self, optimizer) -> None:
        self.dense_v_optimizer = optimizer

    def record_dense_v(self, bucket: dist.GradBucket, dense_gradient: torch.Tensor) -> None:
        if self.dense_v_optimizer is None:
            raise RuntimeError("dense-v hook ran before optimizer attachment")
        self.dense_v_optimizer.update_dense_v(list(bucket.parameters()), dense_gradient)
        self.dense_v_allreduce_count += 1
        self.dense_v_bytes += int(dense_gradient.numel() * dense_gradient.element_size())


def _future_value_tensor(future) -> torch.Tensor:
    value = future.value()
    return value[0] if isinstance(value, (list, tuple)) else value


def dense_v_powersgd_plus_hook(
    state: DenseVPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run PowerSGD+ while updating Adam ``v`` from an auxiliary pre-EF all-reduce."""
    if state.iter < state.start_powerSGD_iter:
        main_future = powerSGD_hook(state, bucket)

        def record_warmup(fut):
            # Warmup must follow native AdamW exactly.  The hook sees the
            # pre-clipping dense gradient, while AdamW's v uses the clipped
            # gradient; recording it here would alter the warmup trajectory.
            return _future_value_tensor(fut)

        return main_future.then(record_warmup)

    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    # Clone before powerSGD_hook/powerSGD_plus_hook adds the prior EF residual.
    raw_gradient = bucket.buffer().detach().clone()
    dense_future = dist.all_reduce(raw_gradient, group=group, async_op=True).get_future()
    main_future = powerSGD_plus_hook(state, bucket)

    def finish(main_done):
        dense_gradient = _future_value_tensor(dense_future).div_(world_size)
        state.record_dense_v(bucket, dense_gradient)
        return _future_value_tensor(main_done)

    return main_future.then(finish)


class EF21PowerSGDState(PowerSGDState):
    """State for memory-efficient EF21-style tracking with PowerSGD.

    All workers track the same averaged estimator.  After an exact
    initialization, communication is applied to each local innovation
    ``gradient - global_estimator`` and its averaged low-rank projection updates
    the shared estimator.  This global Markov form uses one gradient-sized
    buffer rather than the local-plus-global buffers of literal distributed
    EF21, which is essential for large models close to GPU memory capacity.

    This is deliberately separate from classic error feedback: there is no
    residual buffer that is added to the next raw gradient.  The optimizer sees
    the globally tracked estimator instead.
    """

    def __init__(
        self,
        *args,
        global_step_offset: int = 0,
        compression_start_step: int = 300,
        log_interval: int = 25,
        **kwargs,
    ):
        if global_step_offset < 0:
            raise ValueError("PowerSGD global_step_offset must be non-negative")
        if compression_start_step <= 0:
            raise ValueError("EF21 compression_start_step must be positive")
        if log_interval <= 0:
            raise ValueError("EF21 log_interval must be positive")
        super().__init__(*args, **kwargs)
        self.global_step_offset = global_step_offset
        self.compression_start_step = compression_start_step
        self.log_interval = log_interval
        self.global_estimator_dict: dict[int, torch.Tensor] = {}

    def global_step(self) -> int:
        return self.global_step_offset + self.iter + 1


def _parse_step_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Parse inclusive, one-indexed ranges such as ``101-150,201-260``."""
    ranges: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if match is None:
            raise ValueError(
                "PowerSGD window ranges must be comma-separated STEP or "
                f"START-END values, got {item!r}"
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start <= 0 or end < start:
            raise ValueError(f"Invalid PowerSGD window range {item!r}")
        if ranges and start <= ranges[-1][1]:
            raise ValueError("PowerSGD window ranges must be sorted and disjoint")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("At least one PowerSGD window range is required")
    return tuple(ranges)


class WindowedPowerSGDPlusState(PowerSGDPlusState):
    """PowerSGD+ state gated by fixed global-step windows.

    Outside configured windows the hook performs an exact all-reduce. On the
    first exact iteration after a compressed window it includes the local
    error-feedback residual, then discards the old P/Q basis. This prevents a
    stale residual or subspace from leaking into a later window.
    """

    def __init__(
        self,
        *args,
        window_ranges: str,
        global_step_offset: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if global_step_offset < 0:
            raise ValueError("PowerSGD global_step_offset must be non-negative")
        self.window_ranges = _parse_step_ranges(window_ranges)
        self.global_step_offset = global_step_offset
        self.compressed_iter_in_window = 0
        self._prepared_iter = -1
        self._active = False
        self._window_index: int | None = None

    def global_step(self) -> int:
        return self.global_step_offset + self.iter + 1

    def window_index_for_step(self, step: int) -> int | None:
        for index, (start, end) in enumerate(self.window_ranges):
            if start <= step <= end:
                return index
            if step < start:
                break
        return None

    def prepare_iteration(self) -> None:
        if self._prepared_iter == self.iter:
            return
        self._prepared_iter = self.iter
        next_window = self.window_index_for_step(self.global_step())
        next_active = next_window is not None
        if next_active and next_window != self._window_index:
            # Any residual from the preceding window should have been flushed
            # by its first exact iteration. Clear defensively before a new
            # window and cold-start its low-rank factors.
            self.error_dict.clear()
            self.p_memory_dict.clear()
            self.q_memory_dict.clear()
            self.compressed_iter_in_window = 0
        if next_active != self._active and is_rank_zero():
            logger.info(
                "Windowed PowerSGD+ transition global_step=%d active=%s "
                "window_index=%s",
                self.global_step(),
                next_active,
                next_window,
            )
        self._active = next_active
        self._window_index = next_window

    def maybe_increase_iter(self, bucket: dist.GradBucket) -> None:
        was_last = bucket.is_last()
        was_active = self._active
        super().maybe_increase_iter(bucket)
        if was_last and was_active:
            self.compressed_iter_in_window += 1


def _parse_period_schedule(text: str) -> tuple[tuple[int, int, int], ...]:
    """Parse inclusive global-step period ranges such as ``300-452:25``."""
    schedule: list[tuple[int, int, int]] = []
    for item in text.split(","):
        match = re.fullmatch(r"\s*(\d+)-(\d+):(\d+)\s*", item)
        if match is None:
            raise ValueError(f"Invalid PowerSGD period schedule item {item!r}")
        start, end, period = map(int, match.groups())
        if start <= 0 or end < start or period <= 0:
            raise ValueError(f"Invalid PowerSGD period schedule item {item!r}")
        if schedule and start != schedule[-1][1] + 1:
            raise ValueError("PowerSGD period schedule ranges must be contiguous")
        schedule.append((start, end, period))
    if not schedule:
        raise ValueError("PowerSGD period schedule cannot be empty")
    return tuple(schedule)


class ScheduledPowerSGDPlusState(WindowedPowerSGDPlusState):
    """Absolute-step PowerSGD+ policy used by isolated ablation experiments."""

    def __init__(
        self,
        *args,
        policy: str,
        compression_start_step: int,
        high_rank: int = 2,
        high_rank_end_step: int = 452,
        period_schedule: str = "300-452:25,453-734:50,735-1000:100",
        **kwargs,
    ):
        if policy not in {"exact_flush", "lr_rank", "dynamic_period"}:
            raise ValueError(f"Unsupported scheduled PowerSGD+ policy {policy!r}")
        if compression_start_step <= 0:
            raise ValueError("PowerSGD compression_start_step must be positive")
        if high_rank <= 0:
            raise ValueError("PowerSGD high_rank must be positive")
        if high_rank_end_step < compression_start_step:
            raise ValueError("PowerSGD high_rank_end_step precedes compression start")
        super().__init__(
            *args,
            window_ranges=f"{compression_start_step}-1000000000",
            **kwargs,
        )
        self.policy = policy
        self.compression_start_step = compression_start_step
        self.base_rank = self.matrix_approximation_rank
        self.high_rank = high_rank
        self.high_rank_end_step = high_rank_end_step
        self.period_schedule = _parse_period_schedule(period_schedule)
        if self.period_schedule[0][0] != compression_start_step:
            raise ValueError("PowerSGD period schedule must start at compression_start_step")
        self._policy_prepared_iter = -1
        self._restart_this_iter = False
        self._current_period = self.restart_period
        self._last_restart_compressed_iter: int | None = None

    def period_for_step(self, step: int) -> int:
        for start, end, period in self.period_schedule:
            if start <= step <= end:
                return period
        raise ValueError(f"No dynamic PowerSGD period configured for step {step}")

    def prepare_policy_iteration(self) -> None:
        self.prepare_iteration()
        if self._policy_prepared_iter == self.iter:
            return
        self._policy_prepared_iter = self.iter
        self._restart_this_iter = False
        if not self._active:
            return
        step = self.global_step()
        if self.policy == "lr_rank":
            desired_rank = (
                self.high_rank if step <= self.high_rank_end_step else self.base_rank
            )
            if desired_rank != self.matrix_approximation_rank:
                self.matrix_approximation_rank = desired_rank
                self.p_memory_dict.clear()
                self.q_memory_dict.clear()
                if is_rank_zero():
                    logger.info(
                        "Scheduled PowerSGD+ rank transition global_step=%d rank=%d",
                        step,
                        desired_rank,
                    )
        if self.policy == "dynamic_period":
            self._current_period = self.period_for_step(step)
            elapsed = (
                None
                if self._last_restart_compressed_iter is None
                else self.compressed_iter_in_window
                - self._last_restart_compressed_iter
            )
            self._restart_this_iter = elapsed is None or elapsed >= self._current_period
        else:
            self._restart_this_iter = (
                self.compressed_iter_in_window % self.restart_period == 0
            )

    def maybe_increase_iter(self, bucket: dist.GradBucket) -> None:
        was_last = bucket.is_last()
        was_active = self._active
        restart_this_iter = self._restart_this_iter
        compressed_iter = self.compressed_iter_in_window
        super().maybe_increase_iter(bucket)
        if was_last and was_active and restart_this_iter:
            self._last_restart_compressed_iter = compressed_iter


class DynamicPowerSGDState(PowerSGDState):
    """PowerSGD state with Accordion-style gradient-norm rank adaptation.

    The reducer measures the global sum of per-worker squared gradient norms
    every ``adapt_interval`` iterations.  A sufficiently large relative norm
    change marks a critical regime and selects ``high_rank`` for the following
    iteration; otherwise ``low_rank`` is used.  Rank transitions discard only
    the shape-dependent P/Q factors while preserving error feedback.
    """

    def __init__(
        self,
        *args,
        low_rank: int = 1,
        high_rank: int = 4,
        adapt_interval: int = 5,
        relative_change_threshold: float = 0.1,
        **kwargs,
    ):
        if low_rank <= 0 or high_rank < low_rank:
            raise ValueError(
                "Dynamic PowerSGD ranks must satisfy 0 < low_rank <= high_rank"
            )
        if adapt_interval <= 0:
            raise ValueError("Dynamic PowerSGD adapt_interval must be greater than 0")
        if relative_change_threshold < 0:
            raise ValueError(
                "Dynamic PowerSGD relative_change_threshold must be non-negative"
            )
        kwargs["matrix_approximation_rank"] = low_rank
        super().__init__(*args, **kwargs)
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.adapt_interval = adapt_interval
        self.relative_change_threshold = relative_change_threshold
        self.previous_global_grad_norm: float | None = None
        self.pending_rank: int | None = None
        self.norm_sq_accumulator: torch.Tensor | None = None
        self.last_rank_change_iter = -1


class LayerwisePowerSGDState(PowerSGDState):
    """Error-aware per-tensor PowerSGD with periodic dense safeguards.

    A dense refresh first averages the corrected bucket, estimates each
    parameter matrix's leading singular spectrum, and chooses the smallest
    candidate rank whose relative Frobenius reconstruction error meets the
    configured budget.  The resulting rank plan is shared implicitly because
    it is derived from the same globally averaged gradient on every worker.
    Between refreshes, tensors use variable-rank PowerSGD with error feedback.

    Refresh iterations apply the dense averaged gradient directly and reset
    the corresponding residual.  This is intentionally closer to the
    safeguard used by semi-lazy/greedy low-rank methods than to a one-time
    training warm-up: it corrects subspace drift throughout training.
    """

    def __init__(
        self,
        *args,
        candidate_ranks: tuple[int, ...] = (1, 2, 4, 8, 16),
        relative_error_threshold: float = 0.1,
        refresh_period: int = 50,
        power_iterations: int = 2,
        parameter_names: dict[int, str] | None = None,
        uncompressed_patterns: str = "",
        log_tensor_stats: bool = False,
        **kwargs,
    ):
        candidate_ranks = tuple(sorted(set(candidate_ranks)))
        if not candidate_ranks or candidate_ranks[0] <= 0:
            raise ValueError(
                "Layerwise PowerSGD candidate ranks must be positive"
            )
        if not 0 <= relative_error_threshold < 1:
            raise ValueError(
                "Layerwise PowerSGD relative error threshold must be in [0, 1)"
            )
        if refresh_period <= 0:
            raise ValueError(
                "Layerwise PowerSGD refresh period must be greater than 0"
            )
        if power_iterations < 0:
            raise ValueError(
                "Layerwise PowerSGD power iterations must be non-negative"
            )
        super().__init__(*args, **kwargs)
        self.candidate_ranks = candidate_ranks
        self.relative_error_threshold = relative_error_threshold
        self.refresh_period = refresh_period
        self.power_iterations = power_iterations
        self.parameter_names = parameter_names or {}
        self.uncompressed_pattern_text = uncompressed_patterns
        self.uncompressed_regex = (
            re.compile(uncompressed_patterns) if uncompressed_patterns else None
        )
        self.log_tensor_stats = log_tensor_stats
        # Optional static eligibility gates used by precision-only ablations.
        # They are environment-controlled so older launchers remain compatible
        # with the hook API.  A matching parameter must be a sufficiently large
        # GEMM weight before it can receive a low-rank plan; all other tensors
        # remain exact all-reduce tensors.
        self.selective_min_tensor_bytes = int(
            os.environ.get("POWER_SGD_SELECTIVE_MIN_BYTES", "0")
        )
        selective_regex = os.environ.get("POWER_SGD_SELECTIVE_GEMM_REGEX", "")
        self.selective_gemm_regex = (
            re.compile(selective_regex) if selective_regex else None
        )
        self.rank_plan_dict: dict[int, list[int]] = {}
        self.rank_error_dict: dict[int, list[float]] = {}
        self.group_plan_dict: dict[
            int, list[tuple[list[int], int, int, int, int, int]]
        ] = {}


class OracleErrorBudgetPowerSGDState(LayerwisePowerSGDState):
    """Precision-oracle state for per-step, per-tensor rank selection.

    This state is intentionally designed for mechanism validation, not for a
    communication speedup.  Every compressed iteration first materializes the
    exact globally averaged corrected gradient, then applies the smallest
    candidate-rank projection satisfying the configured Frobenius error budget.
    The optimizer only sees the selected low-rank projection, while the dense
    average is used solely as an oracle for the rank decision.  A later systems
    implementation can replace the dense oracle with sketches or cached plans.
    """

    def __init__(self, *args, random_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = random_seed
        # Low-overhead cumulative profile used by the precision-only oracle
        # experiment. Counts alone are misleading because a bias vector and a
        # multi-million-element matrix each count as one tensor decision.
        self.profile_by_rank: dict[str, dict[str, int]] = {}
        self.profile_by_size_bin: dict[str, dict[str, int]] = {}


def _oracle_size_bin(numel: int) -> str:
    if numel < 4 * 1024:
        return "lt4K"
    if numel < 64 * 1024:
        return "4K_64K"
    if numel < 1024 * 1024:
        return "64K_1M"
    if numel < 16 * 1024 * 1024:
        return "1M_16M"
    return "ge16M"


class AccordionPowerSGDPlusState(PowerSGDPlusState):
    """PowerSGD+ with windowed gradient-norm critical-regime protection.

    The original Accordion paper accumulates gradients over coarse windows and
    treats a large relative norm change as a critical learning regime.  Keeping
    a full accumulated gradient for Pi0.5 would add another model-sized buffer,
    so this memory-safe implementation accumulates per-tensor squared norms.
    Decisions are made independently per DDP bucket; a bucket is communicated
    densely for ``protection_steps`` when any tensor in it crosses the relative
    change threshold.  Outside protected windows normal PowerSGD+ is used.
    """

    def __init__(
        self,
        *args,
        detection_window: int = 50,
        relative_change_threshold: float = 0.5,
        protection_steps: int = 25,
        parameter_names: dict[int, str] | None = None,
        uncompressed_patterns: str = "",
        power_iterations: int = 2,
        log_tensor_stats: bool = False,
        **kwargs,
    ):
        if detection_window <= 0:
            raise ValueError("Accordion detection_window must be positive")
        if relative_change_threshold < 0:
            raise ValueError(
                "Accordion relative_change_threshold must be non-negative"
            )
        if protection_steps <= 0:
            raise ValueError("Accordion protection_steps must be positive")
        super().__init__(*args, **kwargs)
        self.detection_window = detection_window
        self.relative_change_threshold = relative_change_threshold
        self.protection_steps = protection_steps
        # Use the norm signal to make a per-tensor rank-1/dense plan.  The
        # layerwise execution path below reuses the existing residual and
        # grouped P/Q buffers, so no extra model-sized telemetry is needed.
        self.norm_selective = True
        self.refresh_period = detection_window
        self.candidate_ranks = (1,)
        self.relative_error_threshold = 1.0
        self.power_iterations = power_iterations
        self.parameter_names = parameter_names or {}
        self.uncompressed_pattern_text = uncompressed_patterns
        self.uncompressed_regex = (
            re.compile(uncompressed_patterns) if uncompressed_patterns else None
        )
        self.log_tensor_stats = log_tensor_stats
        self.rank_plan_dict: dict[int, list[int]] = {}
        self.rank_error_dict: dict[int, list[float]] = {}
        self.group_plan_dict: dict[
            int, list[tuple[list[int], int, int, int, int, int]]
        ] = {}
        self.critical_tensor_mask_dict: dict[int, list[bool]] = {}
        self.norm_energy_dict: dict[int, torch.Tensor] = {}
        self.raw_norm_energy_dict: dict[int, torch.Tensor] = {}
        self.residual_energy_dict: dict[int, torch.Tensor] = {}
        self.previous_window_norm_dict: dict[int, torch.Tensor] = {}
        self.last_window_norm_energy_dict: dict[int, torch.Tensor] = {}
        self.critical_until_iter_dict: dict[int, int] = {}
        # Per-tensor expiry for norm-triggered dense protection.  A bucket
        # level expiry is insufficient because only the tensors whose norm
        # changed sharply should be protected; the rest remain rank-1.
        self.critical_until_tensor_iter_dict: dict[int, list[int]] = {}
        # Window-level coverage diagnostics.  These are scalar metadata only;
        # they do not retain another gradient-sized buffer.
        self.last_window_total_numel_dict: dict[int, int] = {}
        self.last_window_critical_numel_dict: dict[int, int] = {}
        self.last_window_rank1_eligible_numel_dict: dict[int, int] = {}
        self.last_window_rank1_payload_numel_dict: dict[int, int] = {}

    def bucket_is_critical(self, bucket_index: int) -> bool:
        return self.iter < self.critical_until_iter_dict.get(bucket_index, -1)


class ACPSGDState(PowerSGDState):
    """State for Alternate Compressed PowerSGD (ACP-SGD).

    ACP-SGD communicates P and Q on alternating optimization iterations.  It
    retains both factors and the local compression residual for every bucket.
    """


class ACPPowerSGDPlusState(PowerSGDPlusState):
    """ACP-SGD state with periodic PowerSGD+ global-subspace restarts.

    Normal compressed iterations communicate P and Q on alternating steps.
    Every ``restart_period`` compressed iterations the PowerSGD+ restart path
    refreshes both factors from the globally averaged corrected gradient.
    """


class SeparateState:
    """State for SEPARATE common-random-projection gradient compression."""

    def __init__(
        self,
        process_group,
        *,
        start_iter: int = 2,
        compression_ratio: float = 16.0,
        min_compression_rate: float = 2.0,
        error_feedback_beta: float = 0.95,
        error_reset_interval: int = 128,
        random_seed: int = 0,
    ):
        if start_iter <= 1:
            raise ValueError(
                "SEPARATE start_iter must be greater than 1 because it keeps state"
            )
        if compression_ratio <= 1:
            raise ValueError("SEPARATE compression_ratio must be greater than 1")
        if min_compression_rate <= 0:
            raise ValueError(
                "SEPARATE min_compression_rate must be greater than 0"
            )
        if not 0 <= error_feedback_beta <= 1:
            raise ValueError("SEPARATE error_feedback_beta must be in [0, 1]")
        if error_reset_interval <= 0:
            raise ValueError(
                "SEPARATE error_reset_interval must be greater than 0"
            )
        self.process_group = process_group
        self.start_iter = start_iter
        self.compression_ratio = compression_ratio
        self.min_compression_rate = min_compression_rate
        self.error_feedback_beta = error_feedback_beta
        self.error_reset_interval = error_reset_interval
        self.random_seed = random_seed
        self.error_dict: dict[int, torch.Tensor] = {}
        self.iter = 0

    def maybe_increase_iter(self, bucket: dist.GradBucket) -> None:
        if bucket.is_last():
            self.iter += 1

# Comm hooks eligible to be resolved by ``resolve_comm_hook``. All of them
# share the ``(process_group, bucket) -> Future`` signature.
_SUPPORTED_COMM_HOOKS = {
    "allreduce_hook": allreduce_hook,
    "fp16_compress_hook": fp16_compress_hook,
    "bf16_compress_hook": bf16_compress_hook,
    "powersgd_hook": powerSGD_hook,
    "powersgd_fp32_hook": None,  # Set after the function definition below.
    "ef21_powersgd_hook": None,
    "powersgd_plus_hook": None,  # Set after the function definition below.
    "windowed_powersgd_plus_hook": None,
    "exact_flush_powersgd_plus_hook": None,
    "lr_rank_powersgd_plus_hook": None,
    "dynamic_period_powersgd_plus_hook": None,
    "loss_triggered_powersgd_plus_hook": None,
    "acp_powersgd_plus_hook": None,
    "dynamic_powersgd_hook": None,
    "layerwise_powersgd_hook": None,
    "acp_sgd_hook": None,
    "separate_hook": None,
}


def _should_powersgd_compress(
    rows: int,
    cols: int,
    rank: int,
    min_compression_rate: float,
) -> bool:
    """Match PyTorch PowerSGD's payload-size eligibility test."""
    return (rows + cols) * rank * min_compression_rate < rows * cols


def powerSGD_fp32_hook(
    state: PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run PowerSGD with FP32 factors for compressed gradient matrices.

    DDP buckets keep their original dtype. Compressible matrices are promoted
    only for the P/Q matmuls, orthogonalization, and factor all-reduces; the
    reconstruction is copied back to the original bucket dtype. Uncompressed
    tensors and error-feedback buffers deliberately retain the bucket dtype so
    this hook isolates the numerical and communication cost of FP32 factors.
    """
    group = (
        state.process_group
        if state.process_group is not None
        else dist.group.WORLD
    )
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()

    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(lambda fut: fut.value()[0].div_(world_size))

    device = input_tensor.device
    bucket_dtype = input_tensor.dtype
    factor_dtype = torch.float32
    bucket_index = bucket.index()

    input_tensor_copy = None
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            state.error_dict[bucket_index] = torch.zeros_like(input_tensor)
        input_tensor_copy = input_tensor.detach().clone()

    tensors_to_compress = []
    uncompressed_tensors = []
    total_p_size = 0
    total_q_size = 0
    for tensor in bucket.gradients():
        matrix = tensor.view(tensor.shape[0], -1)
        rows, cols = matrix.shape
        rank = min(rows, cols, state.matrix_approximation_rank)
        should_compress = _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        )
        state.total_numel_before_compression += rows * cols
        if should_compress:
            tensors_to_compress.append(matrix)
            total_p_size += rows * rank
            total_q_size += cols * rank
            state.total_numel_after_compression += (rows + cols) * rank
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += rows * cols

    uncompressed_memory = (
        torch.cat([tensor.reshape(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.empty(0, device=device, dtype=bucket_dtype)
    )

    must_allocate = (
        not state.warm_start
        or bucket_index not in state.p_memory_dict
        or state.p_memory_dict[bucket_index].numel() != total_p_size
        or state.q_memory_dict[bucket_index].numel() != total_q_size
        or state.p_memory_dict[bucket_index].dtype != factor_dtype
        or state.q_memory_dict[bucket_index].dtype != factor_dtype
    )
    if must_allocate:
        state.p_memory_dict[bucket_index] = torch.empty(
            total_p_size, device=device, dtype=factor_dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_q_size, device=device, dtype=factor_dtype
        )

    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        shape_to_tensors[tensor.shape].append(tensor)

    def batched_tensors():
        for same_shape_tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                if len(same_shape_tensors) == 1:
                    yield same_shape_tensors[0].unsqueeze(0)
                else:
                    yield torch.stack(same_shape_tensors)
            else:
                for tensor in same_shape_tensors:
                    yield tensor.unsqueeze(0)

    compressed_batches = []
    ps = []
    qs = []
    p_offset = 0
    q_offset = 0
    for tensor in batched_tensors():
        batch_size, rows, cols = tensor.shape
        rank = min(rows, cols, state.matrix_approximation_rank)
        compressed_batches.append(tensor)
        ps.append(
            state.p_memory_dict[bucket_index][
                p_offset : p_offset + batch_size * rows * rank
            ].view(batch_size, rows, rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_offset : q_offset + batch_size * cols * rank
            ].view(batch_size, cols, rank)
        )
        p_offset += batch_size * rows * rank
        q_offset += batch_size * cols * rank

    if not must_allocate:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(torch.randn(q.shape, device="cpu", dtype=factor_dtype))
                _orthogonalize(q, state.orthogonalization_epsilon)

    for tensor, q, p in zip(compressed_batches, qs, ps):
        torch.bmm(tensor.float(), q, out=p)

    uncompressed_future = dist.all_reduce(
        uncompressed_memory, group=group, async_op=True
    ).get_future()

    def unpack_uncompressed_and_reduce_ps(fut):
        reduced = fut.value()[0].div_(world_size)
        offset = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(reduced[offset : offset + tensor.numel()].view_as(tensor))
            offset += tensor.numel()
        return (
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)
        for tensor, p, q in zip(compressed_batches, ps, qs):
            torch.bmm(tensor.transpose(1, 2).float(), p, out=q)
        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        for p, q, tensor in zip(ps, qs, compressed_batches):
            tensor.copy_(torch.bmm(p, q.transpose(1, 2)))

        if state.batch_tensors_with_same_shape:
            for tensor in compressed_batches:
                if tensor.shape[0] == 1:
                    continue
                for index, original in enumerate(shape_to_tensors[tensor.shape[1:]]):
                    original.copy_(tensor[index])

        if input_tensor.is_cuda:
            torch.cuda.synchronize(device)
        if state.use_error_feedback:
            assert input_tensor_copy is not None
            state.error_dict[bucket_index] = input_tensor_copy - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        state.maybe_increase_iter(bucket)
        return input_tensor

    return (
        uncompressed_future.then(unpack_uncompressed_and_reduce_ps)
        .then(compute_qs)
        .then(decompress)
    )


_SUPPORTED_COMM_HOOKS["powersgd_fp32_hook"] = powerSGD_fp32_hook


def ef21_powerSGD_hook(
    state: EF21PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Track a shared gradient estimator and compress only its innovations.

    For a shared orthonormal PowerSGD basis ``P``, worker ``i`` projects
    ``gradient_i - global_estimator`` as ``P P.T delta_i``.  All-reducing the
    corresponding Q factors produces the average projection, which updates the
    shared estimator returned to DDP.  This is an EF21-style global Markov
    compressor, not literal per-worker EF21.

    The last exact iteration before ``compression_start_step`` initializes
    both estimators.  The configured start step therefore communicates its
    innovation and can affect the following optimizer loss.
    """
    group = (
        state.process_group
        if state.process_group is not None
        else dist.group.WORLD
    )
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    current_step = state.global_step()

    if current_step < state.compression_start_step:
        if current_step == state.compression_start_step - 1:
            def initialize_from_exact_prefix(fut):
                reduced = fut.value()[0].div_(world_size)
                state.global_estimator_dict[bucket_index] = (
                    reduced.detach().clone()
                )
                if is_rank_zero():
                    logger.info(
                        "EF21 PowerSGD initialized global_step=%d bucket=%d "
                        "numel=%d",
                        current_step,
                        bucket_index,
                        input_tensor.numel(),
                    )
                state.maybe_increase_iter(bucket)
                return reduced

            return dist.all_reduce(
                input_tensor, group=group, async_op=True
            ).get_future().then(initialize_from_exact_prefix)
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(lambda fut: fut.value()[0].div_(world_size))

    global_estimator = state.global_estimator_dict.get(bucket_index)
    needs_initialization = (
        global_estimator is None
        or global_estimator.shape != input_tensor.shape
    )
    if needs_initialization:
        def initialize_estimators(fut):
            reduced = fut.value()[0].div_(world_size)
            state.global_estimator_dict[bucket_index] = reduced.detach().clone()
            state.p_memory_dict.pop(bucket_index, None)
            state.q_memory_dict.pop(bucket_index, None)
            if is_rank_zero():
                logger.info(
                    "EF21 PowerSGD initialized global_step=%d bucket=%d numel=%d",
                    current_step,
                    bucket_index,
                    input_tensor.numel(),
                )
            state.maybe_increase_iter(bucket)
            return reduced

        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(initialize_estimators)

    assert global_estimator is not None
    input_tensor.sub_(global_estimator)
    device = input_tensor.device
    dtype = input_tensor.dtype
    collect_stats = current_step % state.log_interval == 0
    innovation_sq: torch.Tensor | None = None

    compressed: list[tuple[torch.Tensor, int]] = []
    uncompressed: list[torch.Tensor] = []
    p_size = 0
    q_size = 0
    for tensor in bucket.gradients():
        matrix = tensor.view(tensor.shape[0], -1)
        rows, cols = matrix.shape
        rank = min(rows, cols, state.matrix_approximation_rank)
        if collect_stats:
            matrix_sq = torch.linalg.vector_norm(
                matrix, dtype=torch.float32
            ).square()
            innovation_sq = (
                matrix_sq if innovation_sq is None else innovation_sq + matrix_sq
            )
        state.total_numel_before_compression += rows * cols
        if _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        ):
            compressed.append((matrix, rank))
            p_size += rows * rank
            q_size += cols * rank
            state.total_numel_after_compression += (rows + cols) * rank
        else:
            uncompressed.append(tensor)
            state.total_numel_after_compression += rows * cols

    # If a bucket has no eligible matrices, reduce its innovation exactly.
    if not compressed:
        def update_exact_estimator(fut):
            reduced_innovation = fut.value()[0].div_(world_size)
            global_estimator.add_(reduced_innovation)
            input_tensor.copy_(global_estimator)
            if collect_stats and is_rank_zero():
                assert innovation_sq is not None
                logger.info(
                    "EF21 PowerSGD stats global_step=%d bucket=%d "
                    "innovation_norm=%.8g projection_error_norm=0 "
                    "relative_projection_error=0 compressed_numel=0 total_numel=%d",
                    current_step,
                    bucket_index,
                    math.sqrt(innovation_sq.item()),
                    input_tensor.numel(),
                )
            state.maybe_increase_iter(bucket)
            return input_tensor

        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(update_exact_estimator)

    must_allocate = (
        bucket_index not in state.p_memory_dict
        or bucket_index not in state.q_memory_dict
        or state.p_memory_dict[bucket_index].numel() != p_size
        or state.q_memory_dict[bucket_index].numel() != q_size
    )
    if must_allocate:
        state.p_memory_dict[bucket_index] = torch.empty(
            p_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            q_size, device=device, dtype=dtype
        )

    ps: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    p_offset = 0
    q_offset = 0
    for matrix, rank in compressed:
        rows, cols = matrix.shape
        ps.append(
            state.p_memory_dict[bucket_index][
                p_offset : p_offset + rows * rank
            ].view(rows, rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_offset : q_offset + cols * rank
            ].view(cols, rank)
        )
        p_offset += rows * rank
        q_offset += cols * rank

    if must_allocate:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(torch.randn(q.shape, device="cpu", dtype=dtype))
                _orthogonalize(q.unsqueeze(0), state.orthogonalization_epsilon)
    else:
        for q in qs:
            _orthogonalize(q.unsqueeze(0), state.orthogonalization_epsilon)

    for (matrix, _), q, p in zip(compressed, qs, ps):
        torch.mm(matrix, q, out=p)

    uncompressed_memory = (
        torch.cat([tensor.reshape(-1) for tensor in uncompressed])
        if uncompressed
        else None
    )
    uncompressed_future = (
        dist.all_reduce(
            uncompressed_memory, group=group, async_op=True
        ).get_future()
        if uncompressed_memory is not None
        else None
    )
    projection_error_sq: torch.Tensor | None = None

    def compute_local_qs(fut):
        nonlocal projection_error_sq
        state.p_memory_dict[bucket_index] = fut.value()[0]
        for p in ps:
            _orthogonalize(p.unsqueeze(0), state.orthogonalization_epsilon)
        for (matrix, _), p, q in zip(compressed, ps, qs):
            torch.mm(matrix.transpose(0, 1), p, out=q)
            if collect_stats:
                # P is orthonormal and Q=D.T@P, hence
                # ||D-PQ.T||_F^2 = ||D||_F^2-||Q||_F^2.  Avoiding an explicit
                # residual prevents a gradient-sized temporary allocation.
                matrix_sq = torch.linalg.vector_norm(
                    matrix, dtype=torch.float32
                ).square()
                projection_sq = torch.linalg.vector_norm(
                    q, dtype=torch.float32
                ).square()
                error_sq = torch.clamp(matrix_sq - projection_sq, min=0.0)
                projection_error_sq = (
                    error_sq
                    if projection_error_sq is None
                    else projection_error_sq + error_sq
                )
        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index],
                group=group,
                async_op=True,
            )
            .get_future()
            .wait()[0]
        )

    def update_global_estimator(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        for (matrix, _), p, q in zip(compressed, ps, qs):
            torch.mm(p, q.transpose(0, 1), out=matrix)

        if uncompressed_future is not None:
            reduced = uncompressed_future.wait()[0].div_(world_size)
            reduced_offset = 0
            for tensor in uncompressed:
                tensor.copy_(
                    reduced[
                        reduced_offset : reduced_offset + tensor.numel()
                    ].view_as(tensor)
                )
                reduced_offset += tensor.numel()

        global_estimator.add_(input_tensor)
        input_tensor.copy_(global_estimator)
        if collect_stats and is_rank_zero():
            assert innovation_sq is not None
            error_sq_value = (
                projection_error_sq.item()
                if projection_error_sq is not None
                else 0.0
            )
            innovation_sq_value = innovation_sq.item()
            relative_error = math.sqrt(
                error_sq_value / max(innovation_sq_value, 1e-30)
            )
            logger.info(
                "EF21 PowerSGD stats global_step=%d bucket=%d "
                "innovation_norm=%.8g projection_error_norm=%.8g "
                "relative_projection_error=%.8g compressed_numel=%d "
                "total_numel=%d",
                current_step,
                bucket_index,
                math.sqrt(innovation_sq_value),
                math.sqrt(error_sq_value),
                relative_error,
                sum(matrix.numel() for matrix, _ in compressed),
                input_tensor.numel(),
            )
        state.maybe_increase_iter(bucket)
        return input_tensor

    return (
        dist.all_reduce(
            state.p_memory_dict[bucket_index], group=group, async_op=True
        )
        .get_future()
        .then(compute_local_qs)
        .then(update_global_estimator)
    )


_SUPPORTED_COMM_HOOKS["ef21_powersgd_hook"] = ef21_powerSGD_hook


def _approximate_top_left_subspace(
    matrix: torch.Tensor,
    rank: int,
    niter: int = 2,
) -> torch.Tensor:
    """Approximate leading left singular vectors by subspace iteration.

    Keeping exactly ``rank`` basis vectors avoids SVD/eigendecomposition
    convergence failures on ill-conditioned real-world gradient matrices.
    """
    rows, cols = matrix.shape
    q_rank = min(min(rows, cols), rank)
    q = torch.randn(cols, q_rank, device=matrix.device, dtype=matrix.dtype)
    p = torch.linalg.qr(matrix @ q, mode="reduced").Q
    for _ in range(niter):
        q = torch.linalg.qr(matrix.transpose(0, 1) @ p, mode="reduced").Q
        p = torch.linalg.qr(matrix @ q, mode="reduced").Q
    return p


def _powersgd_plus_svd_restart(
    state: PowerSGDPlusState,
    bucket: dist.GradBucket,
    *,
    apply_exact: bool = False,
) -> torch.futures.Future[torch.Tensor]:
    """Run one periodic PowerSGD+ restart for a DDP bucket.

    Restart steps deliberately use synchronous collectives.  An earlier
    implementation chained asynchronous NCCL work from a Future callback and
    wrote directly into the DDP bucket from that callback.  With
    parameters-as-bucket-view enabled, DDP/autograd could reuse the bucket
    before the callback's CUDA kernels had finished, which manifested as an
    illegal memory access in the NCCL watchdog.  Restart is infrequent, so
    blocking here is a safe trade-off for deterministic buffer ownership.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    output_tensors = bucket.gradients()
    if bucket_index == 0 and is_rank_zero():
        logger.info(
            "%s restart: iter=%d compressed_iter=%d method=%s rank=%d apply_exact=%s",
            type(state).__name__,
            state.iter,
            state.iter - state.start_powerSGD_iter,
            state.restart_method,
            state.matrix_approximation_rank,
            apply_exact,
        )

    # DDP may assign the same bucket index independently per parameter dtype.
    # An all-1D/uncompressible FP32 bucket must not share PowerSGD state with
    # a compressible BF16 bucket carrying the same numeric index.
    has_compressible_tensor = any(
        _should_powersgd_compress(
            tensor.shape[0],
            tensor.numel() // tensor.shape[0],
            min(
                tensor.shape[0],
                tensor.numel() // tensor.shape[0],
                state.matrix_approximation_rank,
            ),
            state.min_compression_rate,
        )
        for tensor in output_tensors
    )
    if not has_compressible_tensor:
        exact_gradient = input_tensor.detach().clone()
        dist.all_reduce(exact_gradient, group=group)
        exact_gradient.div_(world_size)
        input_tensor.copy_(exact_gradient)
        state.maybe_increase_iter(bucket)
        result = torch.futures.Future()
        result.set_result(input_tensor)
        return result

    # Keep the DDP bucket untouched until all restart work has completed.
    local_corrected = input_tensor.detach().clone()
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            local_corrected.add_(state.error_dict[bucket_index])
        else:
            state.error_dict[bucket_index] = torch.zeros_like(input_tensor)

    # Communicate a private full corrected-gradient buffer.  The synchronous
    # call prevents NCCL from writing into storage that autograd may reuse.
    full_gradient = local_corrected.clone()
    dist.all_reduce(full_gradient, group=group)
    global_gradient = full_gradient.div_(world_size)
    specs = []
    total_p_size = 0
    total_q_size = 0
    offset = 0
    for tensor in output_tensors:
        numel = tensor.numel()
        rows, cols = tensor.shape[0], tensor.numel() // tensor.shape[0]
        matrix = global_gradient[offset : offset + numel].view(rows, cols)
        rank = min(rows, cols, state.matrix_approximation_rank)
        if _should_powersgd_compress(rows, cols, rank, state.min_compression_rate):
            specs.append((offset, numel, matrix, rows, cols, rank))
            total_p_size += rows * rank
            total_q_size += cols * rank
        offset += numel

    p_memory = torch.empty(
        total_p_size, device=input_tensor.device, dtype=input_tensor.dtype
    )
    q_memory = torch.empty(
        total_q_size, device=input_tensor.device, dtype=input_tensor.dtype
    )
    state.p_memory_dict[bucket_index] = p_memory
    state.q_memory_dict[bucket_index] = q_memory

    # Uncompressed tensors use the exact global gradient.  Compressed tensors
    # are replaced below after Q has been averaged across workers.
    output_flat = global_gradient.clone()
    local_approximation = local_corrected.clone()
    factors = []
    p_offset = 0
    q_offset = 0
    for factor_index, (
        flat_offset,
        numel,
        global_matrix,
        rows,
        cols,
        rank,
    ) in enumerate(specs):
        p = p_memory[p_offset : p_offset + rows * rank].view(rows, rank)
        q = q_memory[q_offset : q_offset + cols * rank].view(cols, rank)

        # CUDA SVD does not support bf16/fp16. Compute the subspace in fp32
        # and store the small factors in the communication dtype.
        global_matrix_fp32 = global_matrix.float()
        if state.restart_method == "exact":
            u, _, _ = torch.linalg.svd(global_matrix_fp32, full_matrices=False)
        else:
            seed = (
                0x5EED
                + state.iter * 1_000_003
                + bucket_index * 9_176
                + factor_index
            )
            cuda_devices = [input_tensor.device.index] if input_tensor.is_cuda else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                u = _approximate_top_left_subspace(
                    global_matrix_fp32,
                    rank=rank,
                    niter=2,
                )
        p.copy_(u[:, :rank].to(dtype=input_tensor.dtype))
        del global_matrix_fp32, u

        local_matrix = local_corrected[flat_offset : flat_offset + numel].view(rows, cols)
        torch.mm(local_matrix.transpose(0, 1), p, out=q)
        local_approximation[flat_offset : flat_offset + numel].view(rows, cols).copy_(p @ q.transpose(0, 1))
        factors.append((flat_offset, numel, p, q, rows, cols))
        p_offset += rows * rank
        q_offset += cols * rank

    if q_memory.numel() != 0:
        dist.all_reduce(q_memory, group=group)
        q_memory.div_(world_size)

    if apply_exact:
        if state.use_error_feedback:
            state.error_dict[bucket_index].zero_()
    else:
        for flat_offset, numel, p, q, rows, cols in factors:
            output_flat[flat_offset : flat_offset + numel].view(rows, cols).copy_(
                p @ q.transpose(0, 1)
            )
        if state.use_error_feedback:
            state.error_dict[bucket_index] = local_corrected - local_approximation

    # All CUDA work and collectives are complete before DDP can reuse the
    # bucket.  Copy the final averaged gradient into the bucket exactly once.
    input_tensor.copy_(output_flat)
    if input_tensor.is_cuda:
        torch.cuda.synchronize(input_tensor.device)
    state.maybe_increase_iter(bucket)

    result = torch.futures.Future()
    result.set_result(input_tensor)
    return result



def _bounded_residual_diag_hook(
    state: PowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run normal PowerSGD+ and log sampled residual/second-moment signals."""
    diag_path = os.environ.get("RESIDUAL_DIAG_PATH", "")
    if not diag_path:
        return powerSGD_hook(state, bucket)
    max_numel = max(1, int(os.environ.get("RESIDUAL_DIAG_MAX_NUMEL", "200000")))
    raw = bucket.buffer().detach().view(-1)
    sample_n = min(raw.numel(), max_numel)
    bucket_index = bucket.index()
    old_residual = state.error_dict.get(bucket_index)
    if old_residual is None:
        residual_prev = torch.zeros(sample_n, device=raw.device, dtype=torch.float32)
    else:
        residual_prev = old_residual.detach().view(-1)[:sample_n].float().clone()
    raw_local_sample = raw[:sample_n].float().clone()
    raw_sample = raw_local_sample.clone()
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    dense_future = dist.all_reduce(raw_sample, group=group, async_op=True).get_future()
    main_future = powerSGD_hook(state, bucket)

    def finish(main_done):
        output = _future_value_tensor(main_done)
        dense_sample = _future_value_tensor(dense_future).div_(world_size)
        new_residual = state.error_dict.get(bucket_index)
        if new_residual is None:
            residual_next = torch.zeros_like(residual_prev)
        else:
            residual_next = new_residual.detach().view(-1)[:sample_n].float()
        compressed = output.detach().view(-1)[:sample_n].float()
        delta = compressed - dense_sample
        raw_local = raw_local_sample
        corrected_local = raw_local + residual_prev
        dense_sq = float(dense_sample.square().sum().item())
        cross = 2.0 * dense_sample * delta
        quad = delta.square()
        record = {
            "step": int(state.iter + 1),
            "bucket": int(bucket_index),
            "sample_numel": int(sample_n),
            "raw_local_norm": float(raw_local.norm().item()),
            "corrected_local_norm": float(corrected_local.norm().item()),
            "residual_prev_norm": float(residual_prev.norm().item()),
            "residual_next_norm": float(residual_next.norm().item()),
            "residual_delta_norm": float((residual_prev - residual_next).norm().item()),
            "dense_grad_norm": float(dense_sample.norm().item()),
            "compressed_grad_norm": float(compressed.norm().item()),
            "compression_error_norm": float(delta.norm().item()),
            "residual_to_raw_ratio": float(residual_prev.norm().item() / (raw_local.norm().item() + 1e-12)),
            "error_to_dense_ratio": float(delta.norm().item() / (dense_sample.norm().item() + 1e-12)),
            "cross_sum": float(cross.sum().item()),
            "quad_sum": float(quad.sum().item()),
            "cross_abs_sum": float(cross.abs().sum().item()),
            "quad_abs_sum": float(quad.abs().sum().item()),
            "sq_baseline_sum": dense_sq,
            "sq_error_signed_sum": float((cross + quad).sum().item()),
            "sq_error_abs_sum": float((cross + quad).abs().sum().item()),
            "residual_identity_gap_norm": float((delta - (residual_prev - residual_next)).norm().item()),
        }
        if is_rank_zero():
            try:
                with open(diag_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            except OSError:
                logger.warning("Unable to write residual diagnostic", exc_info=True)
        return output

    return main_future.then(finish)



def _bounded_shadow_v_diag_hook(
    state: PowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run native PowerSGD+ and maintain bounded exact-v/EF-v shadow EMAs."""
    diag_path = os.environ.get("SHADOW_V_DIAG_PATH", "")
    if not diag_path:
        return powerSGD_hook(state, bucket)
    max_numel = max(1, int(os.environ.get("SHADOW_V_MAX_NUMEL", "4096")))
    max_buckets = max(1, int(os.environ.get("SHADOW_V_MAX_BUCKETS", "8")))
    bucket_index = int(bucket.index())
    # All ranks make the same bucket decision, so collective ordering stays equal.
    if bucket_index >= max_buckets:
        return powerSGD_hook(state, bucket)
    raw = bucket.buffer().detach().view(-1)
    sample_n = min(raw.numel(), max_numel)
    raw_local = raw[:sample_n].float().clone()
    old_residual = state.error_dict.get(bucket_index)
    if old_residual is None:
        residual_prev = torch.zeros(sample_n, device=raw.device, dtype=torch.float32)
    else:
        residual_prev = old_residual.detach().view(-1)[:sample_n].float().clone()
    # Native PowerSGD+ adds this same residual before compression.  Use the
    # corrected local gradient for the exact dense reference so v_exact and
    # v_ef differ only by compression on this trajectory.
    corrected_local = raw_local + residual_prev
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    dense_future = dist.all_reduce(corrected_local, group=group, async_op=True).get_future()
    main_future = powerSGD_hook(state, bucket)
    beta2 = float(os.environ.get("SHADOW_V_BETA2", "0.95"))
    beta2 = min(max(beta2, 0.0), 0.999999)

    def finish(main_done):
        output = _future_value_tensor(main_done)
        dense = _future_value_tensor(dense_future).div_(world_size)
        compressed = output.detach().view(-1)[:sample_n].float()
        delta = compressed - dense
        cross = 2.0 * dense * delta
        quad = delta.square()
        shadows = getattr(state, "_shadow_v_diag", None)
        if shadows is None:
            shadows = state._shadow_v_diag = {}
        entry = shadows.get(bucket_index)
        if entry is None:
            entry = {
                "v_exact": torch.zeros_like(dense),
                "v_ef": torch.zeros_like(dense),
                "v_cross": torch.zeros_like(dense),
            }
            shadows[bucket_index] = entry
        one_minus = 1.0 - beta2
        entry["v_exact"].mul_(beta2).addcmul_(dense, dense, value=one_minus)
        entry["v_ef"].mul_(beta2).addcmul_(compressed, compressed, value=one_minus)
        entry["v_cross"].mul_(beta2).add_((dense.square() + cross) * one_minus)
        exact = entry["v_exact"]
        ef = entry["v_ef"]
        cross_v = entry["v_cross"]
        exact_norm = exact.norm()
        record = {
            "SHADOW_V_DIAG_INSTALLED": True,
            "step": int(state.iter + 1),
            "bucket": bucket_index,
            "sample_numel": int(sample_n),
            "shadow_v_init_zero": True,
            "dense_grad_norm": float(dense.norm().item()),
            "compressed_grad_norm": float(compressed.norm().item()),
            "compression_error_norm": float(delta.norm().item()),
            "raw_local_norm": float(raw_local.norm().item()),
            "residual_prev_norm": float(residual_prev.norm().item()),
            "cross_over_dense_sq": float(cross.sum().item() / (dense.square().sum().item() + 1e-12)),
            "quad_over_dense_sq": float(quad.sum().item() / (dense.square().sum().item() + 1e-12)),
            "signed_sq_error_over_dense_sq": float((cross + quad).sum().item() / (dense.square().sum().item() + 1e-12)),
            "v_exact_norm": float(exact_norm.item()),
            "v_ef_norm": float(ef.norm().item()),
            "v_diff_norm": float((ef - exact).norm().item()),
            "v_diff_rel": float((ef - exact).norm().item() / (exact_norm.item() + 1e-12)),
            "v_ef_over_exact_norm": float(ef.norm().item() / (exact_norm.item() + 1e-12)),
            "v_cross_norm": float(cross_v.norm().item()),
            "v_cross_diff_rel": float((cross_v - exact).norm().item() / (exact_norm.item() + 1e-12)),
        }
        if is_rank_zero():
            try:
                with open(diag_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            except OSError:
                logger.warning("Unable to write shadow-v diagnostic", exc_info=True)
        return output

    return main_future.then(finish)


def _exact_allreduce_without_powersgd_state(
    state: PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """All-reduce a non-compressed dtype bucket without touching P/Q/EF state.

    The exact bucket can still be DDP's last bucket, so it must participate in
    the state's iteration bookkeeping.  Otherwise a trailing FP32 bucket keeps
    ``state.iter`` at zero and the BF16 bucket remains in dense warmup forever.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    state.maybe_increase_iter(bucket)
    future = dist.all_reduce(input_tensor, group=group, async_op=True).get_future()

    def finish(fut):
        return fut.value()[0].div_(world_size)

    return future.then(finish)


def powerSGD_plus_hook(
    state: PowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD with periodic top-r safeguards from PowerSGD+."""
    if bucket.buffer().dtype == torch.float32:
        return _exact_allreduce_without_powersgd_state(state, bucket)
    if state.iter < state.start_powerSGD_iter:
        _record_powersgd_compression_signal(state, bucket, compressed=False)
        return powerSGD_hook(state, bucket)

    compressed_iter = state.iter - state.start_powerSGD_iter
    if compressed_iter % state.restart_period != 0:
        _record_powersgd_compression_signal(state, bucket, compressed=True)
        if os.environ.get("SHADOW_V_DIAG_PATH"):
            return _bounded_shadow_v_diag_hook(state, bucket)
        if os.environ.get("RESIDUAL_DIAG_PATH"):
            return _bounded_residual_diag_hook(state, bucket)
        return powerSGD_hook(state, bucket)
    _record_powersgd_compression_signal(state, bucket, compressed=True)
    return _powersgd_plus_svd_restart(state, bucket)


def powerSGD_plus_error_averaging_hook(
    state: PowerSGDPlusErrorAveragingState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD+ with periodic averaging of local EF residuals.

    The normal PowerSGD+ path is unchanged except on every configured
    ``error_averaging_period``-th compressed iteration.  At those iterations,
    the residual buffer produced by PowerSGD's error feedback is all-reduced
    in-place and divided by world size.  This targets worker-to-worker
    residual drift while retaining rank-1 compression and the existing
    periodic global-subspace restart schedule.
    """
    iter_at_entry = state.iter
    if iter_at_entry < state.start_powerSGD_iter:
        _record_powersgd_compression_signal(state, bucket, compressed=False)
        return powerSGD_hook(state, bucket)

    compressed_iter = iter_at_entry - state.start_powerSGD_iter
    if compressed_iter % state.restart_period == 0:
        _record_powersgd_compression_signal(state, bucket, compressed=True)
        return _powersgd_plus_svd_restart(state, bucket)

    _record_powersgd_compression_signal(state, bucket, compressed=True)
    future = powerSGD_hook(state, bucket)
    if (compressed_iter + 1) % state.error_averaging_period != 0:
        return future

    group = state.process_group if state.process_group is not None else dist.group.WORLD
    bucket_index = bucket.index()

    def average_residual(fut):
        value = fut.value()
        result = value[0] if isinstance(value, (tuple, list)) else value
        residual = state.error_dict.get(bucket_index)
        if residual is not None:
            dist.all_reduce(residual, group=group)
            residual.div_(dist.get_world_size(group))
        return result

    return future.then(average_residual)


def loss_triggered_powerSGD_plus_hook(
    state: LossTriggeredPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD+ with restart decisions based on observed loss drift."""
    if bucket.index() == 0:
        state.prepare_loss_trigger()
    if state.iter < state.start_powerSGD_iter:
        return powerSGD_hook(state, bucket)
    if state._restart_this_iter:
        return _powersgd_plus_svd_restart(state, bucket)
    return powerSGD_hook(state, bucket)


_SUPPORTED_COMM_HOOKS["powersgd_plus_hook"] = powerSGD_plus_hook
_SUPPORTED_COMM_HOOKS["powersgd_plus_dense_v_hook"] = dense_v_powersgd_plus_hook
_SUPPORTED_COMM_HOOKS["powersgd_plus_error_averaging_hook"] = (
    powerSGD_plus_error_averaging_hook
)
_SUPPORTED_COMM_HOOKS["loss_triggered_powersgd_plus_hook"] = (
    loss_triggered_powerSGD_plus_hook
)


def windowed_powerSGD_plus_hook(
    state: WindowedPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Use PowerSGD+ only in fixed global-step windows.

    Exact iterations immediately after a window flush each bucket's local
    residual through the dense all-reduce before resetting its compressor
    state. All ranks share the same immutable ranges, so collective shapes
    cannot diverge.
    """
    state.prepare_iteration()
    if state._active:
        if state.iter < state.start_powerSGD_iter:
            return powerSGD_hook(state, bucket)
        if state.compressed_iter_in_window % state.restart_period == 0:
            return _powersgd_plus_svd_restart(state, bucket)
        return powerSGD_hook(state, bucket)

    return _windowed_exact_allreduce(state, bucket)


def _windowed_exact_allreduce(
    state: WindowedPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run one exact step and flush any residual left by a preceding window."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])

    state.maybe_increase_iter(bucket)
    future = dist.all_reduce(
        input_tensor, group=group, async_op=True
    ).get_future()

    def finish_exact(fut):
        reduced = fut.value()[0].div_(world_size)
        state.error_dict.pop(bucket_index, None)
        state.p_memory_dict.pop(bucket_index, None)
        state.q_memory_dict.pop(bucket_index, None)
        return reduced

    return future.then(finish_exact)


_SUPPORTED_COMM_HOOKS["windowed_powersgd_plus_hook"] = (
    windowed_powerSGD_plus_hook
)


def _scheduled_powerSGD_plus_hook(
    state: ScheduledPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    if bucket.buffer().dtype == torch.float32:
        return _exact_allreduce_without_powersgd_state(state, bucket)
    state.prepare_policy_iteration()
    if not state._active:
        return _windowed_exact_allreduce(state, bucket)
    if state._restart_this_iter:
        return _powersgd_plus_svd_restart(
            state,
            bucket,
            apply_exact=state.policy == "exact_flush",
        )
    return powerSGD_hook(state, bucket)


def exact_flush_powerSGD_plus_hook(
    state: ScheduledPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD+ with a dense residual-clearing update at each restart."""
    return _scheduled_powerSGD_plus_hook(state, bucket)


def lr_rank_powerSGD_plus_hook(
    state: ScheduledPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD+ using rank 2 in the configured high-LR prefix."""
    return _scheduled_powerSGD_plus_hook(state, bucket)


def dynamic_period_powerSGD_plus_hook(
    state: ScheduledPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """PowerSGD+ with an absolute-step restart-period schedule."""
    return _scheduled_powerSGD_plus_hook(state, bucket)


_SUPPORTED_COMM_HOOKS["exact_flush_powersgd_plus_hook"] = (
    exact_flush_powerSGD_plus_hook
)
_SUPPORTED_COMM_HOOKS["lr_rank_powersgd_plus_hook"] = lr_rank_powerSGD_plus_hook
_SUPPORTED_COMM_HOOKS["dynamic_period_powersgd_plus_hook"] = (
    dynamic_period_powerSGD_plus_hook
)


def dynamic_powerSGD_hook(
    state: DynamicPowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Apply PowerSGD with an Accordion-style two-level dynamic rank."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD

    # A decision made from the previous full gradient becomes active only at
    # the next iteration boundary, so every bucket in one iteration uses the
    # same rank and collective sizes.
    if state.pending_rank is not None and state.last_rank_change_iter != state.iter:
        next_rank = state.pending_rank
        state.pending_rank = None
        state.last_rank_change_iter = state.iter
        if next_rank != state.matrix_approximation_rank:
            state.matrix_approximation_rank = next_rank
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
            if is_rank_zero():
                logger.info(
                    "Dynamic PowerSGD changed rank to %d at local iteration %d",
                    next_rank,
                    state.iter,
                )

    # The summed local squared norm is inexpensive to accumulate.  Reducing
    # it only at adaptation points keeps rank decisions identical on all ranks
    # without adding a scalar collective to every iteration.
    # Ask the reduction kernel for an FP32 scalar directly; materializing a
    # full FP32 copy of the bucket here can double peak memory on large
    # models.
    local_norm_sq = torch.linalg.vector_norm(
        bucket.buffer().detach(), dtype=torch.float32
    ).square()
    if state.norm_sq_accumulator is None:
        state.norm_sq_accumulator = local_norm_sq
    else:
        state.norm_sq_accumulator.add_(local_norm_sq)

    if bucket.is_last():
        completed_iteration = state.iter + 1
        if (
            completed_iteration >= state.start_powerSGD_iter
            and completed_iteration % state.adapt_interval == 0
        ):
            dist.all_reduce(state.norm_sq_accumulator, group=group)
            global_norm = math.sqrt(max(state.norm_sq_accumulator.item(), 0.0))
            if state.previous_global_grad_norm is not None:
                denominator = max(state.previous_global_grad_norm, 1e-12)
                relative_change = abs(
                    global_norm - state.previous_global_grad_norm
                ) / denominator
                state.pending_rank = (
                    state.high_rank
                    if relative_change >= state.relative_change_threshold
                    else state.low_rank
                )
                if is_rank_zero():
                    logger.info(
                        "Dynamic PowerSGD norm=%.6e relative_change=%.6f "
                        "pending_rank=%d",
                        global_norm,
                        relative_change,
                        state.pending_rank,
                    )
            state.previous_global_grad_norm = global_norm
        state.norm_sq_accumulator = None

    return powerSGD_hook(state, bucket)


def _layerwise_parameter_names(
    state: LayerwisePowerSGDState,
    bucket: dist.GradBucket,
) -> list[str]:
    """Resolve stable parameter names for one DDP bucket when available."""
    gradients = bucket.gradients()
    parameters = bucket.parameters() if hasattr(bucket, "parameters") else []
    if len(parameters) != len(gradients):
        return [
            f"bucket{bucket.index()}.tensor{index}"
            for index in range(len(gradients))
        ]
    return [
        state.parameter_names.get(
            id(parameter), f"bucket{bucket.index()}.tensor{index}"
        )
        for index, parameter in enumerate(parameters)
    ]


def _select_layerwise_rank(
    state: LayerwisePowerSGDState,
    matrix: torch.Tensor,
    parameter_name: str,
) -> tuple[int, float, torch.Tensor | None]:
    """Choose a rank from an approximate singular spectrum.

    Rank zero means dense communication.  The returned Q factor initializes
    the next compressed iteration's warm start.
    """
    rows, cols = matrix.shape
    if (
        state.selective_min_tensor_bytes > 0
        and matrix.numel() * matrix.element_size()
        <= state.selective_min_tensor_bytes
    ):
        return 0, 0.0, None
    if (
        state.selective_gemm_regex is not None
        and not state.selective_gemm_regex.search(parameter_name)
    ):
        return 0, 0.0, None
    if (
        state.uncompressed_regex is not None
        and state.uncompressed_regex.search(parameter_name)
    ):
        return 0, 0.0, None

    feasible_ranks = [
        rank
        for rank in state.candidate_ranks
        if rank <= min(rows, cols)
        and _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        )
    ]
    if not feasible_ranks:
        return 0, 0.0, None

    matrix_fp32 = matrix.float()
    total_energy = matrix_fp32.square().sum()
    max_rank = feasible_ranks[-1]
    if total_energy.item() == 0:
        # A zero gradient is trivially rank 0, but says nothing about the next
        # batch.  Treating it as a perfect rank-1 observation creates a stale
        # compression plan for an intermittently active tensor.  The Pi0.5
        # profile showed exactly this false positive for seven very large
        # tensors at every refresh, followed by an immediate loss jump on the
        # first compressed step.
        return 0, 0.0, None

    basis = _approximate_top_left_subspace(
        matrix_fp32,
        rank=max_rank,
        niter=state.power_iterations,
    )
    projected = basis.transpose(0, 1) @ matrix_fp32
    covariance = projected @ projected.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min_(0)
    basis = basis @ eigenvectors[:, order]
    cumulative_energy = torch.cumsum(eigenvalues, dim=0)

    selected_rank = 0
    selected_error = 0.0
    for rank in feasible_ranks:
        residual_energy = torch.clamp(
            total_energy - cumulative_energy[rank - 1], min=0
        )
        relative_error = torch.sqrt(residual_energy / total_energy).item()
        if relative_error <= state.relative_error_threshold:
            selected_rank = rank
            selected_error = relative_error
            break

    if selected_rank == 0:
        return 0, 0.0, None

    p = basis[:, :selected_rank]
    q = matrix_fp32.transpose(0, 1) @ p
    return (
        selected_rank,
        selected_error,
        q.to(device=matrix.device, dtype=matrix.dtype),
    )


def _install_layerwise_factor_plan(
    state: LayerwisePowerSGDState,
    bucket_index: int,
    gradients: list[torch.Tensor],
    rank_plan: list[int],
    initial_qs: list[torch.Tensor | None],
) -> None:
    """Allocate grouped contiguous P/Q memories for a variable-rank plan."""
    grouped_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (tensor, rank) in enumerate(zip(gradients, rank_plan)):
        if rank == 0:
            continue
        matrix = tensor.view(tensor.shape[0], -1)
        grouped_indices[(matrix.shape[0], matrix.shape[1], rank)].append(index)

    total_p_size = sum(
        len(indices) * rows * rank
        for (rows, _, rank), indices in grouped_indices.items()
    )
    total_q_size = sum(
        len(indices) * cols * rank
        for (_, cols, rank), indices in grouped_indices.items()
    )
    device = gradients[0].device
    dtype = gradients[0].dtype
    p_memory = torch.empty(total_p_size, device=device, dtype=dtype)
    q_memory = torch.empty(total_q_size, device=device, dtype=dtype)
    state.p_memory_dict[bucket_index] = p_memory
    state.q_memory_dict[bucket_index] = q_memory

    group_plan = []
    p_offset = 0
    q_offset = 0
    for (rows, cols, rank), indices in grouped_indices.items():
        batch_size = len(indices)
        p_numel = batch_size * rows * rank
        q_numel = batch_size * cols * rank
        q_view = q_memory[q_offset : q_offset + q_numel].view(
            batch_size, cols, rank
        )
        for position, tensor_index in enumerate(indices):
            initial_q = initial_qs[tensor_index]
            if initial_q is None:
                raise RuntimeError(
                    "Layerwise PowerSGD compressed tensor is missing initial Q"
                )
            q_view[position].copy_(initial_q)
        group_plan.append(
            (indices, rows, cols, rank, p_offset, q_offset)
        )
        p_offset += p_numel
        q_offset += q_numel

    state.group_plan_dict[bucket_index] = group_plan


def _layerwise_powersgd_refresh(
    state: LayerwisePowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Apply a dense correction and refresh the per-tensor rank plan."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()

    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])
    else:
        state.error_dict[bucket_index] = torch.zeros_like(input_tensor)

    full_gradient_fut = dist.all_reduce(
        input_tensor, group=group, async_op=True
    ).get_future()

    def select_plan_and_finish(fut):
        fut.value()[0].div_(world_size)
        gradients = list(bucket.gradients())
        names = _layerwise_parameter_names(state, bucket)
        rank_plan: list[int] = []
        rank_errors: list[float] = []
        initial_qs: list[torch.Tensor | None] = []

        before_numel = 0
        after_numel = 0
        critical_mask = getattr(state, "critical_tensor_mask_dict", {}).get(
            bucket_index, [False] * len(gradients)
        )
        window_energy = getattr(state, "last_window_norm_energy_dict", {}).get(
            bucket_index
        )
        energy_shares = None
        if window_energy is not None and window_energy.numel() == len(gradients):
            energy_shares = window_energy / window_energy.sum().clamp_min(1e-12)
        if getattr(state, "norm_selective", False):
            expiries = getattr(state, "critical_until_tensor_iter_dict", {}).get(
                bucket_index, []
            )
            if len(expiries) == len(gradients):
                # Keep each tensor dense for the configured protection window
                # after a norm-change event, instead of only at one refresh.
                critical_mask = [state.iter <= expiry for expiry in expiries]
        for tensor_index, (tensor, name) in enumerate(zip(gradients, names)):
            before_numel += tensor.numel()
            if tensor.ndim <= 1:
                rank, relative_error, initial_q = 0, 0.0, None
            elif getattr(state, "norm_selective", False) and (
                tensor_index < len(critical_mask) and critical_mask[tensor_index]
            ):
                # Accordion marks only tensors whose windowed norm changed
                # sharply as dense; all other eligible tensors use rank 1.
                rank, relative_error, initial_q = 0, 0.0, None
            elif getattr(state, "norm_selective", False) and (
                energy_shares is None
                or tensor_index >= energy_shares.numel()
                or energy_shares[tensor_index].item() > 0.00001
            ):
                # Until a complete norm window exists, or for tensors that
                # carry more than 0.001% of bucket energy, stay dense.  Only
                # the very low-energy tail is eligible for rank-1
                # compression; this conservative gate is intentional because
                # the norm signal alone did not reliably predict loss drift in
                # the earlier 0.1%-energy run.
                rank, relative_error, initial_q = 0, 0.0, None
            else:
                matrix = tensor.view(tensor.shape[0], -1)
                if getattr(state, "norm_selective", False):
                    rows, cols = matrix.shape
                    eligible = (
                        state.uncompressed_regex is None
                        or not state.uncompressed_regex.search(name)
                    ) and _should_powersgd_compress(
                        rows, cols, 1, state.min_compression_rate
                    )
                    if (
                        not eligible
                        or torch.linalg.vector_norm(
                            matrix.detach(), dtype=torch.float32
                        ).square().item()
                        == 0
                    ):
                        rank, relative_error, initial_q = 0, 0.0, None
                    else:
                        matrix_fp32 = matrix.detach().float()
                        basis = _approximate_top_left_subspace(
                            matrix_fp32, rank=1, niter=state.power_iterations
                        )
                        initial_q = (
                            matrix_fp32.transpose(0, 1) @ basis
                        ).to(device=matrix.device, dtype=matrix.dtype)
                        rank, relative_error = 1, 0.0
                else:
                    rank, relative_error, initial_q = _select_layerwise_rank(
                        state, matrix, name
                    )
            rank_plan.append(rank)
            rank_errors.append(relative_error)
            initial_qs.append(initial_q)
            if rank == 0:
                after_numel += tensor.numel()
            else:
                matrix = tensor.view(tensor.shape[0], -1)
                after_numel += (matrix.shape[0] + matrix.shape[1]) * rank

        state.rank_plan_dict[bucket_index] = rank_plan
        state.rank_error_dict[bucket_index] = rank_errors
        _install_layerwise_factor_plan(
            state,
            bucket_index,
            gradients,
            rank_plan,
            initial_qs,
        )
        state.error_dict[bucket_index].zero_()

        if is_rank_zero():
            counts = Counter("dense" if rank == 0 else rank for rank in rank_plan)
            estimated_rate = before_numel / max(after_numel, 1)
            logger.info(
                "Layerwise PowerSGD refresh iter=%d bucket=%d ranks=%s "
                "estimated_compression_rate=%.3fx error_threshold=%.4f",
                state.iter,
                bucket_index,
                dict(counts),
                estimated_rate,
                state.relative_error_threshold,
            )
            if state.log_tensor_stats:
                for name, tensor, rank, relative_error in zip(
                    names, gradients, rank_plan, rank_errors
                ):
                    gradient_norm = None
                    if tensor.ndim > 1:
                        # Scalar-only norm telemetry; do not materialize a
                        # full FP32 tensor copy in the logging path.
                        gradient_norm = torch.linalg.vector_norm(
                            tensor.detach()
                        ).item()
                    logger.info(
                        "Layerwise PowerSGD tensor iter=%d bucket=%d name=%s "
                        "shape=%s rank=%s estimated_relative_error=%.6f "
                        "gradient_norm=%s",
                        state.iter,
                        bucket_index,
                        name,
                        tuple(tensor.shape),
                        "dense" if rank == 0 else rank,
                        relative_error,
                        (
                            "not_evaluated"
                            if gradient_norm is None
                            else f"{gradient_norm:.8e}"
                        ),
                    )

        if input_tensor.is_cuda:
            torch.cuda.empty_cache()
        state.maybe_increase_iter(bucket)
        return input_tensor

    return full_gradient_fut.then(select_plan_and_finish)


def layerwise_powerSGD_hook(
    state: LayerwisePowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Run error-budgeted per-tensor PowerSGD between dense refreshes."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()

    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(lambda fut: fut.value()[0].div_(world_size))

    compressed_iter = state.iter - state.start_powerSGD_iter
    if (
        bucket_index not in state.rank_plan_dict
        or compressed_iter % state.refresh_period == 0
    ):
        return _layerwise_powersgd_refresh(state, bucket)

    gradients = list(bucket.gradients())
    rank_plan = state.rank_plan_dict[bucket_index]
    if len(gradients) != len(rank_plan):
        # DDP may rebuild buckets before the graph becomes static.  A dense
        # refresh is safe and reconstructs a matching plan.
        return _layerwise_powersgd_refresh(state, bucket)

    # Reuse the error-feedback buffer as the saved corrected gradient.  A
    # separate ``input_tensor.clone()`` here duplicates the whole DDP bucket
    # (tens of GiB for Pi0.5) and was enough to push an 80-GiB A800 over its
    # limit once the uncompressed pack and P/Q workspaces were allocated.
    # ``error_buffer`` initially contains E_{t-1}; after adding the raw local
    # gradient it contains G_t + E_{t-1}.  The communication path can then
    # overwrite ``input_tensor`` in place, and the final subtraction leaves
    # E_t in the same buffer without allocating another bucket-sized tensor.
    error_buffer = state.error_dict.get(bucket_index)
    if error_buffer is None:
        error_buffer = torch.zeros_like(input_tensor)
        state.error_dict[bucket_index] = error_buffer
    error_buffer.add_(input_tensor)
    input_tensor.copy_(error_buffer)

    if not state.group_plan_dict[bucket_index]:
        dense_future = dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future()

        def finish_dense_bucket(fut):
            fut.value()[0].div_(world_size)
            state.error_dict[bucket_index].zero_()
            state.total_numel_before_compression += input_tensor.numel()
            state.total_numel_after_compression += input_tensor.numel()
            state.maybe_increase_iter(bucket)
            return input_tensor

        return dense_future.then(finish_dense_bucket)

    uncompressed_tensors = [
        tensor for tensor, rank in zip(gradients, rank_plan) if rank == 0
    ]
    uncompressed_memory = (
        torch.cat([tensor.reshape(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )

    p_memory = state.p_memory_dict[bucket_index]
    q_memory = state.q_memory_dict[bucket_index]
    group_batches = []
    for indices, rows, cols, rank, p_offset, q_offset in state.group_plan_dict[
        bucket_index
    ]:
        batch_size = len(indices)
        matrices = [
            gradients[index].view(rows, cols) for index in indices
        ]
        matrix_batch = (
            matrices[0].unsqueeze(0)
            if batch_size == 1
            else torch.stack(matrices)
        )
        p_numel = batch_size * rows * rank
        q_numel = batch_size * cols * rank
        p = p_memory[p_offset : p_offset + p_numel].view(
            batch_size, rows, rank
        )
        q = q_memory[q_offset : q_offset + q_numel].view(
            batch_size, cols, rank
        )
        _orthogonalize(q, state.orthogonalization_epsilon)
        torch.bmm(matrix_batch, q, out=p)
        group_batches.append((indices, matrix_batch, p, q))

    uncompressed_future = dist.all_reduce(
        uncompressed_memory, group=group, async_op=True
    ).get_future()
    p_future = dist.all_reduce(
        p_memory, group=group, async_op=True
    ).get_future()

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        for _, matrix_batch, p, q in group_batches:
            _orthogonalize(p, state.orthogonalization_epsilon)
            torch.bmm(matrix_batch.transpose(1, 2), p, out=q)
        return (
            dist.all_reduce(q_memory, group=group, async_op=True)
            .get_future()
            .wait()[0]
        )

    def reconstruct_and_finish(fut):
        fut.value().div_(world_size)
        reduced_uncompressed = uncompressed_future.wait()[0].div_(world_size)
        offset = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                reduced_uncompressed[offset : offset + tensor.numel()].view_as(
                    tensor
                )
            )
            offset += tensor.numel()

        for indices, matrix_batch, p, q in group_batches:
            reconstruction = torch.bmm(p, q.transpose(1, 2))
            for position, tensor_index in enumerate(indices):
                # ``gradients[tensor_index]`` may be a convolutional or
                # otherwise higher-rank tensor.  The factorization operates
                # on its flattened ``(rows, cols)`` view, so restore that
                # view before copying back; copying the 2-D reconstruction
                # directly causes shape mismatches such as ``14`` vs ``588``.
                gradients[tensor_index].copy_(
                    reconstruction[position].view_as(gradients[tensor_index])
                )

        if input_tensor.is_cuda:
            torch.cuda.synchronize(input_tensor.device)
        error_buffer.sub_(input_tensor)
        state.total_numel_before_compression += input_tensor.numel()
        state.total_numel_after_compression += (
            uncompressed_memory.numel() + p_memory.numel() + q_memory.numel()
        )
        state.maybe_increase_iter(bucket)
        return input_tensor

    return p_future.then(compute_qs).then(reconstruct_and_finish)


_SUPPORTED_COMM_HOOKS["layerwise_powersgd_hook"] = layerwise_powerSGD_hook


def _select_oracle_projection(
    state: OracleErrorBudgetPowerSGDState,
    matrix: torch.Tensor,
    parameter_name: str,
    *,
    bucket_index: int,
    tensor_index: int,
) -> tuple[int, float, torch.Tensor | None]:
    """Return the smallest budget-satisfying rank and its left basis.

    Rank zero denotes dense fallback.  The globally averaged input matrix is
    identical on every worker; a deterministic seed makes the randomized range
    finder and therefore the collective-free decision identical as well.
    """
    rows, cols = matrix.shape
    if (
        state.uncompressed_regex is not None
        and state.uncompressed_regex.search(parameter_name)
    ):
        return 0, 0.0, None

    feasible_ranks = [
        rank
        for rank in state.candidate_ranks
        if rank <= min(rows, cols)
        and _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        )
    ]
    if not feasible_ranks:
        return 0, 0.0, None

    matrix_fp32 = matrix.float()
    total_energy = matrix_fp32.square().sum()
    if total_energy.item() == 0:
        # A zero snapshot is not evidence that the next gradient is
        # low-rank.  Returning an identity basis here would install a stale
        # compressed plan for intermittently active tensors; keep this tensor
        # dense until a non-zero gradient supplies a real subspace.
        return 0, 0.0, None

    seed = (
        state.random_seed
        + state.iter * 1_000_003
        + bucket_index * 9_176
        + tensor_index * 131
    )
    cuda_devices = [matrix.device.index] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        basis = _approximate_top_left_subspace(
            matrix_fp32,
            rank=feasible_ranks[-1],
            niter=state.power_iterations,
        )

    projected = basis.transpose(0, 1) @ matrix_fp32
    covariance = projected @ projected.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min_(0)
    basis = basis @ eigenvectors[:, order]
    cumulative_energy = torch.cumsum(eigenvalues, dim=0)

    for rank in feasible_ranks:
        residual_energy = torch.clamp(
            total_energy - cumulative_energy[rank - 1], min=0
        )
        relative_error = torch.sqrt(residual_energy / total_energy).item()
        if relative_error <= state.relative_error_threshold:
            return rank, relative_error, basis[:, :rank]
    return 0, 0.0, None


def oracle_error_budget_powerSGD_hook(
    state: OracleErrorBudgetPowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Validate per-step information-budgeted low-rank updates.

    The dense all-reduce is deliberately an oracle and is not counted as the
    simulated payload.  This experiment answers whether accurate, fresh
    per-tensor rank selection reduces loss drift before a communication-saving
    estimator is engineered for Pro6K.
    """
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()

    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(
            input_tensor, group=group, async_op=True
        ).get_future().then(lambda fut: fut.value()[0].div_(world_size))

    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])
    else:
        state.error_dict[bucket_index] = torch.zeros_like(input_tensor)
    local_corrected = input_tensor.detach().clone()

    full_gradient_fut = dist.all_reduce(
        input_tensor, group=group, async_op=True
    ).get_future()

    def project_and_finish(fut):
        fut.value()[0].div_(world_size)
        gradients = list(bucket.gradients())
        names = _layerwise_parameter_names(state, bucket)
        error_flat = state.error_dict[bucket_index]
        ranks: list[int] = []
        errors: list[float] = []
        before_numel = 0
        simulated_after_numel = 0
        offset = 0

        for tensor_index, (tensor, name) in enumerate(zip(gradients, names)):
            numel = tensor.numel()
            before_numel += numel
            local_tensor = local_corrected[offset : offset + numel].view_as(tensor)
            error_tensor = error_flat[offset : offset + numel].view_as(tensor)
            offset += numel

            if tensor.ndim <= 1:
                rank, relative_error, basis = 0, 0.0, None
            else:
                rank, relative_error, basis = _select_oracle_projection(
                    state,
                    tensor.view(tensor.shape[0], -1),
                    name,
                    bucket_index=bucket_index,
                    tensor_index=tensor_index,
                )
            ranks.append(rank)
            errors.append(relative_error)

            rank_key = "dense" if rank == 0 or basis is None else f"rank{rank}"
            rank_stats = state.profile_by_rank.setdefault(
                rank_key, {"count": 0, "numel": 0, "payload_numel": 0}
            )
            rank_stats["count"] += 1
            rank_stats["numel"] += numel
            size_bin = _oracle_size_bin(numel)
            size_stats = state.profile_by_size_bin.setdefault(
                size_bin,
                {
                    "count": 0,
                    "numel": 0,
                    "dense_numel": 0,
                    "compressed_numel": 0,
                },
            )
            size_stats["count"] += 1
            size_stats["numel"] += numel

            if rank == 0 or basis is None:
                error_tensor.zero_()
                simulated_after_numel += numel
                rank_stats["payload_numel"] += numel
                size_stats["dense_numel"] += numel
                continue

            global_matrix = tensor.view(tensor.shape[0], -1)
            global_approximation = basis @ (
                basis.transpose(0, 1) @ global_matrix.float()
            )
            global_matrix.copy_(global_approximation.to(dtype=tensor.dtype))

            local_matrix = local_tensor.view(tensor.shape[0], -1)
            basis_local = basis.to(dtype=local_matrix.dtype)
            local_approximation = basis_local @ (
                basis_local.transpose(0, 1) @ local_matrix
            )
            error_tensor.view_as(local_matrix).copy_(
                local_matrix - local_approximation
            )
            lowrank_payload_numel = (
                global_matrix.shape[0] + global_matrix.shape[1]
            ) * rank
            simulated_after_numel += lowrank_payload_numel
            rank_stats["payload_numel"] += lowrank_payload_numel
            size_stats["compressed_numel"] += numel

        state.total_numel_before_compression += before_numel
        state.total_numel_after_compression += simulated_after_numel
        if is_rank_zero():
            counts = Counter("dense" if rank == 0 else rank for rank in ranks)
            compressed_errors = [
                error for rank, error in zip(ranks, errors) if rank != 0
            ]
            logger.info(
                "Oracle ErrorBudget PowerSGD iter=%d bucket=%d ranks=%s "
                "mean_selected_relative_error=%.6f simulated_rate=%.3fx",
                state.iter,
                bucket_index,
                dict(counts),
                (
                    sum(compressed_errors) / len(compressed_errors)
                    if compressed_errors
                    else 0.0
                ),
                before_numel / max(simulated_after_numel, 1),
            )
            logger.info(
                "Oracle TensorProfile iter=%d rank_stats=%s size_bins=%s "
                "total_numel=%d total_payload_numel=%d",
                state.iter,
                state.profile_by_rank,
                state.profile_by_size_bin,
                sum(item["numel"] for item in state.profile_by_rank.values()),
                sum(
                    item["payload_numel"]
                    for item in state.profile_by_rank.values()
                ),
            )
            if state.log_tensor_stats:
                for name, tensor, rank, relative_error in zip(
                    names, gradients, ranks, errors
                ):
                    logger.info(
                        "Oracle ErrorBudget tensor iter=%d bucket=%d name=%s "
                        "shape=%s rank=%s relative_error=%.6f",
                        state.iter,
                        bucket_index,
                        name,
                        tuple(tensor.shape),
                        "dense" if rank == 0 else rank,
                        relative_error,
                    )
        if input_tensor.is_cuda:
            torch.cuda.synchronize(input_tensor.device)
        state.maybe_increase_iter(bucket)
        return input_tensor

    return full_gradient_fut.then(project_and_finish)


def _accordion_observe_gradient_window(
    state: AccordionPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> None:
    """Accumulate per-tensor energy and update the next critical window."""
    bucket_index = bucket.index()
    gradients = list(bucket.gradients())
    # Reduce each tensor to FP32 scalars without exposing a full FP32
    # temporary tensor.  The corrected energy includes the previous
    # error-feedback residual: ||G+E||² = ||G||² + ||E||² + 2<G,E>.
    residual_flat = state.error_dict.get(bucket_index)
    raw_values: list[torch.Tensor] = []
    corrected_values: list[torch.Tensor] = []
    residual_values: list[torch.Tensor] = []
    offset = 0
    for tensor in gradients:
        raw = tensor.detach()
        raw_sq = torch.linalg.vector_norm(raw, dtype=torch.float32).square()
        if residual_flat is None:
            residual_sq = torch.zeros_like(raw_sq)
            corrected_sq = raw_sq
        else:
            residual = residual_flat[offset : offset + tensor.numel()].view_as(tensor)
            residual_sq = torch.linalg.vector_norm(
                residual.detach(), dtype=torch.float32
            ).square()
            cross = torch.sum(raw * residual.detach(), dtype=torch.float32)
            corrected_sq = (raw_sq + residual_sq + 2.0 * cross).clamp_min(0.0)
        raw_values.append(raw_sq)
        corrected_values.append(corrected_sq)
        residual_values.append(residual_sq)
        offset += tensor.numel()
    local_energy = torch.stack(corrected_values)
    raw_energy = torch.stack(raw_values)
    residual_energy = torch.stack(residual_values)
    accumulator = state.norm_energy_dict.get(bucket_index)
    if accumulator is None or accumulator.shape != local_energy.shape:
        accumulator = torch.zeros_like(local_energy)
        state.norm_energy_dict[bucket_index] = accumulator
        state.raw_norm_energy_dict[bucket_index] = torch.zeros_like(local_energy)
        state.residual_energy_dict[bucket_index] = torch.zeros_like(local_energy)
        state.previous_window_norm_dict.pop(bucket_index, None)
    accumulator.add_(local_energy)
    state.raw_norm_energy_dict[bucket_index].add_(raw_energy)
    state.residual_energy_dict[bucket_index].add_(residual_energy)

    completed_iteration = state.iter + 1
    if completed_iteration % state.detection_window != 0:
        return

    group = state.process_group if state.process_group is not None else dist.group.WORLD
    current_energy = accumulator.clone()
    current_raw_energy = state.raw_norm_energy_dict[bucket_index].clone()
    current_residual_energy = state.residual_energy_dict[bucket_index].clone()
    dist.all_reduce(current_energy, group=group)
    dist.all_reduce(current_raw_energy, group=group)
    dist.all_reduce(current_residual_energy, group=group)
    current_norm = torch.sqrt(current_energy.clamp_min_(0))
    tensor_numels = torch.tensor(
        [tensor.numel() for tensor in gradients],
        device=current_energy.device,
        dtype=torch.long,
    )
    bucket_numel = int(tensor_numels.sum().item())
    eligible_mask = torch.tensor(
        [
            tensor.ndim > 1
            and _should_powersgd_compress(
                tensor.shape[0],
                tensor.numel() // tensor.shape[0],
                min(tensor.shape[0], tensor.numel() // tensor.shape[0], state.matrix_approximation_rank),
                state.min_compression_rate,
            )
            for tensor in gradients
        ],
        device=current_energy.device,
        dtype=torch.bool,
    )
    rank1_payload_numels = torch.tensor(
        [
            (tensor.shape[0] + tensor.numel() // tensor.shape[0])
            * min(tensor.shape[0], tensor.numel() // tensor.shape[0], state.matrix_approximation_rank)
            if bool(eligible)
            else tensor.numel()
            for tensor, eligible in zip(gradients, eligible_mask.tolist())
        ],
        device=current_energy.device,
        dtype=torch.long,
    )
    state.last_window_total_numel_dict[bucket_index] = bucket_numel
    state.last_window_rank1_eligible_numel_dict[bucket_index] = int(
        tensor_numels[eligible_mask].sum().item()
    )
    state.last_window_rank1_payload_numel_dict[bucket_index] = int(
        rank1_payload_numels[eligible_mask].sum().item()
        + tensor_numels[~eligible_mask].sum().item()
    )
    previous_norm = state.previous_window_norm_dict.get(bucket_index)
    if previous_norm is not None and previous_norm.shape == current_norm.shape:
        relative_change = (current_norm - previous_norm).abs() / previous_norm.clamp_min(
            1e-12
        )
        # A sharp change in corrected norm is one risk signal.  A growing
        # error-feedback debt is another: when ||E||/||G|| reaches 5%, the
        # tensor is conservatively kept dense even if its window norm is
        # otherwise stable.  This uses only the scalar norm telemetry already
        # collected and keeps rank-1 as the sole compressed rank.
        residual_ratio = torch.sqrt(
            current_residual_energy.clamp_min_(0)
        ) / (torch.sqrt(current_raw_energy.clamp_min_(0)) + 1e-12)
        critical_mask = (relative_change >= state.relative_change_threshold) | (
            residual_ratio >= 0.05
        )
        # Keep the decision per tensor.  The communication hook uses this
        # mask to dense-reduce only critical tensors while applying rank-1
        # PowerSGD to the remaining eligible tensors in the same bucket.
        state.critical_tensor_mask_dict[bucket_index] = [
            bool(value) for value in critical_mask.tolist()
        ]
        expiries = state.critical_until_tensor_iter_dict.get(
            bucket_index, [-(10**9)] * len(gradients)
        )
        if len(expiries) != len(gradients):
            expiries = [-(10**9)] * len(gradients)
        protection_end = state.iter + 1 + state.protection_steps
        state.critical_until_tensor_iter_dict[bucket_index] = [
            max(old_expiry, protection_end) if bool(is_critical) else old_expiry
            for old_expiry, is_critical in zip(expiries, critical_mask.tolist())
        ]
        critical_count = int(critical_mask.sum().item())
        critical_numel = int(tensor_numels[critical_mask].sum().item())
        state.last_window_critical_numel_dict[bucket_index] = critical_numel
        if critical_count:
            first_protected_iter = state.iter + 1
            state.critical_until_iter_dict[bucket_index] = max(
                state.critical_until_iter_dict.get(bucket_index, -1),
                first_protected_iter + state.protection_steps,
            )
        if is_rank_zero():
            logger.info(
                "Accordion window end_iter=%d bucket=%d max_relative_change=%.6f "
                "max_residual_ratio=%.6f "
                "critical_tensors=%d/%d critical_numel=%d bucket_numel=%d "
                "critical_numel_fraction=%.6f rank1_eligible_numel=%d "
                "rank1_payload_numel=%d protect_until_iter=%d",
                completed_iteration,
                bucket_index,
                relative_change.max().item(),
                residual_ratio.max().item(),
                critical_count,
                relative_change.numel(),
                critical_numel,
                bucket_numel,
                critical_numel / max(bucket_numel, 1),
                state.last_window_rank1_eligible_numel_dict[bucket_index],
                state.last_window_rank1_payload_numel_dict[bucket_index],
                    state.critical_until_iter_dict.get(bucket_index, -1),
                )
    else:
        state.critical_tensor_mask_dict[bucket_index] = [
            False for _ in gradients
        ]
        state.critical_until_tensor_iter_dict[bucket_index] = [
            -(10**9) for _ in gradients
        ]
    state.previous_window_norm_dict[bucket_index] = current_norm
    state.last_window_norm_energy_dict[bucket_index] = current_energy.detach()
    accumulator.zero_()
    state.raw_norm_energy_dict[bucket_index].zero_()
    state.residual_energy_dict[bucket_index].zero_()


def _exact_allreduce_with_powersgd_residual(
    state: PowerSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Apply one dense update, repaying and clearing local PowerSGD debt."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])
    state.error_dict.pop(bucket_index, None)
    state.p_memory_dict.pop(bucket_index, None)
    state.q_memory_dict.pop(bucket_index, None)
    state.maybe_increase_iter(bucket)
    return dist.all_reduce(
        input_tensor, group=group, async_op=True
    ).get_future().then(lambda fut: fut.value()[0].div_(world_size))


def accordion_powerSGD_plus_hook(
    state: AccordionPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Select dense or rank-1 communication per tensor from norm windows."""
    _accordion_observe_gradient_window(state, bucket)
    # ``layerwise_powerSGD_hook`` already supports mixed dense/compressed
    # tensors inside one DDP bucket.  Accordion supplies the per-tensor mask;
    # non-critical eligible tensors are always rank 1 (never rank 2+).
    return layerwise_powerSGD_hook(state, bucket)


_SUPPORTED_COMM_HOOKS["oracle_error_budget_powersgd_hook"] = (
    oracle_error_budget_powerSGD_hook
)
_SUPPORTED_COMM_HOOKS["accordion_powersgd_plus_hook"] = (
    accordion_powerSGD_plus_hook
)


def _allocate_acp_factors(
    state: ACPSGDState,
    bucket_index: int,
    tensor_specs: list[tuple[torch.Tensor, int, int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Allocate contiguous ACP factors and initialize a common random Q."""
    total_p_size = sum(rows * rank for _, rows, _, rank in tensor_specs)
    total_q_size = sum(cols * rank for _, _, cols, rank in tensor_specs)
    state.p_memory_dict[bucket_index] = torch.empty(
        total_p_size, device=device, dtype=dtype
    )
    state.q_memory_dict[bucket_index] = torch.empty(
        total_q_size, device=device, dtype=dtype
    )

    ps: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    p_offset = 0
    q_offset = 0
    for _, rows, cols, rank in tensor_specs:
        ps.append(
            state.p_memory_dict[bucket_index][
                p_offset : p_offset + rows * rank
            ].view(rows, rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_offset : q_offset + cols * rank
            ].view(cols, rank)
        )
        p_offset += rows * rank
        q_offset += cols * rank

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(state.rng.randint(1_000_000_000))
        for q in qs:
            q.copy_(torch.randn(q.shape, device="cpu", dtype=dtype))
            _orthogonalize(q.unsqueeze(0), state.orthogonalization_epsilon)
    return ps, qs


def _view_acp_factors(
    state: ACPSGDState,
    bucket_index: int,
    tensor_specs: list[tuple[torch.Tensor, int, int, int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ps: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    p_offset = 0
    q_offset = 0
    for _, rows, cols, rank in tensor_specs:
        ps.append(
            state.p_memory_dict[bucket_index][
                p_offset : p_offset + rows * rank
            ].view(rows, rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_offset : q_offset + cols * rank
            ].view(cols, rank)
        )
        p_offset += rows * rank
        q_offset += cols * rank
    return ps, qs


def acp_sgd_hook(
    state: ACPSGDState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Alternate P/Q compression and communication as proposed by ACP-SGD."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(input_tensor, group=group, async_op=True).get_future().then(
            lambda fut: fut.value()[0].div_(world_size)
        )

    bucket_index = bucket.index()
    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])
    else:
        state.error_dict[bucket_index] = torch.zeros_like(input_tensor)
    local_corrected = input_tensor.detach().clone()

    tensor_specs: list[tuple[torch.Tensor, int, int, int]] = []
    uncompressed_tensors: list[torch.Tensor] = []
    for tensor in bucket.gradients():
        matrix = tensor.view(tensor.shape[0], -1)
        rows, cols = matrix.shape
        rank = min(rows, cols, state.matrix_approximation_rank)
        if _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        ):
            tensor_specs.append((matrix, rows, cols, rank))
            state.total_numel_before_compression += rows * cols
            state.total_numel_after_compression += (
                rows * rank
                if (state.iter - state.start_powerSGD_iter) % 2 == 0
                else cols * rank
            )
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_before_compression += tensor.numel()
            state.total_numel_after_compression += tensor.numel()

    expected_p_size = sum(rows * rank for _, rows, _, rank in tensor_specs)
    expected_q_size = sum(cols * rank for _, _, cols, rank in tensor_specs)
    must_allocate = (
        bucket_index not in state.p_memory_dict
        or state.p_memory_dict[bucket_index].numel() != expected_p_size
        or state.q_memory_dict[bucket_index].numel() != expected_q_size
    )
    if must_allocate:
        ps, qs = _allocate_acp_factors(
            state,
            bucket_index,
            tensor_specs,
            input_tensor.device,
            input_tensor.dtype,
        )
    else:
        ps, qs = _view_acp_factors(state, bucket_index, tensor_specs)

    uncompressed_memory = (
        torch.cat([tensor.reshape(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    uncompressed_future = dist.all_reduce(
        uncompressed_memory, group=group, async_op=True
    ).get_future()

    # Start with P on the first compressed iteration, then alternate P/Q.
    communicate_p = (state.iter - state.start_powerSGD_iter) % 2 == 0
    compressed_iter = state.iter - state.start_powerSGD_iter
    if (
        bucket_index == 0
        and is_rank_zero()
        and (compressed_iter < 4 or compressed_iter % 100 == 0)
    ):
        logger.info(
            "%s compression: iter=%d compressed_iter=%d factor=%s rank=%d",
            type(state).__name__,
            state.iter,
            compressed_iter,
            "P" if communicate_p else "Q",
            state.matrix_approximation_rank,
        )
    local_approximation = local_corrected.clone()
    flat_offset = 0
    compressed_spec_index = 0
    for tensor in bucket.gradients():
        numel = tensor.numel()
        matrix = tensor.view(tensor.shape[0], -1)
        rows, cols = matrix.shape
        rank = min(rows, cols, state.matrix_approximation_rank)
        if _should_powersgd_compress(
            rows, cols, rank, state.min_compression_rate
        ):
            p = ps[compressed_spec_index]
            q = qs[compressed_spec_index]
            if communicate_p:
                _orthogonalize(q.unsqueeze(0), state.orthogonalization_epsilon)
                torch.mm(matrix, q, out=p)
            else:
                _orthogonalize(p.unsqueeze(0), state.orthogonalization_epsilon)
                torch.mm(matrix.transpose(0, 1), p, out=q)
            local_approximation[
                flat_offset : flat_offset + numel
            ].view(rows, cols).copy_(p @ q.transpose(0, 1))
            compressed_spec_index += 1
        flat_offset += numel

    state.error_dict[bucket_index] = local_corrected - local_approximation
    factor_memory = (
        state.p_memory_dict[bucket_index]
        if communicate_p
        else state.q_memory_dict[bucket_index]
    )
    factor_future = dist.all_reduce(
        factor_memory, group=group, async_op=True
    ).get_future()

    def finish(fut):
        reduced_factor = fut.value()[0].div_(world_size)
        if communicate_p:
            state.p_memory_dict[bucket_index] = reduced_factor
        else:
            state.q_memory_dict[bucket_index] = reduced_factor

        reduced_uncompressed = uncompressed_future.wait()[0].div_(world_size)
        offset = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                reduced_uncompressed[offset : offset + tensor.numel()].view_as(tensor)
            )
            offset += tensor.numel()
        for (matrix, _, _, _), p, q in zip(tensor_specs, ps, qs):
            matrix.copy_(p @ q.transpose(0, 1))
        if input_tensor.is_cuda:
            torch.cuda.synchronize(input_tensor.device)
        state.maybe_increase_iter(bucket)
        return input_tensor

    return factor_future.then(finish)


def acp_powerSGD_plus_hook(
    state: ACPPowerSGDPlusState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Alternate P/Q communication with periodic PowerSGD+ restarts."""
    if state.iter < state.start_powerSGD_iter:
        return acp_sgd_hook(state, bucket)

    compressed_iter = state.iter - state.start_powerSGD_iter
    if compressed_iter % state.restart_period != 0:
        return acp_sgd_hook(state, bucket)
    return _powersgd_plus_svd_restart(state, bucket)


_SUPPORTED_COMM_HOOKS["acp_powersgd_plus_hook"] = acp_powerSGD_plus_hook


def _separate_projection_rank(
    rows: int,
    cols: int,
    compression_ratio: float,
) -> int:
    return min(cols, max(1, int(math.ceil(cols / compression_ratio))))


def separate_hook(
    state: SeparateState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Compress gradients using SEPARATE common Gaussian projections."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    if state.iter < state.start_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(input_tensor, group=group, async_op=True).get_future().then(
            lambda fut: fut.value()[0].div_(world_size)
        )

    bucket_index = bucket.index()
    if bucket_index in state.error_dict:
        input_tensor.add_(state.error_dict[bucket_index])
    else:
        state.error_dict[bucket_index] = torch.zeros_like(input_tensor)
    local_corrected = input_tensor.detach().clone()

    compressed_specs = []
    uncompressed_tensors: list[torch.Tensor] = []
    total_projected_size = 0
    for tensor_index, tensor in enumerate(bucket.gradients()):
        matrix = tensor.view(tensor.shape[0], -1)
        transpose = matrix.shape[0] > matrix.shape[1]
        oriented = matrix.transpose(0, 1) if transpose else matrix
        rows, cols = oriented.shape
        projection_rank = _separate_projection_rank(
            rows, cols, state.compression_ratio
        )
        should_compress = (
            rows * projection_rank * state.min_compression_rate < rows * cols
        )
        if should_compress:
            seed = (
                state.random_seed
                + state.iter * 1_000_003
                + bucket_index * 9_176
                + tensor_index
            )
            compressed_specs.append(
                (
                    tensor_index,
                    matrix,
                    oriented,
                    transpose,
                    rows,
                    cols,
                    projection_rank,
                    seed,
                )
            )
            total_projected_size += rows * projection_rank
        else:
            uncompressed_tensors.append(tensor)

    projected_memory = torch.empty(
        total_projected_size,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    projections: list[torch.Tensor] = []
    projected_views: list[torch.Tensor] = []
    local_approximation = local_corrected.clone()
    projected_offset = 0
    flat_offset = 0
    spec_index = 0
    for tensor_index, tensor in enumerate(bucket.gradients()):
        numel = tensor.numel()
        matrix = tensor.view(tensor.shape[0], -1)
        if (
            spec_index < len(compressed_specs)
            and compressed_specs[spec_index][0] == tensor_index
        ):
            (
                _,
                _,
                oriented,
                transpose,
                rows,
                cols,
                projection_rank,
                seed,
            ) = compressed_specs[spec_index]
            generator = torch.Generator(device=input_tensor.device)
            generator.manual_seed(seed)
            projection = torch.randn(
                (cols, projection_rank),
                generator=generator,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ).mul_(projection_rank ** -0.5)
            projected = projected_memory[
                projected_offset : projected_offset + rows * projection_rank
            ].view(rows, projection_rank)
            torch.mm(oriented, projection, out=projected)
            local_reconstruction = projected @ projection.transpose(0, 1)
            if transpose:
                local_reconstruction = local_reconstruction.transpose(0, 1)
            local_approximation[
                flat_offset : flat_offset + numel
            ].view_as(matrix).copy_(local_reconstruction)
            projections.append(projection)
            projected_views.append(projected)
            projected_offset += rows * projection_rank
            spec_index += 1
        flat_offset += numel

    residual = local_corrected - local_approximation
    next_iter = state.iter + 1
    if next_iter % state.error_reset_interval == 0:
        state.error_dict[bucket_index].zero_()
    else:
        state.error_dict[bucket_index].mul_(1.0 - state.error_feedback_beta).add_(
            residual, alpha=state.error_feedback_beta
        )

    uncompressed_memory = (
        torch.cat([tensor.reshape(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.empty(0, device=input_tensor.device, dtype=input_tensor.dtype)
    )
    uncompressed_future = dist.all_reduce(
        uncompressed_memory, group=group, async_op=True
    ).get_future()
    projected_future = dist.all_reduce(
        projected_memory, group=group, async_op=True
    ).get_future()

    def finish(fut):
        reduced_projected = fut.value()[0].div_(world_size)
        reduced_uncompressed = uncompressed_future.wait()[0].div_(world_size)
        offset = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                reduced_uncompressed[offset : offset + tensor.numel()].view_as(tensor)
            )
            offset += tensor.numel()
        for spec, projection, projected in zip(
            compressed_specs, projections, projected_views
        ):
            _, matrix, _, transpose, _, _, _, _ = spec
            reconstruction = projected @ projection.transpose(0, 1)
            if transpose:
                reconstruction = reconstruction.transpose(0, 1)
            matrix.copy_(reconstruction)
        if input_tensor.is_cuda:
            torch.cuda.synchronize(input_tensor.device)
        state.maybe_increase_iter(bucket)
        return input_tensor

    return projected_future.then(finish)


_SUPPORTED_COMM_HOOKS["dynamic_powersgd_hook"] = dynamic_powerSGD_hook
_SUPPORTED_COMM_HOOKS["acp_sgd_hook"] = acp_sgd_hook
_SUPPORTED_COMM_HOOKS["separate_hook"] = separate_hook


def build_comm_hook(
    hook: Callable | str,
    *,
    use_logging: bool = False,
    process_group=None,
    powersgd_matrix_approximation_rank: int = 1,
    powersgd_start_iter: int = 1_000,
    powersgd_min_compression_rate: float = 2.0,
    powersgd_plus_restart_period: int = 50,
    powersgd_plus_restart_method: str = "approximate",
    powersgd_plus_error_averaging_period: int = 20,
    powersgd_window_ranges: str = "",
    powersgd_global_step_offset: int = 0,
    powersgd_scheduled_start_step: int = 300,
    powersgd_lr_high_rank: int = 2,
    powersgd_lr_high_rank_end_step: int = 452,
    powersgd_period_schedule: str = "300-452:25,453-734:50,735-1000:100",
    ef21_powersgd_log_interval: int = 25,
    dynamic_powersgd_low_rank: int = 1,
    dynamic_powersgd_high_rank: int = 4,
    dynamic_powersgd_adapt_interval: int = 5,
    dynamic_powersgd_relative_change_threshold: float = 0.1,
    layerwise_powersgd_candidate_ranks: str | tuple[int, ...] = "1,2,4,8,16",
    layerwise_powersgd_relative_error: float = 0.1,
    layerwise_powersgd_refresh_period: int = 50,
    layerwise_powersgd_power_iterations: int = 2,
    layerwise_powersgd_uncompressed_patterns: str = "",
    layerwise_powersgd_log_tensor_stats: bool = False,
    oracle_error_budget_random_seed: int = 0,
    accordion_detection_window: int = 50,
    accordion_relative_change_threshold: float = 0.5,
    accordion_protection_steps: int = 25,
    parameter_names: dict[int, str] | None = None,
    separate_compression_ratio: float = 16.0,
    separate_error_feedback_beta: float = 0.95,
    separate_error_reset_interval: int = 128,
) -> tuple[Any, Callable]:
    """Build the state and callable required by ``DDP.register_comm_hook``.

    PowerSGD approximates each compressible matrix-shaped gradient ``M`` as
    ``P @ Q.T``. It communicates ``(rows + cols) * rank`` elements instead of
    ``rows * cols`` and only compresses when::

        (rows + cols) * rank * min_compression_rate < rows * cols

    Error feedback and warm start intentionally use PyTorch's defaults because
    both materially improve convergence for this lossy compressor.
    """
    is_powersgd = hook == "powersgd_hook" or hook is powerSGD_hook
    is_powersgd_fp32 = hook == "powersgd_fp32_hook" or hook is powerSGD_fp32_hook
    is_ef21_powersgd = (
        hook == "ef21_powersgd_hook" or hook is ef21_powerSGD_hook
    )
    is_powersgd_plus = hook == "powersgd_plus_hook" or hook is powerSGD_plus_hook
    is_powersgd_plus_dense_v = (
        hook == "powersgd_plus_dense_v_hook" or hook is dense_v_powersgd_plus_hook
    )
    is_powersgd_plus_error_averaging = (
        hook == "powersgd_plus_error_averaging_hook"
        or hook is powerSGD_plus_error_averaging_hook
    )
    is_loss_triggered_powersgd_plus = (
        hook == "loss_triggered_powersgd_plus_hook"
        or hook is loss_triggered_powerSGD_plus_hook
    )
    is_windowed_powersgd_plus = (
        hook == "windowed_powersgd_plus_hook"
        or hook is windowed_powerSGD_plus_hook
    )
    is_acp_powersgd_plus = (
        hook == "acp_powersgd_plus_hook" or hook is acp_powerSGD_plus_hook
    )
    scheduled_policies = {
        "exact_flush_powersgd_plus_hook": "exact_flush",
        "lr_rank_powersgd_plus_hook": "lr_rank",
        "dynamic_period_powersgd_plus_hook": "dynamic_period",
    }
    scheduled_policy = scheduled_policies.get(hook) if isinstance(hook, str) else None
    is_scheduled_powersgd_plus = scheduled_policy is not None
    is_dynamic_powersgd = (
        hook == "dynamic_powersgd_hook" or hook is dynamic_powerSGD_hook
    )
    is_layerwise_powersgd = (
        hook == "layerwise_powersgd_hook" or hook is layerwise_powerSGD_hook
    )
    is_oracle_error_budget_powersgd = (
        hook == "oracle_error_budget_powersgd_hook"
        or hook is oracle_error_budget_powerSGD_hook
    )
    is_accordion_powersgd_plus = (
        hook == "accordion_powersgd_plus_hook"
        or hook is accordion_powerSGD_plus_hook
    )
    is_acp_sgd = hook == "acp_sgd_hook" or hook is acp_sgd_hook
    is_separate = hook == "separate_hook" or hook is separate_hook
    state = None
    if (
        is_powersgd
        or is_powersgd_fp32
        or is_ef21_powersgd
        or is_powersgd_plus
        or is_powersgd_plus_dense_v
        or is_powersgd_plus_error_averaging
        or is_loss_triggered_powersgd_plus
        or is_windowed_powersgd_plus
        or is_acp_powersgd_plus
        or is_scheduled_powersgd_plus
        or is_dynamic_powersgd
        or is_layerwise_powersgd
        or is_oracle_error_budget_powersgd
        or is_accordion_powersgd_plus
        or is_acp_sgd
    ):
        if powersgd_matrix_approximation_rank <= 0:
            raise ValueError(
                "ddp_powersgd_matrix_approximation_rank must be greater than 0"
            )
        if powersgd_start_iter <= 1:
            raise ValueError(
                "ddp_powersgd_start_iter must be greater than 1 because PowerSGD "
                "error feedback and warm start are enabled"
            )
        if powersgd_min_compression_rate <= 0:
            raise ValueError(
                "ddp_powersgd_min_compression_rate must be greater than 0"
            )
        if is_ef21_powersgd:
            state_cls = EF21PowerSGDState
        elif is_scheduled_powersgd_plus:
            state_cls = ScheduledPowerSGDPlusState
        elif is_windowed_powersgd_plus:
            state_cls = WindowedPowerSGDPlusState
        elif is_acp_powersgd_plus:
            state_cls = ACPPowerSGDPlusState
        elif is_accordion_powersgd_plus:
            state_cls = AccordionPowerSGDPlusState
        elif is_powersgd_plus_error_averaging:
            state_cls = PowerSGDPlusErrorAveragingState
        elif is_powersgd_plus_dense_v:
            state_cls = DenseVPowerSGDPlusState
        elif is_powersgd_plus:
            state_cls = PowerSGDPlusState
        elif is_loss_triggered_powersgd_plus:
            state_cls = LossTriggeredPowerSGDPlusState
        elif is_dynamic_powersgd:
            state_cls = DynamicPowerSGDState
        elif is_layerwise_powersgd:
            state_cls = LayerwisePowerSGDState
        elif is_oracle_error_budget_powersgd:
            state_cls = OracleErrorBudgetPowerSGDState
        elif is_acp_sgd:
            state_cls = ACPSGDState
        else:
            state_cls = PowerSGDState
        state_kwargs = {}
        if is_ef21_powersgd:
            state_kwargs.update(
                global_step_offset=powersgd_global_step_offset,
                compression_start_step=powersgd_scheduled_start_step,
                log_interval=ef21_powersgd_log_interval,
                orthogonalization_epsilon=1e-8,
            )
        elif (
            is_powersgd_plus
            or is_powersgd_plus_dense_v
            or is_powersgd_plus_error_averaging
            or is_loss_triggered_powersgd_plus
            or is_windowed_powersgd_plus
            or is_acp_powersgd_plus
            or is_accordion_powersgd_plus
            or is_scheduled_powersgd_plus
        ):
            if powersgd_plus_restart_period <= 0:
                raise ValueError(
                    "ddp_powersgd_plus_restart_period must be greater than 0"
                )
            state_kwargs["restart_period"] = powersgd_plus_restart_period
            state_kwargs["restart_method"] = powersgd_plus_restart_method
            if is_powersgd_plus_error_averaging:
                state_kwargs["error_averaging_period"] = (
                    powersgd_plus_error_averaging_period
                )
            if is_loss_triggered_powersgd_plus:
                state_kwargs.update(
                    loss_threshold=float(os.getenv("POWER_SGD_LOSS_THRESHOLD", "0.01")),
                    loss_reference_path=os.getenv("POWER_SGD_LOSS_REFERENCE", ""),
                    loss_current_path=os.getenv("POWER_SGD_LOSS_CURRENT", ""),
                    loss_event_path=os.getenv("POWER_SGD_LOSS_EVENTS", ""),
                )
            if is_scheduled_powersgd_plus:
                state_kwargs.update(
                    policy=scheduled_policy,
                    compression_start_step=powersgd_scheduled_start_step,
                    global_step_offset=powersgd_global_step_offset,
                    high_rank=powersgd_lr_high_rank,
                    high_rank_end_step=powersgd_lr_high_rank_end_step,
                    period_schedule=powersgd_period_schedule,
                    # A rank transition can expose exactly rank-deficient
                    # buckets. Keep Gram-Schmidt finite without changing the
                    # communicated rank or policy.
                    orthogonalization_epsilon=1e-8,
                )
            elif is_windowed_powersgd_plus:
                state_kwargs["window_ranges"] = powersgd_window_ranges
                state_kwargs["global_step_offset"] = powersgd_global_step_offset
            elif is_accordion_powersgd_plus:
                state_kwargs.update(
                    detection_window=accordion_detection_window,
                    relative_change_threshold=(
                        accordion_relative_change_threshold
                    ),
                    protection_steps=accordion_protection_steps,
                    parameter_names=parameter_names,
                    uncompressed_patterns=(
                        layerwise_powersgd_uncompressed_patterns
                    ),
                    power_iterations=layerwise_powersgd_power_iterations,
                    log_tensor_stats=layerwise_powersgd_log_tensor_stats,
                    orthogonalization_epsilon=1e-8,
                )
        elif is_dynamic_powersgd:
            state_kwargs.update(
                low_rank=dynamic_powersgd_low_rank,
                high_rank=dynamic_powersgd_high_rank,
                adapt_interval=dynamic_powersgd_adapt_interval,
                relative_change_threshold=(
                    dynamic_powersgd_relative_change_threshold
                ),
                # Dynamic rank increases can expose exactly rank-deficient
                # gradients; epsilon prevents a zero column from producing
                # NaNs during Gram-Schmidt orthogonalization.
                orthogonalization_epsilon=1e-8,
            )
        elif is_layerwise_powersgd or is_oracle_error_budget_powersgd:
            if isinstance(layerwise_powersgd_candidate_ranks, str):
                try:
                    candidate_ranks = tuple(
                        int(value.strip())
                        for value in layerwise_powersgd_candidate_ranks.split(",")
                        if value.strip()
                    )
                except ValueError as exc:
                    raise ValueError(
                        "ddp_layerwise_powersgd_candidate_ranks must be a "
                        "comma-separated list of integers"
                    ) from exc
            else:
                candidate_ranks = tuple(layerwise_powersgd_candidate_ranks)
            state_kwargs.update(
                candidate_ranks=candidate_ranks,
                relative_error_threshold=layerwise_powersgd_relative_error,
                refresh_period=layerwise_powersgd_refresh_period,
                power_iterations=layerwise_powersgd_power_iterations,
                parameter_names=parameter_names,
                uncompressed_patterns=(
                    layerwise_powersgd_uncompressed_patterns
                ),
                log_tensor_stats=layerwise_powersgd_log_tensor_stats,
                orthogonalization_epsilon=1e-8,
            )
            if is_oracle_error_budget_powersgd:
                state_kwargs["random_seed"] = oracle_error_budget_random_seed
            # The inherited scalar rank is not used by the variable-rank hook,
            # but keeping it at the maximum candidate makes state diagnostics
            # meaningful.
            if candidate_ranks:
                powersgd_matrix_approximation_rank = max(candidate_ranks)
        state = state_cls(
            process_group=process_group,
            matrix_approximation_rank=powersgd_matrix_approximation_rank,
            start_powerSGD_iter=powersgd_start_iter,
            min_compression_rate=powersgd_min_compression_rate,
            **state_kwargs,
        )
    elif is_separate:
        state = SeparateState(
            process_group=process_group,
            start_iter=powersgd_start_iter,
            compression_ratio=separate_compression_ratio,
            min_compression_rate=powersgd_min_compression_rate,
            error_feedback_beta=separate_error_feedback_beta,
            error_reset_interval=separate_error_reset_interval,
        )

    return state, resolve_comm_hook(hook, use_logging=use_logging)


def resolve_comm_hook(hook: Callable | str, use_logging: bool = False) -> Callable:
    """Resolve a comm hook name/callable, optionally wrapped with logging.

    Args:
        hook: Either a DDP comm hook callable (e.g. ``allreduce_hook``,
            ``fp16_compress_hook``, ``bf16_compress_hook``, ``powerSGD_hook``)
            or its name.
        use_logging: If True, wrap the resolved hook so it logs bucket info
            on rank 0 before/after it runs.

    Returns:
        A comm hook with the ``(process_group, bucket)`` signature.
    """
    if isinstance(hook, str):
        try:
            hook = _SUPPORTED_COMM_HOOKS[hook]
        except KeyError:
            raise ValueError(
                f"Unsupported comm hook name {hook!r}, expected one of "
                f"{sorted(_SUPPORTED_COMM_HOOKS)}."
            ) from None

    if not use_logging:
        return hook

    hook_name = hook.__name__

    def logging_comm_hook(process_group, bucket):
        if is_rank_zero():
            tensor = bucket.buffer()
            logger.info(
                "DDP %s: bucket_index=%d numel=%d dtype=%s",
                hook_name,
                bucket.index(),
                tensor.numel(),
                tensor.dtype,
            )
        fut = hook(process_group, bucket)
        if is_rank_zero():
            fut.add_done_callback(
                lambda fut: logger.info(
                    "DDP %s done: bucket_index=%d", hook_name, bucket.index()
                )
            )
        return fut

    return logging_comm_hook
