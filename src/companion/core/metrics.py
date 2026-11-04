"""In-process latency metrics.

Only durations and stage names are recorded. No prompt text, conversation
content, memory content or IP address ever enters a metric. A small bounded
ring buffer per stage keeps memory predictable; ``snapshot()`` exposes
count/P50/P95/MAX for the CLI or a health endpoint.

The timings are wall-clock stage durations. Transport latency uses
``time.monotonic()`` inside the client, so a system clock adjustment cannot
produce a negative or inflated measurement there.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Iterable

DEFAULT_MAX_SAMPLES = 500


@dataclass
class LatencyMetrics:
    max_samples: int = DEFAULT_MAX_SAMPLES
    _samples: dict[str, Deque[float]] = field(default_factory=dict)

    def record(self, stage: str, milliseconds: float) -> None:
        if stage not in self._samples:
            self._samples[stage] = deque(maxlen=self.max_samples)
        if math.isfinite(milliseconds) and milliseconds >= 0:
            self._samples[stage].append(float(milliseconds))

    def record_many(self, timings: dict[str, float]) -> None:
        for stage, value in timings.items():
            self.record(stage, value)

    def percentile(self, stage: str, p: float) -> float | None:
        values = self._samples.get(stage)
        if not values:
            return None
        ordered = sorted(values)
        rank = max(1, math.ceil(p / 100.0 * len(ordered)))
        return ordered[min(rank, len(ordered)) - 1]

    def snapshot(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for stage in sorted(self._samples):
            values = self._samples[stage]
            if not values:
                continue
            result[stage] = {
                "count": len(values),
                "p50_ms": round(self.percentile(stage, 50) or 0.0, 2),
                "p95_ms": round(self.percentile(stage, 95) or 0.0, 2),
                "max_ms": round(max(values), 2),
            }
        return result

    def reset(self) -> None:
        self._samples.clear()


def summarise_stages(stages: Iterable[str]) -> str:
    """A log-safe label; never includes measured data."""
    return ",".join(sorted(set(stages)))
