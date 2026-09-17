"""Lightweight, dependency-free observability.

Rather than requiring a paid LangSmith account to demo, this logs every
step of a run (planner output, each tool call + its result, guardrail
verdicts, memory reads/writes, and the final answer) to a structured JSON
trace file. `scripts/view_trace.py` renders it as readable CLI output.
Swapping this for LangSmith/OpenTelemetry later is a one-file change
(everything funnels through `Tracer.log_event`), which is itself worth a
line in an interview about designing for observability from day one.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinelrag.config import settings
from sentinelrag.schema import now_iso


@dataclass
class TraceEvent:
    step: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=now_iso)
    elapsed_ms: float | None = None


class Tracer:
    def __init__(self, run_id: str | None = None, trace_dir: Path | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.trace_dir = trace_dir or (settings.data_dir / "traces")
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[TraceEvent] = []
        self._start = time.time()

    def log_event(self, step: str, **data: Any) -> None:
        self.events.append(TraceEvent(step=step, data=data, elapsed_ms=(time.time() - self._start) * 1000))

    def total_latency_ms(self) -> float:
        return (time.time() - self._start) * 1000

    def save(self) -> Path:
        path = self.trace_dir / f"{self.run_id}.json"
        payload = {
            "run_id": self.run_id,
            "total_latency_ms": self.total_latency_ms(),
            "events": [
                {"step": e.step, "timestamp": e.timestamp, "elapsed_ms": e.elapsed_ms, "data": e.data}
                for e in self.events
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
