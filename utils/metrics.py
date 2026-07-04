"""
Per-utterance latency tracker.

Usage:
    tracker = LatencyTracker()
    tracker.mark("asr_start")
    ... do ASR ...
    tracker.mark("asr_end")
    tracker.mark("translation_end")
    tracker.mark("tts_end")
    tracker.mark("playback_start")
    print(tracker.report())
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional


class LatencyTracker:
    def __init__(self) -> None:
        self._marks: Dict[str, float] = OrderedDict()

    def mark(self, label: str) -> None:
        self._marks[label] = time.perf_counter()

    def elapsed(self, from_label: str, to_label: str) -> Optional[float]:
        """Return elapsed milliseconds between two marks, or None if either is missing."""
        t0 = self._marks.get(from_label)
        t1 = self._marks.get(to_label)
        if t0 is None or t1 is None:
            return None
        return (t1 - t0) * 1000

    def report(self) -> str:
        marks = list(self._marks.keys())
        if len(marks) < 2:
            return "no latency data"

        lines: list[str] = []
        for i in range(len(marks) - 1):
            a, b = marks[i], marks[i + 1]
            ms = self.elapsed(a, b)
            if ms is not None:
                lines.append(f"  {a} → {b}: [bold]{ms:.0f} ms[/bold]")

        total = self.elapsed(marks[0], marks[-1])
        if total is not None:
            lines.append(f"  [dim]─────────────────────────────[/dim]")
            lines.append(f"  end-to-end: [bold green]{total:.0f} ms[/bold green]")

        return "\n".join(lines)
