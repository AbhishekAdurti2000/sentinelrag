#!/usr/bin/env python3
"""Render a saved trace JSON file as readable CLI output.

Usage: python scripts/view_trace.py data/traces/<run_id>.json
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    payload = json.loads(open(sys.argv[1]).read())
    print(f"Run {payload['run_id']}  (total {payload['total_latency_ms']:.0f}ms)")
    print("=" * 70)
    for event in payload["events"]:
        print(f"[{event['elapsed_ms']:>7.1f}ms] {event['step']}")
        for key, value in event["data"].items():
            text = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
            if len(text) > 400:
                text = text[:400] + " ...(truncated)"
            print(f"          {key}: {text}")
        print("-" * 70)


if __name__ == "__main__":
    main()
