#!/usr/bin/env python3
"""
Generate a realistic sample JVM GC log for testing gc_analyzer.py.
Produces a mixed file with JDK 8-style and JDK 9+-style entries.

Usage:
    python generate_sample_gc_log.py [--format jdk8|jdk9|g1|mixed] [--events N] [--out FILE]
"""

import argparse
import random
import math
from pathlib import Path


def _jdk8_young(t: float, before_kb: int, after_kb: int, total_kb: int, pause_s: float) -> str:
    gen_before = int(before_kb * 0.7)
    gen_after = int(after_kb * 0.1)
    gen_total = int(total_kb * 0.35)
    return (
        f"{t:.3f}: [GC (Allocation Failure) "
        f"[PSYoungGen: {gen_before}K->{gen_after}K({gen_total}K)] "
        f"{before_kb}K->{after_kb}K({total_kb}K), "
        f"{pause_s:.7f} secs] "
        f"[Times: user={pause_s*2:.2f} sys=0.00, real={pause_s:.2f} secs]"
    )


def _jdk8_full(t: float, before_kb: int, after_kb: int, total_kb: int, pause_s: float) -> str:
    old_before = int(before_kb * 0.6)
    old_after = int(after_kb * 0.8)
    old_total = int(total_kb * 0.65)
    return (
        f"{t:.3f}: [Full GC (Ergonomics) "
        f"[PSYoungGen: 0K->0K({int(total_kb*0.35)}K)] "
        f"[ParOldGen: {old_before}K->{old_after}K({old_total}K)] "
        f"{before_kb}K->{after_kb}K({total_kb}K), "
        f"[Metaspace: 45000K->45000K(1105920K)], "
        f"{pause_s:.7f} secs] "
        f"[Times: user={pause_s*3:.2f} sys=0.00, real={pause_s:.2f} secs]"
    )


def _jdk9_young(t: float, before_kb: int, after_kb: int, total_kb: int, pause_ms: float, gc_id: int) -> str:
    bm, am, tm = before_kb // 1024, after_kb // 1024, total_kb // 1024
    return f"[{t:.3f}s][info][gc] GC({gc_id}) Pause Young (Normal) (G1 Evacuation Pause) {bm}M->{am}M({tm}M) {pause_ms:.3f}ms"


def _jdk9_full(t: float, before_kb: int, after_kb: int, total_kb: int, pause_ms: float, gc_id: int) -> str:
    bm, am, tm = before_kb // 1024, after_kb // 1024, total_kb // 1024
    return f"[{t:.3f}s][info][gc] GC({gc_id}) Pause Full (System.gc()) {bm}M->{am}M({tm}M) {pause_ms:.3f}ms"


def _g1_young(t: float, before_kb: int, after_kb: int, total_kb: int, pause_s: float) -> str:
    bm, am, tm = before_kb // 1024, after_kb // 1024, total_kb // 1024
    return f"{t:.3f}: [GC pause (G1 Evacuation Pause) (young) {bm}M->{am}M({tm}M), {pause_s:.7f} secs]"


def _g1_mixed(t: float, before_kb: int, after_kb: int, total_kb: int, pause_s: float) -> str:
    bm, am, tm = before_kb // 1024, after_kb // 1024, total_kb // 1024
    return f"{t:.3f}: [GC pause (G1 Evacuation Pause) (mixed) {bm}M->{am}M({tm}M), {pause_s:.7f} secs]"


def generate(fmt: str, n_events: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    lines = []

    heap_kb = 1024 * 256    # 256 MB total heap
    live_kb = 1024 * 40     # ~40 MB live set at start
    gc_id = 0
    t = 0.5

    for i in range(n_events):
        # Simulate allocation ramp — gradual live set growth (mild leak)
        live_kb += int(rng.uniform(200, 800))
        before_kb = min(live_kb + int(rng.uniform(10_000, 60_000)), int(heap_kb * 0.92))

        is_full = (i > 0 and i % 30 == 0) or before_kb > heap_kb * 0.88
        is_mixed = (fmt in ("g1", "mixed")) and (i % 8 == 5)

        if is_full:
            after_kb = int(live_kb * rng.uniform(0.85, 0.95))
            pause_s = rng.uniform(0.08, 0.35)
        else:
            after_kb = int(live_kb * rng.uniform(0.90, 1.05))
            pause_s = rng.uniform(0.002, 0.025)

        after_kb = min(after_kb, before_kb)
        pause_ms = pause_s * 1000
        t += rng.uniform(0.5, 3.0)

        if fmt == "jdk8":
            line = (_jdk8_full if is_full else _jdk8_young)(t, before_kb, after_kb, heap_kb, pause_s)
        elif fmt == "jdk9":
            line = (_jdk9_full if is_full else _jdk9_young)(t, before_kb, after_kb, heap_kb, pause_ms, gc_id)
        elif fmt == "g1":
            if is_full:
                line = _jdk8_full(t, before_kb, after_kb, heap_kb, pause_s)
            elif is_mixed:
                line = _g1_mixed(t, before_kb, after_kb, heap_kb, pause_s)
            else:
                line = _g1_young(t, before_kb, after_kb, heap_kb, pause_s)
        else:  # mixed — alternate formats
            if is_full:
                line = (_jdk9_full if i % 2 == 0 else _jdk8_full)(t, before_kb, after_kb, heap_kb, pause_ms if i % 2 == 0 else pause_s, gc_id) if i % 2 == 0 else _jdk8_full(t, before_kb, after_kb, heap_kb, pause_s)
            elif i % 3 == 0:
                line = _jdk9_young(t, before_kb, after_kb, heap_kb, pause_ms, gc_id)
            elif i % 3 == 1:
                line = _jdk8_young(t, before_kb, after_kb, heap_kb, pause_s)
            else:
                line = _g1_young(t, before_kb, after_kb, heap_kb, pause_s)

        lines.append(line)
        gc_id += 1

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a sample JVM GC log for testing")
    ap.add_argument("--format", choices=["jdk8", "jdk9", "g1", "mixed"], default="mixed",
                    help="GC log format to emit (default: mixed)")
    ap.add_argument("--events", type=int, default=200,
                    help="Number of GC events to generate (default: 200)")
    ap.add_argument("--out", default="sample_gc.log",
                    help="Output file path (default: sample_gc.log)")
    args = ap.parse_args()

    lines = generate(args.format, args.events)
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {len(lines)} GC events ({args.format} format) -> {out}")
    print(f"Run analyzer:  python gc_analyzer.py {out} --out-dir .")


if __name__ == "__main__":
    main()
