#!/usr/bin/env python3
"""
JVM Garbage Collection Log Analyzer
Supports JDK 8 (PrintGCDetails/PrintGCDateStamps) and JDK 9+ (Xlog:gc*) formats.

Usage:
    python gc_analyzer.py <gc_log_file> [--out-dir <dir>] [--csv <file>]
"""

from __future__ import annotations

import re
import sys
import csv
import argparse
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class GCEvent:
    line_no: int
    raw: str
    timestamp_s: Optional[float] = None   # JVM uptime in seconds
    wall_clock_s: Optional[float] = None  # Unix timestamp from wall-clock prefix (wrapper logs)
    gc_type: str = "Unknown"              # Young | Full | Mixed | Concurrent
    cause: str = ""
    heap_before_kb: Optional[int] = None
    heap_after_kb: Optional[int] = None
    heap_total_kb: Optional[int] = None
    pause_ms: Optional[float] = None
    # CPU stats reported per-event (JDK 8 `[Times: user=X sys=Y, real=Z secs]`
    # or JDK 9+ `[gc,cpu] GC(N) User=Xs Sys=Ys Real=Zs`).
    cpu_user_s: Optional[float] = None
    cpu_sys_s:  Optional[float] = None
    cpu_real_s: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

_SIZE_PAT = re.compile(r"(\d+(?:\.\d+)?)([KMGkmg]?)")


def _to_kb(token: str) -> Optional[int]:
    """Convert '65536K', '64M', '1G' -> kilobytes. Returns None on failure."""
    m = _SIZE_PAT.fullmatch(token.strip())
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).upper()
    multiplier = {"": 1, "K": 1, "M": 1024, "G": 1048576}.get(unit, 1)
    return int(val * multiplier)


def _classify(raw: str) -> str:
    t = raw.lower()
    if "full" in t:
        return "Full"
    if any(x in t for x in ("young", "minor", "psyounggen", "parnew", "defnew")):
        return "Young"
    if "mixed" in t:
        return "Mixed"
    if "concurrent" in t:
        return "Concurrent"
    if any(x in t for x in ("evacuation", "g1")):
        return "Young"
    return raw.strip() or "Unknown"


# ── Regex patterns (tried in order; first match wins) ─────────────────────────

# JDK 9+ unified logging
# [0.123s][info][gc] GC(0) Pause Young (Normal) (G1 Evacuation Pause) 10M->5M(256M) 2.345ms
_P_JDK9 = re.compile(
    r"\[(\d+\.\d+)s\](?:\[[\w, ]+\])*\[gc[*]?\s*\]\s+"
    r"GC\(\d+\)\s+"
    r"(Pause \w+|Concurrent \w+|[\w][\w ]*?)\s+"
    r"(?:\((?:[^()]*|\([^()]*\))*\)\s*)*"  # cause/phase (handles nested parens like System.gc())
    r"(\d+(?:\.\d+)?[KMGkmg]?)"
    r"->"
    r"(\d+(?:\.\d+)?[KMGkmg]?)"
    r"\((\d+(?:\.\d+)?[KMGkmg]?)\)"
    r"\s+(\d+\.\d+)ms"
)

# G1GC JDK 8
# 1.234: [GC pause (G1 Evacuation Pause) (young) 45M->20M(256M), 0.0123456 secs]
_P_G1_JDK8 = re.compile(
    r"(\d+\.\d+):\s*"
    r"\[GC pause \(([^)]+)\)"
    r"(?:\s*\([^)]*\))?"
    r"\s+"
    r"(\d+[KMGkmg])->(\d+[KMGkmg])\((\d+[KMGkmg])\)"
    r",\s*(\d+\.\d+)\s*secs"
)

# JDK 8 generic (with optional ISO datetime prefix and optional gen detail blocks)
# 2021-01-01T00:00:00.000+0000: 3.321: [GC (Allocation Failure) [PSYoungGen: 1K->2K(3K)] 65536K->7168K(251392K), 0.003 secs]
# 3.321: [Full GC (Ergonomics) 65536K->7168K(251392K), 0.05 secs]
_P_JDK8 = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}T[\d:+.\-Z]+:\s*)?"
    r"(\d+\.\d+):\s*"
    r"\[(Full GC|GC)(?:\s*\([^)]+\))?"
    r"(?:\s+\[[^\]]*\])*"
    r"\s*(\d+[KMGkmg])->(\d+[KMGkmg])\((\d+[KMGkmg])\)"
    r"(?:,\s*\[[^\]]*\])*"          # optional blocks like [Metaspace: ...]
    r",\s*(\d+\.\d+)\s*secs"
)

# ZGC / Shenandoah (heap with percentage; pause is optional)
# [0.123s][info][gc] GC(0) Garbage Collection (Allocation Failure) 10M(4%)->5M(2%)
_P_ZGC = re.compile(
    r"\[(\d+\.\d+)s\](?:\[[\w, ]+\])*\[gc\s*\]\s+"
    r"GC\(\d+\)[^(]*"
    r"(\d+(?:\.\d+)?[KMGkmg])\(\d+%\)->(\d+(?:\.\d+)?[KMGkmg])\(\d+%\)"
    r"(?:\s+(\d+\.\d+)ms)?"
)

# JDK 9+ concurrent / remark / cleanup — no heap-change data
# [2.456s][info][gc] GC(0) Concurrent Mark 333.456ms
# [3.001s][info][gc] GC(1) Pause Remark 45.678ms
_P_JDK9_CONC = re.compile(
    r"\[(\d+\.\d+)s\](?:\[[\w, ]+\])*\[gc[*]?\s*\]\s+"
    r"GC\(\d+\)\s+"
    r"(Concurrent [\w ]+?|Pause Remark|Pause Cleanup)"
    r"(?:\s+\([\d.]+s,\s*[\d.]+s\))?"   # optional (start, end) timestamps
    r"\s+(\d+\.\d+)ms\s*$"
)

# JDK 8 G1GC concurrent phases and remark
# 3.333: [GC concurrent-mark, 0.333 secs]
# 3.456: [GC remark, 0.045 secs]
# 3.460: [GC cleanup 10M->8M(256M), 0.002 secs]   ← cleanup with heap handled by _P_JDK8
_P_JDK8_CONC = re.compile(
    r"(\d+\.\d+):\s*"
    r"\[GC\s+(concurrent-[\w-]+|remark)"
    r"(?:[^,\[\]]*)"
    r",\s*(\d+\.\d+)\s*secs"
)

_CAUSE_PAT = re.compile(r"\(((?:[^()]*|\([^()]*\))+)\)")

# GC ID, e.g. "GC(123)" — used to correlate JDK 9+ [gc,cpu] lines back to events.
_GC_ID_PAT = re.compile(r"GC\((\d+)\)")

# JDK 9+ unified-logging CPU stats line, e.g.
#   [...][gc,cpu] GC(25027) User=0.13s Sys=0.00s Real=0.05s
_P_GC_CPU = re.compile(
    r"\[gc,cpu\s*\][^G]*GC\((\d+)\)\s+"
    r"User=(\d+\.\d+)s\s+Sys=(\d+\.\d+)s\s+Real=(\d+\.\d+)s"
)

# JDK 8 inline times block, e.g.
#   [Times: user=0.10 sys=0.01, real=0.05 secs]
_P_TIMES = re.compile(
    r"\[Times:\s*user=(\d+\.\d+)\s+sys=(\d+\.\d+),\s*real=(\d+\.\d+)\s*secs\]"
)

# Wall-clock timestamp patterns (for wrapper logs like tanuki/jsvc)
# Matches: 2026/02/16 09:01:14  or  2026-02-16 09:01:14  or  2026-02-16T09:01:14
_WALL_PAT = re.compile(
    r"(\d{4})[/\-](\d{2})[/\-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2})"
)


def _parse_wall_clock(line: str) -> Optional[float]:
    m = _WALL_PAT.search(line)
    if not m:
        return None
    try:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                      int(m.group(4)), int(m.group(5)), int(m.group(6)))
        return dt.timestamp()
    except ValueError:
        return None


def _attach_inline_cpu_times(ev: GCEvent, line: str) -> None:
    """Populate cpu_*_s from a JDK 8 `[Times: user=… sys=… real=…]` block, if present."""
    m = _P_TIMES.search(line)
    if m:
        ev.cpu_user_s = float(m.group(1))
        ev.cpu_sys_s  = float(m.group(2))
        ev.cpu_real_s = float(m.group(3))


def _parse_line(line: str, line_no: int) -> Optional[GCEvent]:
    ev = GCEvent(line_no=line_no, raw=line)
    ev.wall_clock_s = _parse_wall_clock(line)

    m = _P_JDK9.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.gc_type = _classify(m.group(2))
        ev.heap_before_kb = _to_kb(m.group(3))
        ev.heap_after_kb = _to_kb(m.group(4))
        ev.heap_total_kb = _to_kb(m.group(5))
        ev.pause_ms = float(m.group(6))
        causes = _CAUSE_PAT.findall(line[m.start():m.end()])
        ev.cause = causes[-1] if causes else ""
        return ev

    m = _P_ZGC.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.gc_type = "Concurrent"
        ev.heap_before_kb = _to_kb(m.group(2))
        ev.heap_after_kb = _to_kb(m.group(3))
        ev.pause_ms = float(m.group(4)) if m.group(4) else None
        return ev

    m = _P_G1_JDK8.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.cause = m.group(2)
        ev.gc_type = "Young" if "young" in ev.cause.lower() else _classify(ev.cause)
        ev.heap_before_kb = _to_kb(m.group(3))
        ev.heap_after_kb = _to_kb(m.group(4))
        ev.heap_total_kb = _to_kb(m.group(5))
        ev.pause_ms = float(m.group(6)) * 1000
        _attach_inline_cpu_times(ev, line)
        return ev

    m = _P_JDK8.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.gc_type = "Full" if "Full" in m.group(2) else "Young"
        ev.heap_before_kb = _to_kb(m.group(3))
        ev.heap_after_kb = _to_kb(m.group(4))
        ev.heap_total_kb = _to_kb(m.group(5))
        ev.pause_ms = float(m.group(6)) * 1000
        causes = _CAUSE_PAT.findall(line)
        ev.cause = causes[0] if causes else ""
        _attach_inline_cpu_times(ev, line)
        return ev

    m = _P_JDK9_CONC.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.gc_type     = "Concurrent"
        ev.cause       = m.group(2).strip()
        ev.pause_ms    = float(m.group(3))
        return ev

    m = _P_JDK8_CONC.search(line)
    if m:
        ev.timestamp_s = float(m.group(1))
        ev.gc_type     = "Concurrent"
        ev.cause       = m.group(2)
        ev.pause_ms    = float(m.group(3)) * 1000
        return ev

    return None


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> Tuple[List[GCEvent], List[str]]:
    events: List[GCEvent] = []
    unmatched: List[str] = []
    gc_id_to_event: dict = {}  # JDK 9+ correlates [gc,cpu] lines to events by GC(N)

    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip()
            if not line:
                continue
            ev = _parse_line(line, line_no)
            if ev:
                events.append(ev)
                gc_id_m = _GC_ID_PAT.search(line)
                if gc_id_m:
                    gc_id_to_event[gc_id_m.group(1)] = ev
                continue

            # JDK 9+ standalone [gc,cpu] line — attach to its event by GC ID.
            cpu_m = _P_GC_CPU.search(line)
            if cpu_m:
                target = gc_id_to_event.get(cpu_m.group(1))
                if target is not None:
                    target.cpu_user_s = float(cpu_m.group(2))
                    target.cpu_sys_s  = float(cpu_m.group(3))
                    target.cpu_real_s = float(cpu_m.group(4))
                continue

            if re.search(r"\bGC\b", line, re.IGNORECASE):
                unmatched.append(f"line {line_no:>6}: {line[:120]}")
    return events, unmatched


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(events: List[GCEvent]) -> dict:
    def _s(fn, data):
        return fn(data) if data else None

    # Pause statistics count Stop-the-World pauses only. G1 concurrent-cycle
    # durations are wall-clock time of work that runs *alongside* the
    # application, not pauses — including them would inflate the mean (this is
    # also what gceasy.io does).
    pauses_ms = [e.pause_ms for e in events
                 if e.pause_ms is not None and e.gc_type != "Concurrent"]
    reclaimed = [
        e.heap_before_kb - e.heap_after_kb
        for e in events
        if e.heap_before_kb is not None and e.heap_after_kb is not None
    ]
    before_kb = [e.heap_before_kb for e in events if e.heap_before_kb is not None]
    after_kb = [e.heap_after_kb for e in events if e.heap_after_kb is not None]
    times = sorted(e.timestamp_s for e in events if e.timestamp_s is not None)

    by_type: dict = {}
    for e in events:
        by_type[e.gc_type] = by_type.get(e.gc_type, 0) + 1

    duration_s = (times[-1] - times[0]) if len(times) >= 2 else 0
    total_pause_s = sum(pauses_ms) / 1000
    sp = sorted(pauses_ms)

    return dict(
        total=len(events),
        by_type=by_type,
        duration_s=duration_s,
        total_pause_s=total_pause_s,
        throughput_pct=((duration_s - total_pause_s) / duration_s * 100) if duration_s > 0 else None,
        gc_rate_per_min=len(events) / (duration_s / 60) if duration_s > 0 else None,
        pause_min=_s(min, sp),
        pause_max=_s(max, sp),
        pause_mean=_s(statistics.mean, sp),
        pause_median=_s(statistics.median, sp),
        pause_p95=sp[int(len(sp) * 0.95)] if len(sp) >= 20 else None,
        pause_p99=sp[int(len(sp) * 0.99)] if len(sp) >= 100 else None,
        heap_before_max_kb=_s(max, before_kb),
        heap_before_min_kb=_s(min, before_kb),
        heap_after_max_kb=_s(max, after_kb),
        heap_after_min_kb=_s(min, after_kb),
        reclaimed_total_kb=sum(reclaimed),
        reclaimed_mean_kb=_s(statistics.mean, reclaimed),
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

_COLORS = {
    "Full":       "#e74c3c",
    "Young":      "#3498db",
    "Mixed":      "#9b59b6",
    "Concurrent": "#2ecc71",
    "Unknown":    "#95a5a6",
}


def _draw_heap_chart(events: List[GCEvent], attr: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    # Faint connecting line
    pts = sorted(
        ((e.timestamp_s, getattr(e, attr) / 1024)
         for e in events if e.timestamp_s is not None and getattr(e, attr) is not None),
        key=lambda t: t[0],
    )
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#2c3e50", linewidth=0.7, alpha=0.3, zorder=2)

    # Scatter colored by GC type
    for gc_type, color in _COLORS.items():
        xs = [e.timestamp_s for e in events
              if e.gc_type == gc_type and e.timestamp_s is not None and getattr(e, attr) is not None]
        ys = [getattr(e, attr) / 1024 for e in events
              if e.gc_type == gc_type and e.timestamp_s is not None and getattr(e, attr) is not None]
        if xs:
            ax.scatter(xs, ys, color=color, label=f"{gc_type} GC", s=38, alpha=0.85, zorder=3)

    ax.set_xlabel("JVM Uptime (seconds)", fontsize=11)
    ax.set_ylabel("Heap Size (MB)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="GC Type", loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_heap(events: List[GCEvent], out_dir: Path) -> List[Path]:
    if not HAS_MPL:
        return []
    saved: List[Path] = []
    for attr, title, fname in [
        ("heap_before_kb", "Heap Before GC — Over Time", "heap_before_gc.png"),
        ("heap_after_kb",  "Heap After GC — Over Time",  "heap_after_gc.png"),
    ]:
        if any(getattr(e, attr) is not None for e in events):
            p = out_dir / fname
            _draw_heap_chart(events, attr, title, p)
            saved.append(p)
    return saved


# ── Observations ──────────────────────────────────────────────────────────────

def build_observations(events: List[GCEvent], stats: dict) -> List[str]:
    obs: List[str] = []

    if stats["pause_max"] is not None and stats["pause_max"] > 500:
        obs.append(
            f"[WARN] Max pause {stats['pause_max']:.0f} ms exceeds 500 ms threshold. "
            "Consider increasing heap or switching to a low-latency GC (ZGC/Shenandoah)."
        )

    if stats["throughput_pct"] is not None and stats["throughput_pct"] < 95:
        obs.append(
            f"[WARN] Application throughput is {stats['throughput_pct']:.1f}% "
            "(target ≥ 95%). JVM is spending too much time collecting garbage."
        )

    full_count = stats["by_type"].get("Full", 0)
    if full_count > 0:
        obs.append(
            f"[WARN] {full_count} Full GC event(s) detected. "
            "Full GCs are stop-the-world events and indicate heap pressure or explicit GC calls."
        )

    if stats["gc_rate_per_min"] is not None and stats["gc_rate_per_min"] > 60:
        obs.append(
            f"[WARN] GC rate {stats['gc_rate_per_min']:.1f}/min is high. "
            "Consider increasing -Xmx or reducing object allocation rate."
        )

    after_vals = [e.heap_after_kb for e in events if e.heap_after_kb is not None]
    if len(after_vals) >= 10:
        mid = len(after_vals) // 2
        first_avg = statistics.mean(after_vals[:mid])
        second_avg = statistics.mean(after_vals[mid:])
        if second_avg > first_avg * 1.3:
            growth = (second_avg - first_avg) / first_avg * 100
            obs.append(
                f"[WARN] Post-GC heap baseline grew ~{growth:.0f}% from first to second half of log. "
                "Possible memory leak or sustained heap pressure."
            )

    if stats["pause_p95"] is not None and stats["pause_p95"] > 200:
        obs.append(
            f"[INFO] p95 pause is {stats['pause_p95']:.1f} ms. "
            "Latency-sensitive services should target p99 < 100 ms."
        )

    if stats["pause_p99"] is not None:
        obs.append(f"[INFO] p99 pause is {stats['pause_p99']:.1f} ms.")

    if not obs:
        obs.append("[OK]   No significant GC anomalies detected.")

    return obs


# ── Report ────────────────────────────────────────────────────────────────────

def _mb(kb: Optional[float]) -> str:
    return f"{kb / 1024:.1f} MB" if kb is not None else "N/A"

def _ms(v: Optional[float]) -> str:
    return f"{v:.3f} ms" if v is not None else "N/A"

def _pct(v: Optional[float]) -> str:
    return f"{v:.2f}%" if v is not None else "N/A"


def print_report(
    log_path: Path,
    events: List[GCEvent],
    stats: dict,
    unmatched: List[str],
    plot_paths: List[Path],
) -> None:
    W = 72

    def div(title: str) -> str:
        return f"\n  {title}\n  {'-' * (W - 2)}"

    print(f"\n{'=' * W}")
    print(f"  JVM GC Analysis  -  {log_path.name}")
    print(f"  {log_path.resolve()}")
    print("=" * W)

    print(div("OVERVIEW"))
    for k, v in [
        ("GC events parsed",    str(stats["total"])),
        ("Log duration",        f"{stats['duration_s']:.1f} s  ({stats['duration_s'] / 60:.2f} min)"),
        ("Total GC pause time", f"{stats['total_pause_s']:.3f} s"),
        ("App throughput",      _pct(stats["throughput_pct"])),
        ("GC rate",             f"{stats['gc_rate_per_min']:.2f} /min" if stats["gc_rate_per_min"] else "N/A"),
        ("Total heap reclaimed",_mb(stats["reclaimed_total_kb"])),
    ]:
        print(f"  {k:<24} {v}")

    print(div("GC EVENT BREAKDOWN"))
    for gc_type, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        pct = count / stats["total"] * 100
        print(f"  {gc_type:<12} {count:>5}  ({pct:>5.1f}%)  {bar}")

    if unmatched:
        print(f"\n  Unrecognized GC lines: {len(unmatched)}")
        for line in unmatched[:5]:
            print(f"    {line}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    print(div("PAUSE DURATION METRICS"))
    for k, v in [
        ("Minimum",  _ms(stats["pause_min"])),
        ("Mean",     _ms(stats["pause_mean"])),
        ("Median",   _ms(stats["pause_median"])),
        ("p95",      _ms(stats["pause_p95"])),
        ("p99",      _ms(stats["pause_p99"])),
        ("Maximum",  _ms(stats["pause_max"])),
    ]:
        print(f"  {k:<10} {v}")

    print(div("HEAP METRICS"))
    for k, v in [
        ("Before GC  max",       _mb(stats["heap_before_max_kb"])),
        ("Before GC  min",       _mb(stats["heap_before_min_kb"])),
        ("After GC   max",       _mb(stats["heap_after_max_kb"])),
        ("After GC   min",       _mb(stats["heap_after_min_kb"])),
        ("Reclaimed  mean/event",_mb(stats["reclaimed_mean_kb"])),
        ("Reclaimed  total",     _mb(stats["reclaimed_total_kb"])),
    ]:
        print(f"  {k:<26} {v}")

    print(div("OBSERVATIONS & ANOMALIES"))
    for o in build_observations(events, stats):
        print(f"  {o}")

    if plot_paths:
        print(div("VISUALIZATIONS"))
        for p in plot_paths:
            print(f"  Saved: {p}")
    elif not HAS_MPL:
        print("\n  Install matplotlib for heap charts:  pip install matplotlib")

    print(f"\n{'=' * W}\n")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(events: List[GCEvent], csv_path: str) -> None:
    fieldnames = [
        "line_no", "timestamp_s", "gc_type", "cause",
        "heap_before_kb", "heap_after_kb", "heap_total_kb", "pause_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in events:
            w.writerow({fn: getattr(e, fn) for fn in fieldnames})
    print(f"CSV written to: {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="JVM GC Log Analyzer — parse, summarize, and visualize GC logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  • JDK 8 -XX:+PrintGC / -XX:+PrintGCDetails / -XX:+PrintGCDateStamps
  • JDK 8 G1GC pause-style entries
  • JDK 9+ -Xlog:gc* unified logging
  • ZGC / Shenandoah concurrent entries
        """,
    )
    ap.add_argument("log_file", help="Path to the GC log file")
    ap.add_argument("--out-dir", default=".", metavar="DIR",
                    help="Directory for chart images (default: current dir)")
    ap.add_argument("--csv", metavar="FILE",
                    help="Export parsed events to a CSV file")
    args = ap.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_file():
        sys.exit(f"Error: file not found: {log_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {log_path} ...")
    events, unmatched = parse_log(log_path)

    if not events:
        print("No GC events could be parsed. Verify the file contains valid GC log output.")
        if unmatched:
            print("Lines that looked like GC events but did not match any known format:")
            for s in unmatched[:10]:
                print(f"  {s}")
        sys.exit(1)

    print(f"Found {len(events)} GC events. Generating report ...")

    stats = compute_stats(events)
    plot_paths = plot_heap(events, out_dir) if HAS_MPL else []

    if args.csv:
        export_csv(events, args.csv)

    print_report(log_path, events, stats, unmatched, plot_paths)


if __name__ == "__main__":
    main()
