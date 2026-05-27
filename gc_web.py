#!/usr/bin/env python3
"""
JVM GC Log Analyzer — Web Interface
Run:  python gc_web.py
Open: http://localhost:5000
"""

import io
import os
import base64
import tempfile
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify, Response
from gc_analyzer import parse_log, compute_stats, build_observations

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_last_analysis: dict = {}
_last_throughput: dict = {}
_last_heap: dict = {}
_last_stw_pauses_ms: list = []  # cached STW pauses for /rebucket without re-parsing


def _ms(v):  return f"{v:.2f} ms"       if v is not None else "N/A"
def _mb(kb): return f"{kb/1024:.1f} MB" if kb is not None else "N/A"
def _pct(v): return f"{v:.2f}%"         if v is not None else "N/A"


_TYPE_COLOR = {
    "Full": "#ef4444", "Young": "#3b82f6",
    "Mixed": "#8b5cf6", "Concurrent": "#10b981", "Unknown": "#94a3b8",
}


# ── Shared chart helpers ───────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color="#e2e8f0")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.3, color="#e2e8f0")


def _event_time(e) -> Optional[float]:
    """Return the best available time value for an event.
    Prefers wall_clock_s (correct across JVM restarts) over JVM uptime timestamp_s."""
    return e.wall_clock_s if e.wall_clock_s is not None else e.timestamp_s


_EPOCH_THRESHOLD = 1_000_000_000  # > 2001-09-09 in Unix seconds → treat as wall-clock


def _ts_to_dt(ts):
    """Convert epoch-second floats to local datetimes when they look like wall-clock
    timestamps; otherwise return the list unchanged (uptime in seconds)."""
    if ts and ts[0] > _EPOCH_THRESHOLD:
        return [datetime.fromtimestamp(t) for t in ts]
    return list(ts)


def _format_time_axis(ax, xs):
    """Configure the x-axis with readable date/time labels.

    `xs` should be the same values plotted on the x-axis: a list of datetimes
    when the log has wall-clock timestamps, or floats (seconds since JVM start)
    otherwise. Picks a tick cadence that matches the time span shown.
    """
    if not xs:
        return

    if isinstance(xs[0], datetime):
        span_s = (xs[-1] - xs[0]).total_seconds()
        if span_s <= 3600 * 6:
            interval = max(1, int(span_s / 3600 / 8) or 1)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif span_s <= 86400 * 2:
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:%M"))
        elif span_s <= 86400 * 14:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:%M"))
        else:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        span_s = xs[-1] - xs[0]

        def _fmt_uptime(s, _):
            s = int(s)
            if span_s < 60:
                return f"{s}s"
            if span_s < 3600:
                return f"{s // 60:02d}:{s % 60:02d}"
            if span_s < 86400:
                return f"{s // 3600}h{(s % 3600) // 60:02d}m"
            return f"{s // 86400}d{(s % 86400) // 3600:02d}h"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_uptime))

    ax.tick_params(axis="x", labelbottom=True, labelsize=9,
                   colors="#64748b", length=4, color="#cbd5e1")


def _timed_pauses(events):
    """Return sorted list of (time, pause_ms, gc_type) for events with both fields."""
    return sorted(
        [(_event_time(e), e.pause_ms, e.gc_type)
         for e in events if _event_time(e) is not None and e.pause_ms is not None],
        key=lambda x: x[0],
    )


def _time_buckets(timed, n=20):
    """Divide timed events into n equal-width time buckets. Returns (bucket_mid, pauses, width)."""
    if len(timed) < 2:
        return []
    t0, t1 = timed[0][0], timed[-1][0]
    if t1 <= t0:
        return []
    w = (t1 - t0) / n
    result = []
    for i in range(n):
        b0 = t0 + i * w
        b1 = b0 + w
        pauses = [p for t, p, _ in timed if b0 <= t < b1]
        result.append((t0 + (i + 0.5) * w, pauses, w))
    return result


# ── Heap chart observations ────────────────────────────────────────────────────

def _chart_obs(events, which: str) -> list:
    """
    Return a list of {"level": "OK"|"INFO"|"WARN", "text": str} dicts
    describing patterns specific to the heap-before or heap-after chart.
    """
    vals = [
        (e.heap_before_kb if which == "before" else e.heap_after_kb) / 1024
        for e in events
        if (e.heap_before_kb if which == "before" else e.heap_after_kb) is not None
        and _event_time(e) is not None
    ]

    if not vals:
        return [{"level": "INFO", "text": "No heap data available for this chart."}]

    obs = []
    mean_val = statistics.mean(vals)
    max_val  = max(vals)

    mid = len(vals) // 2
    if mid >= 2:
        first_avg  = statistics.mean(vals[:mid])
        second_avg = statistics.mean(vals[mid:])
        growth_pct = (second_avg - first_avg) / first_avg * 100 if first_avg else 0

        if which == "before":
            if growth_pct > 30:
                obs.append({"level": "WARN",
                    "text": f"Pre-GC heap peaks are trending upward by {growth_pct:.0f}% across the log. "
                            "The JVM is filling more heap before each collection over time, "
                            "suggesting increasing allocation pressure or a growing live set."})
            elif growth_pct > 10:
                obs.append({"level": "INFO",
                    "text": f"Pre-GC heap peaks show a moderate upward trend ({growth_pct:.0f}% increase). "
                            "This may reflect normal application warm-up — worth monitoring under sustained load."})
            else:
                obs.append({"level": "OK",
                    "text": f"Pre-GC heap peaks are stable across the log (trend: {growth_pct:+.0f}%). "
                            "Allocation pressure is consistent throughout the run."})
        else:
            if growth_pct > 25:
                obs.append({"level": "WARN",
                    "text": f"Post-GC heap baseline is rising by {growth_pct:.0f}% across the log. "
                            "GC cannot fully reclaim memory between cycles — "
                            "this is a strong indicator of a memory leak or a continuously growing live set."})
            elif growth_pct > 10:
                obs.append({"level": "WARN",
                    "text": f"Post-GC heap baseline shows a {growth_pct:.0f}% upward trend. "
                            "The live set appears to be growing steadily. "
                            "Investigate for long-lived object accumulation or caches without eviction."})
            elif growth_pct > 3:
                obs.append({"level": "INFO",
                    "text": f"Post-GC baseline is slightly higher in the second half ({growth_pct:.0f}% increase). "
                            "Could reflect normal cache warm-up; monitor over a longer run."})
            else:
                obs.append({"level": "OK",
                    "text": f"Post-GC heap baseline is stable throughout the log (trend: {growth_pct:+.0f}%). "
                            "No signs of a memory leak or growing live set."})

    if len(vals) >= 5:
        stdev = statistics.stdev(vals)
        cv    = stdev / mean_val * 100 if mean_val else 0

        if which == "before":
            if cv > 40:
                obs.append({"level": "WARN",
                    "text": f"High variability in pre-GC heap sizes (CV = {cv:.0f}%). "
                            "The heap fills to very different levels before each collection, "
                            "indicating bursty allocation or irregular GC triggering."})
            elif cv < 15:
                obs.append({"level": "OK",
                    "text": f"Pre-GC heap sizes are highly consistent (CV = {cv:.0f}%), "
                            "reflecting a regular and predictable allocation pattern."})
        else:
            if cv > 35:
                obs.append({"level": "WARN",
                    "text": f"Post-GC heap sizes vary widely (CV = {cv:.0f}%). "
                            "Some collections reclaim far more than others — "
                            "likely a mix of minor (Young) and major (Full) GC events "
                            "with very different reclamation amounts."})
            elif cv < 15:
                obs.append({"level": "OK",
                    "text": f"Post-GC heap levels are consistent (CV = {cv:.0f}%), "
                            "indicating predictable and uniform GC reclamation behavior."})

    if which == "after":
        ratios = [
            1 - e.heap_after_kb / e.heap_before_kb
            for e in events
            if e.heap_before_kb and e.heap_after_kb and e.heap_before_kb > 0
        ]
        if ratios:
            avg_reclaim = statistics.mean(ratios) * 100
            if avg_reclaim < 20:
                obs.append({"level": "WARN",
                    "text": f"GC is reclaiming only {avg_reclaim:.0f}% of heap on average. "
                            "Very low reclamation efficiency suggests a large live set, "
                            "premature GC triggering, or excessive object promotion to the old generation."})
            elif avg_reclaim >= 50:
                obs.append({"level": "OK",
                    "text": f"GC reclaims {avg_reclaim:.0f}% of heap per collection on average — "
                            "good collection efficiency."})
            else:
                obs.append({"level": "INFO",
                    "text": f"GC reclaims {avg_reclaim:.0f}% of heap per collection on average. "
                            "Efficiency is moderate; a larger heap may reduce GC frequency."})

    if which == "before":
        total_kbs = [e.heap_total_kb for e in events if e.heap_total_kb]
        if total_kbs:
            max_total_mb = max(total_kbs) / 1024
            pct_used     = max_val / max_total_mb * 100 if max_total_mb else 0
            if pct_used > 92:
                obs.append({"level": "WARN",
                    "text": f"Heap reaches {pct_used:.0f}% of maximum capacity "
                            f"({max_val:.0f} MB of {max_total_mb:.0f} MB) before GC fires. "
                            "The JVM is operating near the heap ceiling — "
                            "consider increasing -Xmx to reduce GC pressure."})
            elif pct_used > 75:
                obs.append({"level": "INFO",
                    "text": f"Pre-GC heap peaks at {pct_used:.0f}% of maximum capacity "
                            f"({max_val:.0f} MB of {max_total_mb:.0f} MB). "
                            "Adequate headroom for now, but monitor under higher load."})

    if which == "before":
        full_events  = [e for e in events if e.gc_type == "Full" and e.heap_before_kb]
        young_events = [e for e in events if e.gc_type == "Young" and e.heap_before_kb]
        if full_events:
            full_avg  = statistics.mean(e.heap_before_kb / 1024 for e in full_events)
            young_avg = statistics.mean(e.heap_before_kb / 1024 for e in young_events) if young_events else None
            if young_avg and full_avg < young_avg * 0.85:
                obs.append({"level": "INFO",
                    "text": f"Full GCs triggered at a lower heap level ({full_avg:.0f} MB) "
                            f"than Young GCs ({young_avg:.0f} MB on average). "
                            "This may indicate explicit System.gc() calls or promotion failures "
                            "rather than natural heap exhaustion."})

    return obs if obs else [{"level": "OK", "text": "No anomalies detected in this chart."}]


# ── Heap chart rendering ───────────────────────────────────────────────────────

def _heap_chart_b64(events, which: str) -> str:
    raw = [
        (_event_time(e), e.heap_before_kb, e.heap_after_kb, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.heap_before_kb and e.heap_after_kb
    ]
    if not raw:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.text(0.5, 0.5, "No heap data available", ha="center", va="center", transform=ax.transAxes)
        return _fig_to_b64(fig)

    raw.sort(key=lambda x: x[0])

    ts       = _ts_to_dt([r[0] for r in raw])
    ys       = [r[1] / 1024 if which == "before" else r[2] / 1024 for r in raw]
    gc_types = [r[3] for r in raw]

    full_xs = [ts[i] for i, t in enumerate(gc_types) if t == "Full"]
    full_ys = [ys[i] for i, t in enumerate(gc_types) if t == "Full"]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.fill_between(ts, ys, step="post", alpha=0.70, color="#2dd4bf", linewidth=0)
    ax.step(ts, ys, where="post", color="#0d9488", linewidth=1.2, zorder=3)

    if full_xs:
        ax.scatter(full_xs, full_ys, marker="^", color="#ef4444",
                   s=55, zorder=5, linewidths=0, label="Full GC")

    label = "after" if which == "after" else "before"
    ax.set_title(f"Heap Usage ({label} GC)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Heap size (mb)", fontsize=10, color="#64748b")

    data_max = max(ys) if ys else 0
    for tick_step in [50, 100, 200, 250, 500, 1000, 2000]:
        if data_max / tick_step <= 10:
            break
    else:
        tick_step = int(data_max / 8 / 100 + 1) * 100
    y_max = (int(data_max / tick_step) + 1) * tick_step
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(0, y_max + 1, tick_step))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=9, colors="#64748b")

    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.5, color="#e2e8f0", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.set_xlim(left=ts[0], right=ts[-1])
    _format_time_axis(ax, ts)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── GC Duration chart ─────────────────────────────────────────────────────────

def _gc_duration_chart_b64(events) -> str:
    data = [
        (_event_time(e), e.pause_ms, e.gc_type)
        for e in events
        if _event_time(e) is not None and e.pause_ms is not None
    ]
    if not data:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.text(0.5, 0.5, "No pause data available", ha="center", va="center", transform=ax.transAxes)
        return _fig_to_b64(fig)

    data.sort(key=lambda x: x[0])
    if data and data[0][0] > _EPOCH_THRESHOLD:
        data = [(datetime.fromtimestamp(t), p, gc) for t, p, gc in data]

    non_full = [(d[0], d[1]) for d in data if d[2] != "Full"]
    full     = [(d[0], d[1]) for d in data if d[2] == "Full"]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if non_full:
        ax.scatter([p[0] for p in non_full], [p[1] for p in non_full],
                   marker="s", color="#2dd4bf", s=40, zorder=3,
                   linewidths=0, label="Young GC")
    if full:
        ax.scatter([p[0] for p in full], [p[1] for p in full],
                   marker="^", color="#ef4444", s=60, zorder=4,
                   linewidths=0, label="Full GC")

    ax.set_title("GC Duration (pause time per event)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Pause duration (ms)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4, color="#cbd5e1")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    all_xs = [d[0] for d in data]
    ax.set_xlim(left=all_xs[0], right=all_xs[-1])
    ax.set_ylim(bottom=0)
    _format_time_axis(ax, all_xs)
    ax.legend(fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _gc_duration_chart_obs(events) -> list:
    pauses = [
        (_event_time(e), e.pause_ms, e.gc_type)
        for e in events
        if _event_time(e) is not None and e.pause_ms is not None
    ]
    if not pauses:
        return [{"level": "INFO", "text": "No pause duration data available."}]

    pauses.sort(key=lambda x: x[0])
    vals = [p for _, p, _ in pauses]
    obs  = []

    # Trend
    mid = len(vals) // 2
    if mid >= 2:
        first_avg  = statistics.mean(vals[:mid])
        second_avg = statistics.mean(vals[mid:])
        trend = (second_avg - first_avg) / first_avg * 100 if first_avg else 0
        if trend > 30:
            obs.append({"level": "WARN",
                "text": f"Pause durations increased by {trend:.0f}% from first to second half. "
                        "GC pauses are worsening — likely growing heap pressure or old-gen accumulation."})
        elif trend > 10:
            obs.append({"level": "INFO",
                "text": f"Pause durations show a modest upward trend ({trend:.0f}%). "
                        "Worth monitoring under extended runtime."})
        else:
            obs.append({"level": "OK",
                "text": f"Pause durations are stable across the log (trend: {trend:+.0f}%). "
                        "GC pause performance is consistent."})

    # Variability
    if len(vals) >= 5:
        cv = statistics.stdev(vals) / statistics.mean(vals) * 100 if statistics.mean(vals) else 0
        if cv > 50:
            obs.append({"level": "WARN",
                "text": f"High variability in pause durations (CV = {cv:.0f}%). "
                        "Some GC cycles are dramatically longer — likely a mix of Young and Full GC events."})
        elif cv < 20:
            obs.append({"level": "OK",
                "text": f"Pause durations are highly consistent (CV = {cv:.0f}%), "
                        "indicating predictable GC behaviour."})

    # Spike detection: any single pause > 3× median
    sp     = sorted(vals)
    median = statistics.median(vals)
    spikes = [p for p in vals if p > median * 3]
    if spikes:
        obs.append({"level": "WARN",
            "text": f"{len(spikes)} pause spike(s) detected exceeding 3× the median "
                    f"({median:.0f} ms). Peak spike: {max(spikes):.0f} ms. "
                    "Investigate for Full GC or promotion failure events."})

    # Full GC contribution
    full_pauses = [p for _, p, t in pauses if t == "Full"]
    if full_pauses:
        full_avg = statistics.mean(full_pauses)
        obs.append({"level": "WARN",
            "text": f"{len(full_pauses)} Full GC event(s) with average pause of {full_avg:.0f} ms "
                    "are visible in the chart (red triangles). "
                    "Full GCs are stop-the-world events — consider tuning heap sizing."})

    return obs if obs else [{"level": "OK", "text": "No anomalies detected in GC duration."}]


# ── Pause GC Duration chart (stop-the-world events only) ─────────────────────

def _pause_gc_chart_b64(events) -> str:
    data = [
        (_event_time(e), e.pause_ms, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.pause_ms is not None
        and e.gc_type not in ("Concurrent", "Unknown")
    ]
    if not data:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.text(0.5, 0.5, "No stop-the-world pause data available",
                ha="center", va="center", transform=ax.transAxes)
        return _fig_to_b64(fig)

    data.sort(key=lambda x: x[0])
    if data and data[0][0] > _EPOCH_THRESHOLD:
        data = [(datetime.fromtimestamp(t), p, gc) for t, p, gc in data]

    non_full = [(d[0], d[1]) for d in data if d[2] != "Full"]
    full     = [(d[0], d[1]) for d in data if d[2] == "Full"]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if non_full:
        ax.scatter([p[0] for p in non_full], [p[1] for p in non_full],
                   marker="s", color="#2dd4bf", s=40, zorder=3,
                   linewidths=0, label="Young GC")
    if full:
        ax.scatter([p[0] for p in full], [p[1] for p in full],
                   marker="^", color="#ef4444", s=60, zorder=4,
                   linewidths=0, label="Full GC")

    ax.set_title("Pause GC Duration (stop-the-world events)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Pause Duration (ms)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4, color="#cbd5e1")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    all_xs = [d[0] for d in data]
    ax.set_xlim(left=all_xs[0], right=all_xs[-1])
    ax.set_ylim(bottom=0)
    _format_time_axis(ax, all_xs)
    ax.legend(fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _pause_gc_chart_obs(events) -> list:
    pauses = [
        (_event_time(e), e.pause_ms, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.pause_ms is not None
        and e.gc_type not in ("Concurrent", "Unknown")
    ]
    if not pauses:
        return [{"level": "INFO", "text": "No stop-the-world pause data available."}]

    pauses.sort(key=lambda x: x[0])
    vals = [p for _, p, _ in pauses]
    obs  = []

    # Trend: first half vs second half
    mid = len(vals) // 2
    if mid >= 2:
        first_avg  = statistics.mean(vals[:mid])
        second_avg = statistics.mean(vals[mid:])
        trend = (second_avg - first_avg) / first_avg * 100 if first_avg else 0
        if trend > 30:
            obs.append({"level": "WARN",
                "text": f"Stop-the-world pauses increased {trend:.0f}% from first to second half "
                        f"(avg {first_avg:.0f} ms → {second_avg:.0f} ms). "
                        "Heap pressure is growing — review heap sizing or GC policy."})
        elif trend > 10:
            obs.append({"level": "INFO",
                "text": f"Modest upward trend in pause times ({trend:.0f}%). "
                        "Monitor over a longer run to confirm."})
        else:
            obs.append({"level": "OK",
                "text": f"Pause durations stable across the log (trend: {trend:+.0f}%). "
                        "Stop-the-world pause performance is consistent."})

    # Variability
    if len(vals) >= 5:
        mean_v = statistics.mean(vals)
        cv = statistics.stdev(vals) / mean_v * 100 if mean_v else 0
        if cv > 50:
            obs.append({"level": "WARN",
                "text": f"High variability in pause times (CV = {cv:.0f}%). "
                        "Erratic pauses often indicate mixed Young/Full GC workloads or promotion failures."})
        elif cv < 20:
            obs.append({"level": "OK",
                "text": f"Low pause variability (CV = {cv:.0f}%) — GC behaviour is predictable."})

    # Spike detection: >3× median
    median = statistics.median(vals)
    spikes = [p for p in vals if p > median * 3]
    if spikes:
        obs.append({"level": "WARN",
            "text": f"{len(spikes)} pause spike(s) exceeded 3× the median ({median:.0f} ms). "
                    f"Worst spike: {max(spikes):.0f} ms. "
                    "Investigate Full GC or heap exhaustion events."})

    # Full GC contribution
    full_pauses = [p for _, p, t in pauses if t == "Full"]
    if full_pauses:
        pct = len(full_pauses) / len(pauses) * 100
        obs.append({"level": "WARN",
            "text": f"{len(full_pauses)} Full GC event(s) account for {pct:.1f}% of stop-the-world pauses "
                    f"(avg {statistics.mean(full_pauses):.0f} ms, red triangles in chart). "
                    "Reducing Full GCs is the highest-impact tuning opportunity."})
    else:
        obs.append({"level": "OK",
            "text": "No Full GC events in stop-the-world pauses — only Young/Mixed GC pauses observed."})

    return obs if obs else [{"level": "OK", "text": "No anomalies detected in pause GC duration."}]


# ── Reclaimed Bytes chart ─────────────────────────────────────────────────────

def _reclaimed_chart_b64(events) -> str:
    data = [
        (_event_time(e), (e.heap_before_kb - e.heap_after_kb) / 1024, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.heap_before_kb is not None and e.heap_after_kb is not None
        and e.heap_before_kb >= e.heap_after_kb
    ]
    if not data:
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.text(0.5, 0.5, "No reclaimed data available", ha="center", va="center", transform=ax.transAxes)
        return _fig_to_b64(fig)

    data.sort(key=lambda x: x[0])
    if data and data[0][0] > _EPOCH_THRESHOLD:
        data = [(datetime.fromtimestamp(t), v, gc) for t, v, gc in data]

    non_full = [(d[0], d[1]) for d in data if d[2] != "Full"]
    full     = [(d[0], d[1]) for d in data if d[2] == "Full"]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if non_full:
        ax.scatter([p[0] for p in non_full], [p[1] for p in non_full],
                   marker="s", color="#2dd4bf", s=40, zorder=3,
                   linewidths=0, label="Young GC")
    if full:
        ax.scatter([p[0] for p in full], [p[1] for p in full],
                   marker="^", color="#ef4444", s=60, zorder=4,
                   linewidths=0, label="Full GC")

    ax.set_title("Reclaimed Bytes", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Reclaimed (mb)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4, color="#cbd5e1")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    all_xs = [d[0] for d in data]
    ax.set_xlim(left=all_xs[0], right=all_xs[-1])
    ax.set_ylim(bottom=0)
    _format_time_axis(ax, all_xs)
    ax.legend(fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _reclaimed_chart_obs(events) -> list:
    rows = [
        (_event_time(e), (e.heap_before_kb - e.heap_after_kb) / 1024, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.heap_before_kb is not None and e.heap_after_kb is not None
        and e.heap_before_kb >= e.heap_after_kb
    ]
    if not rows:
        return [{"level": "INFO", "text": "No reclaimed data available."}]

    rows.sort(key=lambda x: x[0])
    vals = [r[1] for r in rows]
    obs  = []

    mean_val = statistics.mean(vals)
    max_val  = max(vals)

    # Trend
    mid = len(vals) // 2
    if mid >= 2:
        first_avg  = statistics.mean(vals[:mid])
        second_avg = statistics.mean(vals[mid:])
        trend = (second_avg - first_avg) / first_avg * 100 if first_avg else 0
        if trend > 25:
            obs.append({"level": "INFO",
                "text": f"Reclaimed bytes per event increased by {trend:.0f}% from first to second half. "
                        "GC is collecting more memory over time — may indicate larger objects or "
                        "more accumulated garbage."})
        elif trend < -25:
            obs.append({"level": "WARN",
                "text": f"Reclaimed bytes per event dropped by {abs(trend):.0f}% from first to second half. "
                        "GC efficiency is declining — the heap may be filling with long-lived objects "
                        "that cannot be reclaimed."})
        else:
            obs.append({"level": "OK",
                "text": f"Reclaimed bytes per event are stable across the log (trend: {trend:+.0f}%). "
                        "Consistent GC reclamation behaviour."})

    # Average efficiency
    ratios = [
        1 - e.heap_after_kb / e.heap_before_kb
        for e in events
        if e.heap_before_kb and e.heap_after_kb and e.heap_before_kb > 0
    ]
    if ratios:
        avg_pct = statistics.mean(ratios) * 100
        if avg_pct < 20:
            obs.append({"level": "WARN",
                "text": f"Average reclamation efficiency is only {avg_pct:.0f}% per event. "
                        "Very little heap is freed per cycle — the live set may be large or "
                        "GC is triggering too early."})
        elif avg_pct >= 50:
            obs.append({"level": "OK",
                "text": f"Average reclamation efficiency is {avg_pct:.0f}% per event — "
                        "GC is freeing more than half the heap each cycle."})
        else:
            obs.append({"level": "INFO",
                "text": f"Average reclamation efficiency is {avg_pct:.0f}% per event. "
                        "Moderate; a larger heap may reduce GC frequency."})

    # Full GC vs Young GC reclamation
    full_vals  = [r[1] for r in rows if r[2] == "Full"]
    young_vals = [r[1] for r in rows if r[2] == "Young"]
    if full_vals and young_vals:
        full_avg  = statistics.mean(full_vals)
        young_avg = statistics.mean(young_vals)
        obs.append({"level": "INFO",
            "text": f"Full GC reclaims {full_avg:.0f} MB on average vs "
                    f"{young_avg:.0f} MB for Young GC events."})

    # Low-reclaim outliers
    low_threshold = mean_val * 0.1
    low_count = sum(1 for v in vals if v < low_threshold)
    if low_count:
        obs.append({"level": "INFO",
            "text": f"{low_count} event(s) reclaimed less than 10% of the average ({mean_val:.0f} MB). "
                    "These near-zero reclamation events may indicate premature GC triggering."})

    return obs if obs else [{"level": "OK", "text": "No anomalies detected in reclaimed bytes."}]


# ── Interactive (Plotly) data for the 4 time-series charts ────────────────────
# These return JSON-serialisable Plotly figure dicts so the frontend can render
# them with pan / zoom / reset controls. The static _b64 versions above are kept
# for the PDF report path, which embeds PNGs.

def _plotly_x_values(ts):
    """Format x values for Plotly: ISO strings for wall-clock, raw floats for uptime."""
    if not ts:
        return []
    if ts[0] > _EPOCH_THRESHOLD:
        return [datetime.fromtimestamp(t).isoformat() for t in ts]
    return list(ts)


def _plotly_base_layout(title: str, xs, ytitle: str = "") -> dict:
    """Common Plotly layout matching the existing matplotlib styling."""
    is_wall_clock = bool(xs) and isinstance(xs[0], str)  # ISO date strings
    xaxis = {
        "showgrid": False,
        "showline": True,
        "linecolor": "#e2e8f0",
        "ticks": "outside",
        "tickcolor": "#cbd5e1",
        "tickfont": {"size": 10, "color": "#64748b"},
        "automargin": True,
    }
    if is_wall_clock:
        xaxis["type"] = "date"

    return {
        "title": {
            "text": f"<b>{title}</b>",
            "font": {"size": 14, "color": "#1e293b"},
            "x": 0.5, "xanchor": "center",
        },
        "autosize": True,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "xaxis": xaxis,
        "yaxis": {
            "title": {"text": ytitle, "font": {"size": 10, "color": "#64748b"}},
            "showgrid": True,
            "gridcolor": "#e2e8f0",
            "gridwidth": 1,
            "zeroline": False,
            "rangemode": "tozero",
            "tickfont": {"size": 10, "color": "#64748b"},
            "tickformat": ",.0f",
            "automargin": True,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top", "y": -0.12,
            "xanchor": "center", "x": 0.5,
            "font": {"size": 10, "color": "#475569"},
        },
        "hovermode": "x unified",
        "height": 420,
    }


def _heap_chart_plotly(events, which: str) -> dict:
    raw = [
        (_event_time(e), e.heap_before_kb, e.heap_after_kb, e.gc_type)
        for e in events
        if _event_time(e) is not None
        and e.heap_before_kb and e.heap_after_kb
    ]
    if not raw:
        return {}
    raw.sort(key=lambda x: x[0])

    ts_raw   = [r[0] for r in raw]
    ys       = [r[1] / 1024 if which == "before" else r[2] / 1024 for r in raw]
    gc_types = [r[3] for r in raw]

    full_ts_raw = [ts_raw[i] for i, t in enumerate(gc_types) if t == "Full"]
    full_ys     = [ys[i]     for i, t in enumerate(gc_types) if t == "Full"]

    xs      = _plotly_x_values(ts_raw)
    full_xs = _plotly_x_values(full_ts_raw)

    data = [{
        "type": "scatter",
        "mode": "lines",
        "x": xs, "y": ys,
        "line": {"shape": "hv", "color": "#0d9488", "width": 1.2},
        "fill": "tozeroy",
        "fillcolor": "rgba(45, 212, 191, 0.55)",
        "name": "Heap",
        "hovertemplate": "%{x}<br>%{y:,.0f} MB<extra></extra>",
        "showlegend": False,
    }]
    if full_xs:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": full_xs, "y": full_ys,
            "marker": {"symbol": "triangle-up", "color": "#ef4444",
                       "size": 9, "line": {"width": 0}},
            "name": "Full GC",
            "hovertemplate": "Full GC<br>%{x}<br>%{y:,.0f} MB<extra></extra>",
        })

    label  = "after" if which == "after" else "before"
    layout = _plotly_base_layout(f"Heap Usage ({label} GC)", xs, ytitle="Heap size (MB)")
    return {"data": data, "layout": layout}


def _scatter_chart_plotly(events, title: str, ytitle: str, y_unit: str,
                          value_fn, filter_fn=None) -> dict:
    """Shared builder for the Young/Full-GC scatter charts
    (GC Duration, Pause GC Duration, Reclaimed Bytes)."""
    rows = []
    for e in events:
        t = _event_time(e)
        if t is None:
            continue
        if filter_fn is not None and not filter_fn(e):
            continue
        val = value_fn(e)
        if val is None:
            continue
        rows.append((t, val, e.gc_type))
    if not rows:
        return {}
    rows.sort(key=lambda r: r[0])

    non_full = [(t, v) for t, v, gc in rows if gc != "Full"]
    full     = [(t, v) for t, v, gc in rows if gc == "Full"]

    data = []
    if non_full:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": _plotly_x_values([r[0] for r in non_full]),
            "y": [r[1] for r in non_full],
            "marker": {"symbol": "square", "color": "#2dd4bf",
                       "size": 7, "line": {"width": 0}},
            "name": "Young GC",
            "hovertemplate": f"Young GC<br>%{{x}}<br>%{{y:,.0f}} {y_unit}<extra></extra>",
        })
    if full:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": _plotly_x_values([r[0] for r in full]),
            "y": [r[1] for r in full],
            "marker": {"symbol": "triangle-up", "color": "#ef4444",
                       "size": 10, "line": {"width": 0}},
            "name": "Full GC",
            "hovertemplate": f"Full GC<br>%{{x}}<br>%{{y:,.0f}} {y_unit}<extra></extra>",
        })

    xs_all = _plotly_x_values([r[0] for r in rows])
    layout = _plotly_base_layout(title, xs_all, ytitle=ytitle)
    return {"data": data, "layout": layout}


def _gc_duration_chart_plotly(events) -> dict:
    return _scatter_chart_plotly(
        events,
        title="GC Duration (pause time per event)",
        ytitle="Pause duration (ms)", y_unit="ms",
        value_fn=lambda e: e.pause_ms,
    )


def _pause_gc_chart_plotly(events) -> dict:
    return _scatter_chart_plotly(
        events,
        title="Pause GC Duration (stop-the-world events)",
        ytitle="Pause Duration (ms)", y_unit="ms",
        value_fn=lambda e: e.pause_ms,
        filter_fn=lambda e: e.gc_type not in ("Concurrent", "Unknown"),
    )


def _reclaimed_chart_plotly(events) -> dict:
    def _reclaimed(e):
        if (e.heap_before_kb is None or e.heap_after_kb is None
                or e.heap_before_kb < e.heap_after_kb):
            return None
        return (e.heap_before_kb - e.heap_after_kb) / 1024
    return _scatter_chart_plotly(
        events,
        title="Reclaimed Bytes",
        ytitle="Reclaimed (MB)", y_unit="MB",
        value_fn=_reclaimed,
    )


# ── KPI Chart: Throughput ──────────────────────────────────────────────────────

def _throughput_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if len(timed) < 2:
        return ""
    n = min(30, max(8, len(timed) // 2))
    buckets = _time_buckets(timed, n=n)
    if not buckets:
        return ""

    xs, ys = [], []
    for mid, pauses, w in buckets:
        wms = w * 1000
        tp = max(0.0, (wms - sum(pauses)) / wms * 100) if wms > 0 else 100.0
        xs.append(mid)
        ys.append(tp)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.fill_between(xs, ys, 100, alpha=0.15, color="#ef4444", label="GC overhead")
    ax.fill_between(xs, 0, ys, alpha=0.25, color="#10b981", label="App time")
    ax.plot(xs, ys, color="#059669", linewidth=2, zorder=4)
    ax.axhline(95, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.9,
               label="95% target", zorder=3)

    ax.set_title("Throughput — Application vs GC Time (%)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Throughput (%)", fontsize=10)
    ax.set_ylim(0, 102)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Chart: CPU Time ────────────────────────────────────────────────────────

def _cpu_time_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if len(timed) < 2:
        return ""

    ts     = [t for t, p, _ in timed]
    ps     = [p for _, p, _ in timed]
    cum_s  = [sum(ps[:i + 1]) / 1000 for i in range(len(ps))]
    colors = [_TYPE_COLOR.get(gc, "#94a3b8") for _, _, gc in timed]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.fill_between(ts, cum_s, alpha=0.30, color="#f97316")
    ax.plot(ts, cum_s, color="#ea580c", linewidth=2, zorder=4, label="Cumulative GC pause time")
    ax.scatter(ts, cum_s, c=colors, s=20, zorder=5, linewidths=0.4, edgecolors="#ffffff")

    ax.set_title("CPU Time — Cumulative GC Pause Time", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Cumulative GC Time (s)", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Chart: Latency ────────────────────────────────────────────────────────

def _latency_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if not timed:
        return ""

    ts     = [t for t, p, _ in timed]
    ps     = [p for _, p, _ in timed]
    colors = [_TYPE_COLOR.get(gc, "#94a3b8") for _, _, gc in timed]
    sp     = sorted(ps)
    n      = len(sp)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.scatter(ts, ps, c=colors, s=30, alpha=0.85, zorder=4, linewidths=0.5, edgecolors="#ffffff")

    p50 = sp[n // 2]
    ax.axhline(p50, color="#6366f1", linewidth=1.2, linestyle="--", alpha=0.85,
               label=f"p50: {p50:.1f} ms")
    if n >= 20:
        p95 = sp[int(n * 0.95)]
        ax.axhline(p95, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.85,
                   label=f"p95: {p95:.1f} ms")
    if n >= 100:
        p99 = sp[int(n * 0.99)]
        ax.axhline(p99, color="#ef4444", linewidth=1.2, linestyle="--", alpha=0.85,
                   label=f"p99: {p99:.1f} ms")

    ax.set_title("Latency — GC Pause Time per Event", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Pause Duration (ms)", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right", title="Percentiles", title_fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Chart: Average GC Pause Time ──────────────────────────────────────────

def _avg_pause_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if len(timed) < 3:
        return ""

    ts      = [t for t, p, _ in timed]
    ps      = [p for _, p, _ in timed]
    window  = min(10, max(2, len(ps) // 5))
    roll    = [statistics.mean(ps[max(0, i - window + 1):i + 1]) for i in range(len(ps))]
    mean_all = statistics.mean(ps)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.scatter(ts, ps, color="#cbd5e1", s=16, alpha=0.55, zorder=2, label="Individual events")
    ax.plot(ts, roll, color="#3b82f6", linewidth=2.2, zorder=4, label=f"Rolling avg (n={window})")
    ax.axhline(mean_all, color="#6366f1", linewidth=1.1, linestyle="--", alpha=0.8,
               label=f"Overall mean: {mean_all:.1f} ms", zorder=3)

    ax.set_title("Average GC Pause Time — Rolling Average", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Pause Duration (ms)", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Chart: Maximum GC Pause Time ──────────────────────────────────────────

def _max_pause_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if not timed:
        return ""

    ts     = [t for t, p, _ in timed]
    ps     = [p for _, p, _ in timed]
    colors = [_TYPE_COLOR.get(gc, "#94a3b8") for _, _, gc in timed]

    running_max, cur = [], 0
    for p in ps:
        cur = max(cur, p)
        running_max.append(cur)

    max_val = max(ps)
    max_idx = ps.index(max_val)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.scatter(ts, ps, c=colors, s=22, alpha=0.55, zorder=2, linewidths=0.4, edgecolors="#ffffff")
    ax.step(ts, running_max, color="#ef4444", linewidth=2, where="post", zorder=4,
            label="Running max")
    ax.scatter([ts[max_idx]], [ps[max_idx]], color="#ef4444", s=100, zorder=6,
               marker="*", label=f"Peak: {max_val:.1f} ms @ t={ts[max_idx]:.1f}s")

    ax.set_title("Maximum GC Pause Time — Running Maximum", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Pause Duration (ms)", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Chart: GC Duration Time Range ─────────────────────────────────────────

def _duration_range_chart_b64(events) -> str:
    timed = _timed_pauses(events)
    if len(timed) < 2:
        return ""

    n = min(20, max(5, len(timed) // 3))
    buckets = _time_buckets(timed, n=n)
    xs, mins, means, maxs = [], [], [], []
    for mid, pauses, _ in buckets:
        if pauses:
            xs.append(mid)
            mins.append(min(pauses))
            means.append(statistics.mean(pauses))
            maxs.append(max(pauses))

    if not xs:
        return ""

    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    _style_ax(ax)

    ax.fill_between(xs, mins, maxs, alpha=0.20, color="#8b5cf6", label="Min–Max range")
    ax.plot(xs, maxs, color="#8b5cf6", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(xs, mins, color="#8b5cf6", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(xs, means, color="#6d28d9", linewidth=2.2, zorder=4, label="Mean pause per window")

    ax.set_title("GC Duration Time Range — Min / Mean / Max per Window",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Pause Duration (ms)", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Summary: pause duration buckets + horizontal bar chart ────────────────

def _bucket_pauses(pauses_ms, bucket_size_ms):
    """Bucket pause durations (ms) into ranges of ``bucket_size_ms`` width.

    Returns a dict shaped for direct JSON serialisation:
        {"unit":    "sec" or "ms",         # display unit for the table header
         "size_ms": <bucket size in ms>,
         "rows":    [{"range": "0 - 1", "count": N, "pct": P}, ...]}

    The display unit is seconds when ``bucket_size_ms`` is a whole-second value
    (matches GCeasy's convention), milliseconds otherwise.
    """
    if not pauses_ms or bucket_size_ms <= 0:
        return {"unit": "sec", "size_ms": bucket_size_ms, "rows": []}

    if bucket_size_ms >= 1000 and bucket_size_ms % 1000 == 0:
        unit, divisor = "sec", 1000
    else:
        unit, divisor = "ms", 1

    num_buckets = max(1, int(max(pauses_ms) / bucket_size_ms) + 1)
    counts = [0] * num_buckets
    for p in pauses_ms:
        counts[min(int(p / bucket_size_ms), num_buckets - 1)] += 1
    total = len(pauses_ms)

    def _fmt(x):
        return f"{int(x)}" if x == int(x) else f"{x:g}"

    rows = []
    for i, c in enumerate(counts):
        lo = (i * bucket_size_ms) / divisor
        hi = ((i + 1) * bucket_size_ms) / divisor
        rows.append({
            "range": f"{_fmt(lo)} - {_fmt(hi)}",
            "count": c,
            "pct":   round(c / total * 100, 2) if total else 0.0,
        })
    return {"unit": unit, "size_ms": bucket_size_ms, "rows": rows}


def _pause_duration_buckets(events, bucket_size_ms: float = 1000):
    """Default bucket size is 1 second to match GCeasy's pause-duration table."""
    # Same rationale as compute_stats: STW pauses only — concurrent durations
    # are not application pauses and would distort the bucket distribution.
    pauses = [e.pause_ms for e in events
              if e.pause_ms is not None and e.gc_type != "Concurrent"]
    return _bucket_pauses(pauses, bucket_size_ms)


def _jvm_memory_chart_b64(events) -> str:
    allocated_kb  = max((e.heap_total_kb  for e in events if e.heap_total_kb),  default=None)
    peak_usage_kb = max((e.heap_before_kb for e in events if e.heap_before_kb), default=None)
    if not allocated_kb and not peak_usage_kb:
        return ""

    alloc_gb = (allocated_kb  or 0) / 1_048_576
    peak_gb  = (peak_usage_kb or 0) / 1_048_576
    max_val  = max(alloc_gb, peak_gb)

    BAR_COLOR = "#1e3a8a"
    labels = ["allocated", "peak usage"]
    values = [alloc_gb, peak_gb]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    bars = ax.barh(labels, values, color=BAR_COLOR, height=0.50, zorder=3)

    for bar, val in zip(bars, values):
        # value label centred inside the bar
        ax.text(
            bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} gb", va="center", ha="center",
            fontsize=12, color="#ffffff", fontweight="bold"
        )
        # exact value annotated just beyond the bar end
        ax.text(
            bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} GB", va="center", ha="left",
            fontsize=9, color="#475569"
        )

    ax.set_xlim(0, max_val * 1.22 or 0.1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(axis="x", labelsize=9, colors="#64748b")
    ax.tick_params(axis="y", labelsize=11, colors="#1e293b", pad=6)
    ax.set_xlabel("Size (GB)", fontsize=9, color="#64748b", labelpad=6)
    ax.set_title("JVM Memory Size — Allocated vs Peak", fontsize=12,
                 fontweight="bold", pad=10, color="#1e293b")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.invert_yaxis()

    # legend below axes
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BAR_COLOR)]
    ax.legend(handles, ["Heap size"], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1,
              fontsize=9, frameon=False, handlelength=1.2, handleheight=0.9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _kpi_summary_chart_b64(bucket_data) -> str:
    # bucket_data is the dict produced by _bucket_pauses (with "rows" + "unit")
    rows = bucket_data.get("rows", []) if isinstance(bucket_data, dict) else (bucket_data or [])
    if not rows:
        return ""
    unit = bucket_data.get("unit", "ms") if isinstance(bucket_data, dict) else "ms"
    labels = [f"{b['range']} {unit}" for b in rows]
    values = [b["pct"] for b in rows]

    fig_h = max(2.5, len(labels) * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, values, color="#1a3a6b", height=0.55)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", va="center", ha="left", fontsize=9, color="#1e293b")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Percentage (%)", fontsize=9)
    ax.set_title("GC Duration Time Range", fontsize=11, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax.set_xlim(0, (max(values) if values else 100) * 1.2)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, color="#e2e8f0")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── KPI Observations ──────────────────────────────────────────────────────────

def _kpi_obs(events, stats, which: str) -> list:
    timed = _timed_pauses(events)
    obs = []

    if which == "throughput":
        tp = stats.get("throughput_pct")
        if tp is None:
            return [{"level": "INFO", "text": "Insufficient data to compute throughput."}]
        if tp >= 99:
            obs.append({"level": "OK",
                "text": f"Throughput is excellent at {tp:.1f}%. GC overhead is negligible."})
        elif tp >= 95:
            obs.append({"level": "OK",
                "text": f"Throughput is {tp:.1f}%, meeting the ≥95% production target. "
                        "Application time dominates; GC is well-tuned."})
        else:
            obs.append({"level": "WARN",
                "text": f"Throughput is {tp:.1f}%, below the 95% target. "
                        "The JVM is spending too much time collecting garbage — "
                        "consider increasing heap size (-Xmx) or reviewing GC algorithm choice."})

        if len(timed) >= 4:
            bkts = _time_buckets(timed, n=min(20, max(5, len(timed) // 2)))
            window_tps = []
            for _, pauses, w in bkts:
                if pauses:
                    wms = w * 1000
                    window_tps.append(max(0.0, (wms - sum(pauses)) / wms * 100))
            if window_tps:
                min_tp = min(window_tps)
                if min_tp < 85:
                    obs.append({"level": "WARN",
                        "text": f"Worst single window throughput dropped to {min_tp:.1f}%. "
                                "Concentrated GC bursts are causing significant application pauses."})
                elif min_tp < 95:
                    obs.append({"level": "INFO",
                        "text": f"Throughput dipped to {min_tp:.1f}% in at least one window. "
                                "Brief GC bursts are present; overall trend is acceptable."})
                else:
                    obs.append({"level": "OK",
                        "text": f"Throughput remained above 95% in every time window. "
                                "No burst-GC periods detected."})

    elif which == "cpu_time":
        total_pause_s = stats.get("total_pause_s", 0)
        dur = stats.get("duration_s", 0)
        gc_pct = total_pause_s / dur * 100 if dur > 0 else 0
        if gc_pct > 10:
            obs.append({"level": "WARN",
                "text": f"GC consumed {total_pause_s:.2f}s of CPU time ({gc_pct:.1f}% of log duration). "
                        "Significant overhead — review heap sizing and GC algorithm choice."})
        elif gc_pct > 5:
            obs.append({"level": "INFO",
                "text": f"GC consumed {total_pause_s:.2f}s ({gc_pct:.1f}% of log duration). "
                        "Moderate overhead; monitor under peak load."})
        else:
            obs.append({"level": "OK",
                "text": f"GC CPU time is {total_pause_s:.2f}s ({gc_pct:.1f}% of log duration). "
                        "Low overhead — GC is not a significant CPU consumer."})

        if len(timed) >= 6:
            mid = len(timed) // 2
            t_first  = max(timed[mid - 1][0] - timed[0][0], 0.001)
            t_second = max(timed[-1][0] - timed[mid][0], 0.001)
            rate_first  = sum(p for _, p, _ in timed[:mid])  / t_first
            rate_second = sum(p for _, p, _ in timed[mid:]) / t_second
            if rate_second > rate_first * 1.5:
                obs.append({"level": "WARN",
                    "text": "GC pause rate is accelerating in the second half of the log — "
                            "CPU time consumed by GC is growing, indicating increasing memory pressure."})
            else:
                obs.append({"level": "OK",
                    "text": "GC CPU overhead is consistent across the log — no acceleration in pause rate."})

    elif which == "latency":
        pauses = [p for _, p, _ in timed]
        if not pauses:
            return [{"level": "INFO", "text": "No pause data available."}]
        sp    = sorted(pauses)
        n     = len(sp)
        max_p = sp[-1]
        p95   = sp[int(n * 0.95)] if n >= 20 else None

        if max_p > 1000:
            obs.append({"level": "WARN",
                "text": f"Maximum pause of {max_p:.0f} ms exceeds 1 second. "
                        "Latency-sensitive services (APIs, UIs) will experience significant hangs."})
        elif max_p > 500:
            obs.append({"level": "WARN",
                "text": f"Maximum pause of {max_p:.0f} ms exceeds 500 ms. "
                        "Consider ZGC or Shenandoah for lower-latency requirements."})
        else:
            obs.append({"level": "OK",
                "text": f"Maximum pause is {max_p:.1f} ms — no extreme latency spikes detected."})

        if p95 is not None:
            if p95 > 200:
                obs.append({"level": "WARN",
                    "text": f"p95 latency is {p95:.1f} ms — 95% of GC events pause the app for over 200 ms. "
                            "Review GC tuning for latency-sensitive workloads."})
            elif p95 > 100:
                obs.append({"level": "INFO",
                    "text": f"p95 latency is {p95:.1f} ms — acceptable for throughput-focused apps "
                            "but may impact latency-sensitive services."})
            else:
                obs.append({"level": "OK",
                    "text": f"p95 latency is {p95:.1f} ms — good for most workloads."})

        mean_p = statistics.mean(pauses)
        label  = "very low" if mean_p < 10 else "low" if mean_p < 50 else "moderate" if mean_p < 200 else "high"
        obs.append({"level": "OK" if mean_p < 100 else "INFO",
            "text": f"Mean pause time is {mean_p:.1f} ms ({label} average latency)."})

    elif which == "avg_pause":
        pauses = [p for _, p, _ in timed]
        if len(pauses) < 3:
            return [{"level": "INFO", "text": "Insufficient events for rolling average analysis."}]

        window   = min(10, max(2, len(pauses) // 5))
        roll     = [statistics.mean(pauses[max(0, i - window + 1):i + 1]) for i in range(len(pauses))]
        mean_all = statistics.mean(pauses)
        mid      = len(roll) // 2

        if mid >= 2:
            first_avg  = statistics.mean(roll[:mid])
            second_avg = statistics.mean(roll[mid:])
            trend      = (second_avg - first_avg) / first_avg * 100 if first_avg else 0
            if trend > 30:
                obs.append({"level": "WARN",
                    "text": f"Rolling average pause time increased by {trend:.0f}% from first to second half. "
                            "GC pauses are worsening — likely growing heap pressure or old-gen accumulation."})
            elif trend > 10:
                obs.append({"level": "INFO",
                    "text": f"Rolling average pause shows a modest upward trend ({trend:.0f}%). "
                            "Worth monitoring under extended runtime."})
            else:
                obs.append({"level": "OK",
                    "text": f"Rolling average pause time is stable (trend: {trend:+.0f}%). "
                            "GC pause performance is consistent throughout the log."})

        label = "low" if mean_all < 50 else "moderate" if mean_all < 200 else "high"
        obs.append({"level": "OK" if mean_all < 100 else "INFO",
            "text": f"Overall mean pause is {mean_all:.1f} ms ({label} average latency)."})

    elif which == "max_pause":
        pauses    = [p for _, p, _ in timed]
        ts        = [t for t, _, _ in timed]
        gc_types  = [gc for _, _, gc in timed]
        if not pauses:
            return [{"level": "INFO", "text": "No pause data available."}]

        max_val  = max(pauses)
        max_idx  = pauses.index(max_val)
        max_type = gc_types[max_idx]
        max_t    = ts[max_idx]
        level    = "WARN" if max_val > 500 else "INFO"
        obs.append({"level": level,
            "text": f"Worst event: {max_val:.1f} ms at t={max_t:.1f}s ({max_type} GC). "
                    + ("This is a significant stop-the-world pause requiring immediate attention."
                       if max_val > 500 else "Within acceptable range for most workloads.")})

        running_max, cur = [], 0
        for p in pauses:
            cur = max(cur, p)
            running_max.append(cur)
        threshold = max_val * 0.9
        idx_90    = next((i for i, v in enumerate(running_max) if v >= threshold), len(pauses) - 1)
        pct_through = idx_90 / len(pauses) * 100
        if pct_through < 25:
            obs.append({"level": "OK",
                "text": f"The running maximum stabilized within the first {pct_through:.0f}% of events. "
                        "No new worst-case outliers emerged later in the log."})
        else:
            obs.append({"level": "INFO",
                "text": f"The running maximum was still climbing at the {pct_through:.0f}% mark — "
                        "new pause spikes continued to appear throughout the log."})

    elif which == "duration_range":
        pauses = [p for _, p, _ in timed]
        if not pauses:
            return [{"level": "INFO", "text": "No duration data available."}]

        total_range = max(pauses) - min(pauses)
        cv = (statistics.stdev(pauses) / statistics.mean(pauses) * 100
              if len(pauses) >= 2 and statistics.mean(pauses) > 0 else 0)

        if total_range > 1000:
            obs.append({"level": "WARN",
                "text": f"GC pause duration spans {min(pauses):.0f}–{max(pauses):.0f} ms — "
                        "wide range suggests a mix of short Young GCs and long stop-the-world Full GCs."})
        elif total_range > 200:
            obs.append({"level": "INFO",
                "text": f"GC duration ranges from {min(pauses):.0f} to {max(pauses):.0f} ms — "
                        "moderate variability, likely from mixed GC types."})
        else:
            obs.append({"level": "OK",
                "text": f"GC duration range is narrow: {min(pauses):.0f}–{max(pauses):.0f} ms. "
                        "Consistent, predictable pause behavior."})

        if cv > 50:
            obs.append({"level": "WARN",
                "text": f"High coefficient of variation in pause durations (CV={cv:.0f}%). "
                        "Some GC cycles are dramatically longer than others — "
                        "investigate for Full GCs or promotion failures."})
        elif cv < 20:
            obs.append({"level": "OK",
                "text": f"Low variability in pause durations (CV={cv:.0f}%) — GC behavior is highly predictable."})
        else:
            obs.append({"level": "INFO",
                "text": f"Moderate variability in pause durations (CV={cv:.0f}%) across time windows."})

    return obs if obs else [{"level": "OK", "text": "No anomalies detected."}]


# ── KPI Summary Values (text, no chart) ───────────────────────────────────────

def _kpi_values(events, stats) -> dict:
    timed  = _timed_pauses(events)
    pauses = [p for _, p, _ in timed]

    def _pv(v): return f"{v:.2f} ms" if v is not None else "N/A"

    dur = stats.get("duration_s", 0) or 1
    tp  = stats.get("total_pause_s", 0)

    throughput_vals = [
        {"label": "App Throughput",  "value": _pct(stats.get("throughput_pct"))},
        {"label": "Total GC Pause",  "value": f"{tp:.3f} s"},
        {"label": "Log Duration",    "value": f"{stats.get('duration_s', 0):.1f} s"},
        {"label": "GC Rate",         "value": f"{stats['gc_rate_per_min']:.1f} /min" if stats.get("gc_rate_per_min") else "N/A"},
    ]

    cpu_vals = [
        {"label": "Total GC CPU Time", "value": f"{tp:.3f} s"},
        {"label": "GC CPU Overhead",   "value": f"{tp / dur * 100:.2f}%"},
        {"label": "Total Events",      "value": str(stats.get("total", 0))},
        {"label": "GC Rate",           "value": f"{stats['gc_rate_per_min']:.1f} /min" if stats.get("gc_rate_per_min") else "N/A"},
    ]

    latency_vals = [
        {"label": "Min",    "value": _pv(stats.get("pause_min"))},
        {"label": "Mean",   "value": _pv(stats.get("pause_mean"))},
        {"label": "Median", "value": _pv(stats.get("pause_median"))},
        {"label": "p95",    "value": _pv(stats.get("pause_p95"))},
        {"label": "p99",    "value": _pv(stats.get("pause_p99"))},
        {"label": "Max",    "value": _pv(stats.get("pause_max"))},
    ]

    stdev_str = f"{statistics.stdev(pauses):.2f} ms" if len(pauses) >= 2 else "N/A"
    avg_vals = [
        {"label": "Mean Pause",   "value": _pv(stats.get("pause_mean"))},
        {"label": "Median Pause", "value": _pv(stats.get("pause_median"))},
        {"label": "Std Dev",      "value": stdev_str},
        {"label": "Total Events", "value": str(stats.get("total", 0))},
    ]

    max_val  = stats.get("pause_max")
    max_type = max_ts = "N/A"
    if pauses:
        mi       = pauses.index(max(pauses))
        max_type = timed[mi][2]
        max_ts   = f"{timed[mi][0]:.1f} s"
    max_vals = [
        {"label": "Maximum Pause", "value": _pv(max_val)},
        {"label": "Occurred At",   "value": max_ts},
        {"label": "GC Type",       "value": max_type},
        {"label": "p99",           "value": _pv(stats.get("pause_p99"))},
        {"label": "p95",           "value": _pv(stats.get("pause_p95"))},
    ]

    return {
        "throughput": throughput_vals,
        "cpu_time":   cpu_vals,
        "latency":    latency_vals,
        "avg_pause":  avg_vals,
        "max_pause":  max_vals,
    }


# ── G1 GC Time ────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        return f"{h} hr {m} min {s} sec"
    if seconds >= 60:
        m  = int(seconds // 60)
        s  = int(seconds % 60)
        return f"{m} min {s} sec"
    s  = int(seconds)
    ms = int(round((seconds - s) * 1000))
    return f"{s} sec {ms} ms"


def _fmt_dhms(seconds: float) -> str:
    """Format a seconds value as "X days, Y hours, Z minutes, W.WW seconds",
    always showing all four units (used for the Log Duration overview card)."""
    days,    rem = divmod(seconds, 86400)
    hours,   rem = divmod(rem,      3600)
    minutes, rem = divmod(rem,        60)
    return (f"{int(days)} days, {int(hours)} hours, "
            f"{int(minutes)} minutes, {rem:.2f} seconds")


def _fmt_min_sec_ms(seconds: float) -> str:
    """Format a seconds value as 'X mins, Y sec, Z ms' — used for the CPU Time KPI card."""
    total_ms = int(round(seconds * 1000))
    m, rem = divmod(total_ms, 60_000)
    s, ms  = divmod(rem,        1000)
    return f"{m} mins, {s} sec, {ms} ms"


def _fmt_hms(seconds: float) -> str:
    """Format a seconds value as "Xhr Ymin Zsec Wms", dropping leading zero units."""
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem,    60_000)
    s, ms  = divmod(rem,    1000)
    parts = []
    if h: parts.append(f"{h} hr")
    if m or h: parts.append(f"{m} min")
    if s or m or h: parts.append(f"{s} sec")
    parts.append(f"{ms} ms")
    return " ".join(parts)


def _cpu_stats(events) -> dict:
    """Total user / sys / CPU (=user+sys) time consumed by GC, summed from each
    event's `[Times: ...]` (JDK 8) or `[gc,cpu] User=…s Sys=…s` (JDK 9+) line.

    Counts only primary collection pauses (Young / Mixed / Full) to match
    GCeasy's convention. G1's `Pause Remark` and `Pause Cleanup` phases are
    excluded because they are coordination phases of a concurrent cycle, not
    standalone collections — including them inflates the totals by ~10% on
    workloads with many concurrent cycles.

    Returns ``{"has_data": False, ...}`` with N/A strings when the log doesn't
    include CPU stats — neither JDK 8 nor JDK 9+ emit them by default, they
    only appear when -XX:+PrintGCDetails / -Xlog:gc+cpu is configured.
    """
    PRIMARY = {"Young", "Mixed", "Full"}
    users = [e.cpu_user_s for e in events
             if e.cpu_user_s is not None and e.gc_type in PRIMARY]
    syss  = [e.cpu_sys_s  for e in events
             if e.cpu_sys_s  is not None and e.gc_type in PRIMARY]

    if not users and not syss:
        return {
            "has_data":      False,
            "cpu_time":      "N/A",
            "user_time":     "N/A",
            "sys_time":      "N/A",
            "cpu_time_kpi":  "N/A",
        }

    user_total = sum(users)
    sys_total  = sum(syss)
    return {
        "has_data":      True,
        "cpu_time":      _fmt_hms(user_total + sys_total),
        "user_time":     _fmt_hms(user_total),
        "sys_time":      _fmt_hms(sys_total),
        "cpu_time_kpi":  _fmt_min_sec_ms(user_total + sys_total),
    }


def _g1_gc_time(events):
    pause_ms = [e.pause_ms for e in events
                if e.gc_type != "Concurrent" and e.pause_ms is not None]
    conc_ms  = [e.pause_ms for e in events
                if e.gc_type == "Concurrent" and e.pause_ms is not None]

    def _stats(vals):
        if not vals:
            return None
        return {
            "total_s": sum(vals) / 1000,
            "events":  len(vals),
            "avg_ms":  statistics.mean(vals),
            "std_ms":  statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "min_ms":  min(vals),
            "max_ms":  max(vals),
        }

    ps = _stats(pause_ms)
    cs = _stats(conc_ms)

    total_p = ps["total_s"] if ps else 0
    total_c = cs["total_s"] if cs else 0
    avg_p   = (ps["avg_ms"] / 1000) if ps else 0
    avg_c   = (cs["avg_ms"] / 1000) if cs else 0

    # ── Pie chart ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    fig1.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    sizes  = [total_p, total_c] if total_c else [total_p, 0.0001]
    colors = ["#0f2850", "#2dd4bf"]
    labels_pie = [f"{total_p:,.2f}", f"{total_c:,.2f}"]
    wedges, _ = ax1.pie(sizes, colors=colors, startangle=90,
                        wedgeprops={"linewidth": 0})
    for i, (wedge, lbl) in enumerate(zip(wedges, labels_pie)):
        ang   = (wedge.theta1 + wedge.theta2) / 2
        import math
        x = math.cos(math.radians(ang)) * 1.18
        y = math.sin(math.radians(ang)) * 1.18
        ax1.text(x, y, lbl, ha="center", va="center",
                 fontsize=8.5, color="#1e293b")
    ax1.set_title("Pause, concurrent Total Time (secs)",
                  fontsize=10, fontweight="bold", pad=10)
    ax1.legend(["Pause GC Time", "Concurrent GC Time"],
               loc="lower center", bbox_to_anchor=(0.5, -0.1),
               ncol=2, fontsize=8, frameon=False,
               handlelength=1.0, handleheight=0.8,
               markerscale=0.8)
    fig1.tight_layout()
    pie_b64 = _fig_to_b64(fig1)

    # ── Bar chart ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")
    bars = ax2.bar(["Pause Time", "Concurrent Time"], [avg_p, avg_c],
                   color=["#0f2850", "#2dd4bf"], width=0.45)
    for bar, val in zip(bars, [avg_p, avg_c]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(avg_p, avg_c) * 0.02,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=8.5, color="#1e293b")
    ax2.set_title("Pause, concurrent Avg Time (secs)",
                  fontsize=10, fontweight="bold", pad=10)
    ax2.set_ylabel("Seconds", fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.set_ylim(bottom=0, top=max(avg_p, avg_c) * 1.25 or 1)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax2.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.4, color="#cbd5e1")
    ax2.grid(axis="x", visible=False)
    fig2.tight_layout()
    bar_b64 = _fig_to_b64(fig2)

    def _fms(v):
        if v is None: return "N/A"
        if v >= 1000: return f"{v/1000:.3f} sec {int(v%1000)} ms"
        return f"{v:.1f} ms"

    pause_rows = [
        {"label": "Total GC Pause Time",  "value": _fmt_duration(total_p) if ps else "N/A"},
        {"label": "GC Pause Events",       "value": str(ps["events"]) if ps else "0"},
        {"label": "Avg GC Pause Time",     "value": _fms(ps["avg_ms"]) if ps else "N/A"},
        {"label": "Std Dev GC Pause Time", "value": _fms(ps["std_ms"]) if ps else "N/A"},
        {"label": "Min GC Pause Time",     "value": _fms(ps["min_ms"]) if ps else "N/A"},
        {"label": "Max GC Pause Time",     "value": _fms(ps["max_ms"]) if ps else "N/A"},
    ] if ps else []

    conc_rows = [
        {"label": "Total Concurrent GC Time",  "value": _fmt_duration(total_c) if cs else "N/A"},
        {"label": "Concurrent GC Events",       "value": str(cs["events"]) if cs else "0"},
        {"label": "Avg Concurrent GC Time",     "value": _fms(cs["avg_ms"]) if cs else "N/A"},
        {"label": "Std Dev Concurrent GC Time", "value": _fms(cs["std_ms"]) if cs else "N/A"},
        {"label": "Min Concurrent GC Time",     "value": _fms(cs["min_ms"]) if cs else "N/A"},
        {"label": "Max Concurrent GC Time",     "value": _fms(cs["max_ms"]) if cs else "N/A"},
    ] if cs else []

    return {
        "pie_chart":   pie_b64,
        "bar_chart":   bar_b64,
        "pause_rows":  pause_rows,
        "conc_rows":   conc_rows,
        "has_data":    bool(ps),
    }


# ── G1 Collection Phases Statistics ───────────────────────────────────────────

def _g1_collection_phases(events):
    import math

    def _phase(evts):
        pauses = [e.pause_ms for e in evts if e.pause_ms is not None]
        if not pauses:
            return None
        times = sorted(_event_time(e) for e in evts if _event_time(e) is not None)
        intervals = [(times[i + 1] - times[i]) * 1000 for i in range(len(times) - 1)]
        return {
            "total_s":     sum(pauses) / 1000,
            "avg_ms":      statistics.mean(pauses),
            "std_ms":      statistics.stdev(pauses) if len(pauses) >= 2 else 0.0,
            "min_ms":      min(pauses),
            "max_ms":      max(pauses),
            "interval_ms": statistics.mean(intervals) if intervals else None,
            "count":       len(pauses),
        }

    # Young GC summary follows GCeasy's convention here: it bundles every
    # `Pause Young (...)` event with the `Concurrent Cycle` events that those
    # pauses kick off (Concurrent Start → cycle). The cycle's wall-clock
    # duration is therefore included in the per-type "Young GC" total, even
    # though concurrent durations are *excluded* from the overall avg-pause
    # statistic (which measures actual STW impact on application latency).
    # Both conventions are kept deliberately so each metric answers its
    # intended question and the table matches the GCeasy reference output.
    young_evts   = [e for e in events if e.gc_type in ("Young", "Mixed", "Concurrent")]
    remark_evts  = [e for e in events if "remark"  in e.raw.lower() and e.pause_ms is not None]
    cleanup_evts = [e for e in events if "cleanup" in e.raw.lower() and e.pause_ms is not None
                    and "remark" not in e.raw.lower()]

    ys = _phase(young_evts)
    rs = _phase(remark_evts)
    cs = _phase(cleanup_evts)

    if not any([ys, rs, cs]):
        return {"has_data": False}

    C_YOUNG   = "#0f2850"
    C_REMARK  = "#2dd4bf"
    C_CLEANUP = "#38bdf8"

    # ── Horizontal bar chart — Avg Time (ms) ────────────────
    bar_phases, bar_vals, bar_cols = [], [], []
    for stat, lbl, col in [(cs, "Cleanup", C_CLEANUP), (rs, "Remark", C_REMARK), (ys, "Young GC", C_YOUNG)]:
        if stat:
            bar_phases.append(lbl); bar_vals.append(stat["avg_ms"]); bar_cols.append(col)

    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    fig1.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    bars = ax1.barh(bar_phases, bar_vals, color=bar_cols, height=0.45)
    max_v = max(bar_vals) if bar_vals else 1
    for bar, val in zip(bars, bar_vals):
        ax1.text(val + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8.5, color="#1e293b")
    ax1.set_title("Avg Time (ms)", fontsize=10, fontweight="bold", pad=8)
    ax1.set_xlim(0, max_v * 1.2)
    ax1.tick_params(labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color("#e2e8f0")
    ax1.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.4, color="#cbd5e1")
    ax1.grid(axis="y", visible=False)
    fig1.tight_layout()
    bar_b64 = _fig_to_b64(fig1)

    # ── Pie chart — Cumulative Time (Secs) ──────────────────
    pie_labels, pie_sizes, pie_cols = [], [], []
    for stat, lbl, col in [(ys, "Young GC", C_YOUNG), (rs, "Remark", C_REMARK), (cs, "Cleanup", C_CLEANUP)]:
        if stat and stat["total_s"] > 0:
            pie_labels.append(lbl); pie_sizes.append(stat["total_s"]); pie_cols.append(col)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")
    wedges, _ = ax2.pie(pie_sizes, colors=pie_cols, startangle=90,
                        wedgeprops={"linewidth": 0})
    for wedge, val in zip(wedges, pie_sizes):
        ang = (wedge.theta1 + wedge.theta2) / 2
        x = math.cos(math.radians(ang)) * 1.22
        y = math.sin(math.radians(ang)) * 1.22
        ax2.text(x, y, f"{val:,.2f}", ha="center", va="center",
                 fontsize=8, color="#1e293b")
    ax2.set_title("Cumulative Time (Secs)", fontsize=10, fontweight="bold", pad=8)
    ax2.legend(pie_labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=3, fontsize=8, frameon=False,
               handlelength=1.0, handleheight=0.8)
    fig2.tight_layout()
    pie_b64 = _fig_to_b64(fig2)

    def _fms(ms):
        if ms is None: return "n/a"
        if ms >= 60000:
            return f"{int(ms//60000)} min {int((ms%60000)//1000)} sec"
        if ms >= 1000:
            return f"{int(ms//1000)} sec {int(ms%1000)} ms"
        return f"{ms:.3f} ms"

    def _col(stat):
        if not stat:
            return {"total": "n/a", "avg": "n/a", "std": "n/a",
                    "min": "n/a", "max": "n/a", "interval": "n/a", "count": "n/a"}
        return {
            "total":    _fmt_duration(stat["total_s"]),
            "avg":      _fms(stat["avg_ms"]),
            "std":      _fms(stat["std_ms"]),
            "min":      _fms(stat["min_ms"]),
            "max":      _fms(stat["max_ms"]),
            "interval": _fms(stat["interval_ms"]),
            "count":    str(stat["count"]),
        }

    return {
        "has_data":  True,
        "bar_chart": bar_b64,
        "pie_chart": pie_b64,
        "young":     _col(ys),
        "remark":    _col(rs),
        "cleanup":   _col(cs),
    }


# ── Image-analysis PDF Report (shared between Throughput and Heap) ────────────

def _generate_throughput_pdf_report(data: dict) -> bytes:
    """Build a PDF report for a /analyze-throughput payload."""
    return _generate_image_analysis_pdf(
        data,
        header_title    = 'Throughput Image Analysis',
        header_subtitle = 'Management Console Review',
        footer_text     = 'JVM GC Log Analyzer — Throughput Image Report',
        section_label   = 'Throughput Graph',
    )


def _generate_heap_pdf_report(data: dict) -> bytes:
    """Build a PDF report for a /analyze-heap payload."""
    return _generate_image_analysis_pdf(
        data,
        header_title    = 'Heap Memory Analysis',
        header_subtitle = 'Management Console Review',
        footer_text     = 'JVM GC Log Analyzer — Heap Memory Report',
        section_label   = 'Heap Memory Graph',
    )


def _generate_image_analysis_pdf(data: dict, *, header_title: str, header_subtitle: str,
                                 footer_text: str, section_label: str) -> bytes:
    """Shared PDF generator for image-based analysis payloads (throughput, heap, ...)."""
    import io as _io
    import base64 as _b64
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether,
    )

    NAVY  = HexColor('#0f2850'); BLUE = HexColor('#2563eb')
    LGRAY = HexColor('#f1f5f9'); GRAY = HexColor('#e2e8f0')
    TEXT  = HexColor('#1e293b'); SOFT = HexColor('#64748b')
    WARN_BG  = HexColor('#fffbeb'); WARN_BDR = HexColor('#f59e0b')
    INFO_BG  = HexColor('#eff6ff'); INFO_BDR = HexColor('#3b82f6')
    OK_BG    = HexColor('#f0fdf4'); OK_BDR   = HexColor('#22c55e')
    TREND_COLOR = {
        'Stable':      HexColor('#2FAD68'),
        'Increasing':  HexColor('#1A81F4'),
        'Decreasing':  HexColor('#D64545'),
        'Fluctuating': HexColor('#FA7E2E'),
    }
    DEFAULT_TREND = HexColor('#FA7E2E')  # sudden drops/spikes

    PAGE_W, PAGE_H = letter
    MARGIN = 0.75 * inch
    CW     = PAGE_W - 2 * MARGIN

    def _s(name, **kw):
        s = ParagraphStyle(name, fontName='Helvetica', fontSize=9,
                           textColor=TEXT, leading=13, spaceAfter=2)
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    SH1   = _s('h1', fontSize=20, fontName='Helvetica-Bold', textColor=white, leading=24)
    SH2   = _s('h2', fontSize=13, fontName='Helvetica-Bold', textColor=NAVY,  spaceBefore=14, spaceAfter=8)
    SH3   = _s('h3', fontSize=10, fontName='Helvetica-Bold', textColor=BLUE,  spaceBefore=8,  spaceAfter=5)
    SMETA = _s('m',  fontSize=9,  textColor=SOFT)
    STAG  = _s('t',  fontSize=8,  fontName='Helvetica-Bold')
    SOBS  = _s('o',  fontSize=9,  leading=13)
    SBODY = _s('b',  fontSize=9,  leading=13)
    SSUB  = _s('ss', fontSize=10, textColor=HexColor('#93c5fd'), leading=13)
    SPILL = _s('sp', fontSize=8,  fontName='Helvetica-Bold', textColor=white, alignment=1, leading=10)

    buf = _io.BytesIO()

    def _on_page(c, doc):
        c.saveState()
        c.setFont('Helvetica', 7.5)
        c.setFillColor(SOFT)
        c.drawString(MARGIN, 0.4 * inch, footer_text)
        c.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f'Page {doc.page}')
        if doc.page > 1:
            c.setStrokeColor(GRAY); c.setLineWidth(0.5)
            c.line(MARGIN, PAGE_H - 0.4 * inch, PAGE_W - MARGIN, PAGE_H - 0.4 * inch)
        c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=0.65 * inch)

    def _img(uri, w=None, max_h=None):
        if not uri:
            return None
        try:
            raw   = uri.split(',', 1)[-1] if ',' in uri else uri
            b     = _b64.b64decode(raw)
            iw, ih = ImageReader(_io.BytesIO(b)).getSize()
            width  = w if w is not None else CW
            height = width * ih / iw
            if max_h is not None and height > max_h:
                height = max_h
                width  = height * iw / ih
            return RLImage(_io.BytesIO(b), width=width, height=height)
        except Exception:
            return None

    def _obs(level, text):
        if level == 'WARN':   bg, bdr, tc = WARN_BG, WARN_BDR, '#d97706'
        elif level == 'INFO': bg, bdr, tc = INFO_BG, INFO_BDR, '#2563eb'
        elif level == 'OK':   bg, bdr, tc = OK_BG,   OK_BDR,   '#16a34a'
        else:                 bg, bdr, tc = LGRAY,   GRAY,     '#64748b'
        row = [[Paragraph(f'<font color="{tc}"><b>[{level}]</b></font>', STAG),
                Paragraph(text, SOBS)]]
        t = Table(row, colWidths=[0.65 * inch, CW - 0.65 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), bg),
            ('LINEAFTER',     (0, 0), ( 0, -1), 2,   bdr),
            ('BOX',           (0, 0), (-1, -1), 0.4, bdr),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _kv_tbl(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',       (0, 0), (-1, -1), 9),
            ('TEXTCOLOR',      (0, 0), (-1,  0), NAVY),
            ('BACKGROUND',     (0, 0), (-1,  0), LGRAY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LGRAY]),
            ('GRID',           (0, 0), (-1, -1), 0.4, GRAY),
            ('TOPPADDING',     (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
            ('LEFTPADDING',    (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',   (0, 0), (-1, -1), 8),
            ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _trend_pill(trend):
        c = TREND_COLOR.get(trend, DEFAULT_TREND)
        # Compact pill: solid coloured cell with white bold text.
        label = trend if trend else 'Unknown'
        t = Table([[Paragraph(label, SPILL)]], colWidths=[2.0 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), c),
            ('BOX',           (0, 0), (-1, -1), 0.6, c),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _sp(n=8): return Spacer(1, n)
    def _hr():    return HRFlowable(width='100%', thickness=0.5, color=GRAY, spaceAfter=6, spaceBefore=2)

    story = []

    # Header banner
    hdr = Table(
        [[Paragraph(header_title, SH1),
          Paragraph(header_subtitle, SSUB)]],
        colWidths=[CW * 0.65, CW * 0.35],
    )
    hdr.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), NAVY),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',         (1, 0), ( 1,  0), 'RIGHT'),
        ('TOPPADDING',    (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
        ('LEFTPADDING',   (0, 0), (-1, -1), 18),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 18),
    ]))
    story += [hdr, _sp(10)]

    # Report metadata
    now      = datetime.now().strftime('%Y-%m-%d %H:%M')
    folder   = data.get('folder', '') or 'N/A'
    analyses = data.get('analyses', {}) or {}
    missing  = data.get('missing', []) or []

    period_order = ['30-day', '7-day', '24-hour']
    found_in_order = [p for p in period_order if p in analyses]

    meta = [
        [Paragraph('<b>Folder:</b>',            SMETA), Paragraph(folder, SMETA)],
        [Paragraph('<b>Generated:</b>',         SMETA), Paragraph(now, SMETA)],
        [Paragraph('<b>Periods analyzed:</b>',  SMETA),
         Paragraph(', '.join(found_in_order) or 'None', SMETA)],
    ]
    if missing:
        meta.append([Paragraph('<b>Missing files:</b>', SMETA),
                     Paragraph(', '.join(missing), SMETA)])

    meta_tbl = Table(meta, colWidths=[1.4 * inch, CW - 1.4 * inch])
    meta_tbl.setStyle(TableStyle([
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',     (0, 0), (-1, -1), SOFT),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    story += [meta_tbl, _sp(8)]

    # Per-period sections
    images_b64 = data.get('images', {}) or {}
    for label in found_in_order:
        a = analyses[label]
        m = a.get('metrics', {}) or {}
        trend = a.get('trend', '')

        # Section title row: heading + trend pill aligned right.
        title_row = Table(
            [[Paragraph(f'{label} {section_label}', SH2), _trend_pill(trend)]],
            colWidths=[CW - 2.0 * inch - 4, 2.0 * inch + 4],
        )
        title_row.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN',         (1, 0), ( 1,  0), 'RIGHT'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story += [title_row, _hr()]

        # Original screenshot (capped height so it doesn't overflow a page).
        img = _img(images_b64.get(label), w=CW, max_h=3.5 * inch)
        if img is not None:
            story += [img, _sp(8)]

        # Extracted metrics
        rows = [['Metric', 'Value']]
        rows.append(['Mean (normalized 0–100)',  f"{m.get('mean_norm', 'N/A')}"])
        rows.append(['Min / Max (normalized)',   f"{m.get('min_norm', 'N/A')} / {m.get('max_norm', 'N/A')}"])
        rows.append(['Variability (CoV)',        f"{m.get('cov', 'N/A')}"])
        rows.append(['Slope across window',      f"{m.get('slope_per_timeline', 'N/A')}"])
        rows.append(['Drops detected',           str(m.get('drops_detected', 'N/A'))])
        rows.append(['Spikes detected',          str(m.get('spikes_detected', 'N/A'))])
        rows.append(['Highest at',               f"{m.get('highest_position_pct', 'N/A')}% into window"])
        rows.append(['Lowest at',                f"{m.get('lowest_position_pct', 'N/A')}% into window"])
        rows.append(['Sampled columns',          str(m.get('samples', 'N/A'))])
        per = m.get('periodicity')
        if per:
            rows.append(['Periodicity',
                         f"period ≈ {per['period_fraction']*100:.0f}% "
                         f"(correlation {per['strength']:.2f})"])
        story += [Paragraph('Extracted Metrics', SH3),
                  _kv_tbl(rows, [CW * 0.45, CW * 0.55]),
                  _sp(8)]

        # Key observations
        obs_items = a.get('observations', []) or []
        if obs_items:
            story.append(Paragraph('Key Observations', SH3))
            for o in obs_items:
                story.append(_obs(o.get('level', 'INFO'), o.get('text', '')))
                story.append(_sp(3))

        # Technical interpretation
        interp = a.get('interpretation', []) or []
        if interp:
            story.append(Paragraph('Technical Interpretation', SH3))
            for t in interp:
                story.append(Paragraph(t, SBODY))
                story.append(_sp(3))

        # Recommendations
        recs = a.get('recommendations', []) or []
        if recs:
            story.append(Paragraph('Recommendations', SH3))
            for t in recs:
                story.append(Paragraph(f'•&nbsp; {t}', SBODY))
                story.append(_sp(3))

        story.append(_sp(8))

    # Cross-window comparison
    comparison = data.get('comparison', []) or []
    if comparison:
        title = ' vs '.join(found_in_order) + ' Comparison' if found_in_order else 'Cross-Window Comparison'
        story.append(Paragraph(title, SH2))
        for o in comparison:
            story.append(_obs(o.get('level', 'INFO'), o.get('text', '')))
            story.append(_sp(3))

    if not found_in_order:
        story.append(Paragraph('No images were analyzed.', SBODY))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.read()


# ── PDF Report ────────────────────────────────────────────────────────────────

def _generate_pdf_report(data: dict) -> bytes:
    import re as _re
    import io as _io
    import base64 as _b64
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )

    NAVY     = HexColor('#0f2850'); BLUE  = HexColor('#2563eb')
    LGRAY    = HexColor('#f1f5f9'); GRAY  = HexColor('#e2e8f0')
    TEXT     = HexColor('#1e293b'); SOFT  = HexColor('#64748b')
    WARN_BG  = HexColor('#fffbeb'); WARN_BDR = HexColor('#f59e0b')
    INFO_BG  = HexColor('#eff6ff'); INFO_BDR = HexColor('#3b82f6')
    OK_BG    = HexColor('#f0fdf4'); OK_BDR   = HexColor('#22c55e')

    PAGE_W, PAGE_H = letter
    MARGIN = 0.75 * inch
    CW     = PAGE_W - 2 * MARGIN
    HALF   = (CW - 12) / 2

    def _s(name, **kw):
        s = ParagraphStyle(name, fontName='Helvetica', fontSize=9,
                           textColor=TEXT, leading=13, spaceAfter=2)
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    SH1  = _s('sh1', fontSize=20, fontName='Helvetica-Bold', textColor=white, leading=24)
    SH2  = _s('sh2', fontSize=13, fontName='Helvetica-Bold', textColor=NAVY,  spaceBefore=14, spaceAfter=8)
    SH3  = _s('sh3', fontSize=10, fontName='Helvetica-Bold', textColor=BLUE,  spaceBefore=8,  spaceAfter=5)
    SMETA = _s('sm',  fontSize=9,  textColor=SOFT)
    STAG  = _s('st',  fontSize=8,  fontName='Helvetica-Bold')
    SOBS  = _s('so',  fontSize=9,  leading=13)
    SKVL  = _s('sl',  fontSize=7,  fontName='Helvetica-Bold', textColor=SOFT, leading=10)
    SKVV  = _s('sv',  fontSize=13, fontName='Helvetica-Bold', leading=16)
    SSUB  = _s('ss',  fontSize=10, textColor=HexColor('#93c5fd'), leading=13)

    buf = _io.BytesIO()

    def _on_page(c, doc):
        c.saveState()
        c.setFont('Helvetica', 7.5)
        c.setFillColor(SOFT)
        c.drawString(MARGIN, 0.4 * inch, 'JVM GC Log Analyzer — Performance Report')
        c.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f'Page {doc.page}')
        if doc.page > 1:
            c.setStrokeColor(GRAY); c.setLineWidth(0.5)
            c.line(MARGIN, PAGE_H - 0.4 * inch, PAGE_W - MARGIN, PAGE_H - 0.4 * inch)
        c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=0.65 * inch)

    def _img(uri, w=None):
        if not uri:
            return None
        try:
            raw   = uri.split(',', 1)[-1] if ',' in uri else uri
            b     = _b64.b64decode(raw)
            iw, ih = ImageReader(_io.BytesIO(b)).getSize()
            width  = w if w is not None else CW
            return RLImage(_io.BytesIO(b), width=width, height=width * ih / iw)
        except Exception:
            return None

    def _tbl(rows, widths=None, extras=None):
        style = [
            ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',       (0, 0), (-1, -1), 9),
            ('TEXTCOLOR',      (0, 0), (-1,  0), NAVY),
            ('BACKGROUND',     (0, 0), (-1,  0), LGRAY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LGRAY]),
            ('GRID',           (0, 0), (-1, -1), 0.4, GRAY),
            ('TOPPADDING',     (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
            ('LEFTPADDING',    (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',   (0, 0), (-1, -1), 8),
            ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ]
        if extras:
            style.extend(extras)
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle(style))
        return t

    def _obs(level, text):
        if level == 'WARN':   bg, bdr, tc = WARN_BG, WARN_BDR, '#d97706'
        elif level == 'INFO': bg, bdr, tc = INFO_BG, INFO_BDR, '#2563eb'
        elif level == 'OK':   bg, bdr, tc = OK_BG,   OK_BDR,   '#16a34a'
        else:                 bg, bdr, tc = LGRAY,    GRAY,     '#64748b'
        row = [[Paragraph(f'<font color="{tc}"><b>[{level}]</b></font>', STAG),
                Paragraph(text, SOBS)]]
        t = Table(row, colWidths=[0.65 * inch, CW - 0.65 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), bg),
            ('LINEAFTER',     (0, 0), ( 0, -1), 2,   bdr),
            ('BOX',           (0, 0), (-1, -1), 0.4, bdr),
            ('TOPPADDING',    (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _sec(text): return Paragraph(text, SH2)
    def _sub(text): return Paragraph(text, SH3)
    def _hr():      return HRFlowable(width='100%', thickness=0.5, color=GRAY, spaceAfter=6, spaceBefore=2)
    def _sp(n=8):   return Spacer(1, n)

    story = []

    # ── Header banner ──────────────────────────────────────────────────────────
    hdr = Table(
        [[Paragraph('JVM GC Log Analyzer', SH1), Paragraph('Performance Report', SSUB)]],
        colWidths=[CW * 0.65, CW * 0.35],
    )
    hdr.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), NAVY),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',         (1, 0), ( 1,  0), 'RIGHT'),
        ('TOPPADDING',    (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
        ('LEFTPADDING',   (0, 0), (-1, -1), 18),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 18),
    ]))
    story += [hdr, _sp(10)]

    # Report metadata
    ovw   = data.get('overview', [])
    _lkup = {o['label']: o['value'] for o in ovw}
    now   = datetime.now().strftime('%Y-%m-%d %H:%M')
    meta  = [
        [Paragraph('<b>File:</b>',      SMETA), Paragraph(data.get('filename', ''), SMETA)],
        [Paragraph('<b>Generated:</b>', SMETA), Paragraph(now, SMETA)],
    ]
    if 'Log Duration' in _lkup:
        meta.append([Paragraph('<b>Log Duration:</b>', SMETA), Paragraph(_lkup['Log Duration'], SMETA)])
    if 'GC Events Parsed' in _lkup:
        meta.append([Paragraph('<b>GC Events:</b>', SMETA), Paragraph(_lkup['GC Events Parsed'], SMETA)])
    mt = Table(meta, colWidths=[1.4 * inch, CW - 1.4 * inch])
    mt.setStyle(TableStyle([
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))
    story += [mt, _sp(10), _hr()]

    # ── Overview ───────────────────────────────────────────────────────────────
    if ovw:
        story.append(_sec('Overview'))
        cw3 = CW / 3
        def _cell(label, value):
            inner = Table(
                [[Paragraph(label.upper(), SKVL)], [Paragraph(str(value), SKVV)]],
                colWidths=[cw3 - 24],
            )
            inner.setStyle(TableStyle([
                ('LEFTPADDING',   (0, 0), (-1, -1), 0),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
                ('TOPPADDING',    (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ]))
            return inner
        ov_groups = [ovw[i:i+3] for i in range(0, len(ovw), 3)]
        ov_rows   = [[_cell(c['label'], c['value']) for c in g] +
                     [_cell('', '')] * (3 - len(g)) for g in ov_groups]
        ov_tbl = Table(ov_rows, colWidths=[cw3] * 3)
        ov_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), LGRAY),
            ('BOX',           (0, 0), (-1, -1), 0.5, GRAY),
            ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY),
            ('TOPPADDING',    (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ]))
        story += [ov_tbl, _sp(12)]

    unc = data.get('unmatched_count', 0)
    if unc:
        story += [Paragraph(f'Note: {unc} line(s) resembled GC events but matched no known format.', SMETA), _sp(6)]

    # ── GC Event Breakdown & Pause Duration ───────────────────────────────────
    by_type = data.get('by_type', [])
    p_rows  = data.get('pause_rows', [])
    if by_type or p_rows:
        story.append(_sec('GC Event Breakdown & Pause Duration'))
        parts = []
        if by_type:
            bd = [['Type', 'Count', 'Share']] + \
                 [[r['type'], str(r['count']), f"{r['pct']}%"] for r in by_type]
            parts.append(_tbl(bd, [HALF * 0.45, HALF * 0.25, HALF * 0.30]))
        if p_rows:
            pr = [['Metric', 'Value']] + [[r['label'], r['value']] for r in p_rows]
            parts.append(_tbl(pr, [HALF * 0.55, HALF * 0.45]))
        if len(parts) == 2:
            side = Table([[parts[0], '', parts[1]]], colWidths=[HALF, 12, HALF])
            side.setStyle(TableStyle([
                ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING',   (0, 0), (-1, -1), 0),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
                ('TOPPADDING',    (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            story.append(side)
        else:
            story.append(parts[0])
        story.append(_sp(12))

    # ── Heap Metrics ───────────────────────────────────────────────────────────
    h_rows = data.get('heap_rows', [])
    if h_rows:
        story.append(_sec('Heap Metrics'))
        hm = [['Metric', 'Value']] + [[r['label'], r['value']] for r in h_rows]
        story += [_tbl(hm, [CW * 0.6, CW * 0.4]), _sp(12)]

    # ── JVM Memory Allocation ──────────────────────────────────────────────────
    jvm_img = _img(data.get('jvm_memory_chart', ''), w=CW * 0.75)
    if jvm_img:
        story += [_sec('JVM Memory Allocation'), jvm_img, _sp(12)]

    # ── Observations ───────────────────────────────────────────────────────────
    obs_list = data.get('observations', [])
    if obs_list:
        story.append(_sec('Observations'))
        for obs in obs_list:
            m     = _re.match(r'^\[(WARN|INFO|OK)\]\s*', obs)
            level = m.group(1) if m else 'INFO'
            text  = obs[m.end():] if m else obs
            story += [_obs(level, text), _sp(3)]
        story.append(_sp(6))

    # ── KPI Dashboard ──────────────────────────────────────────────────────────
    kpi_values = data.get('kpi_values', {})
    if kpi_values:
        story.append(PageBreak())
        story.append(_sec('Key Performance Indicators'))
        KPI_ORDER = [
            ('throughput', 'Throughput'), ('cpu_time', 'CPU Time'),
            ('latency', 'Latency'),       ('avg_pause', 'Average Pause'),
            ('max_pause', 'Maximum Pause'),
        ]
        for key, label in KPI_ORDER:
            items = kpi_values.get(key, [])
            if not items:
                continue
            story.append(_sub(label))
            kv = [['Metric', 'Value']] + [[i.get('label', ''), i.get('value', 'N/A')] for i in items]
            story += [_tbl(kv, [CW * 0.6, CW * 0.4]), _sp(6)]
            for o in data.get('kpi_obs', {}).get(key, []):
                story += [_obs(o.get('level', 'INFO'), o.get('text', '')), _sp(3)]
            story.append(_sp(8))

        # duration_buckets is now {unit, size_ms, rows}; legacy list also handled.
        _bd  = data.get('duration_buckets', {})
        _rows = _bd.get('rows', []) if isinstance(_bd, dict) else (_bd or [])
        _unit = _bd.get('unit', 'sec') if isinstance(_bd, dict) else 'ms'
        buckets = [b for b in _rows if b.get('count', 0) > 0]
        if buckets:
            story.append(_sub('GC Pause Duration Time Range'))
            bk = [[f'Duration ({_unit})', 'No. of GCs', 'Percentage']] + \
                 [[b.get('range', ''), str(b.get('count', 0)), f"{b.get('pct', 0):.2f}%"]
                  for b in buckets]
            story += [_tbl(bk, [CW * 0.45, CW * 0.25, CW * 0.30]), _sp(8)]

        kpi_chart = _img(data.get('kpi_summary_chart', ''))
        if kpi_chart:
            story += [_sub('Duration Distribution Chart'), kpi_chart, _sp(8)]

    # ── Analysis Charts ────────────────────────────────────────────────────────
    chart_defs = [
        ('Heap Usage — After GC',       data.get('charts', {}).get('after',  ''), data.get('chart_obs', {}).get('after',  [])),
        ('Heap Usage — Before GC',      data.get('charts', {}).get('before', ''), data.get('chart_obs', {}).get('before', [])),
        ('GC Pause Duration',           data.get('gc_duration_chart',    ''),     data.get('gc_duration_chart_obs', [])),
        ('Stop-the-World Pauses Only',  data.get('pause_gc_chart',       ''),     data.get('pause_gc_chart_obs', [])),
        ('Memory Reclaimed per Event',  data.get('reclaimed_chart',      ''),     data.get('reclaimed_chart_obs', [])),
    ]
    if any(u for _, u, _ in chart_defs):
        story.append(PageBreak())
        story.append(_sec('Analysis Charts'))
        for title, uri, obs_items in chart_defs:
            if not uri:
                continue
            img = _img(uri)
            if not img:
                continue
            story += [_sub(title), img, _sp(4)]
            for o in obs_items:
                lvl = o.get('level', 'INFO') if isinstance(o, dict) else 'INFO'
                txt = o.get('text', '')       if isinstance(o, dict) else str(o)
                story += [_obs(lvl, txt), _sp(3)]
            story.append(_sp(14))

    # ── G1 Collection Phases ───────────────────────────────────────────────────
    g1p = data.get('g1_phases', {})
    if g1p and g1p.get('has_data'):
        story.append(PageBreak())
        story.append(_sec('G1 Collection Phases'))
        pb = _img(g1p.get('bar_chart', ''), w=CW * 0.58)
        pp = _img(g1p.get('pie_chart', ''), w=CW * 0.42 - 8)
        if pb and pp:
            pc = Table([[pb, pp]], colWidths=[CW * 0.58, CW * 0.42])
            pc.setStyle(TableStyle([
                ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING',  (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            story += [pc, _sp(10)]
        PHASES = [('young', 'Young GC'), ('remark', 'Remark'), ('cleanup', 'Cleanup')]
        METS   = [('Total Time', 'total'), ('Avg Time', 'avg'), ('Std Dev', 'std'),
                  ('Min', 'min'), ('Max', 'max'), ('Interval', 'interval'), ('Count', 'count')]
        if any(g1p.get(k) for k, _ in PHASES):
            ph = [['Metric'] + [lbl for _, lbl in PHASES]] + [
                [ml] + [(g1p.get(pk) or {}).get(mk, 'N/A') for pk, _ in PHASES]
                for ml, mk in METS
            ]
            cw4 = CW / (1 + len(PHASES))
            story += [_tbl(ph, [cw4] * (1 + len(PHASES))), _sp(12)]

    # ── G1 GC Time Analysis ────────────────────────────────────────────────────
    g1t = data.get('g1_time', {})
    if g1t and g1t.get('has_data'):
        story.append(PageBreak())
        story.append(_sec('G1 GC Time Analysis'))
        pie = _img(g1t.get('pie_chart', ''), w=HALF - 6)
        bar = _img(g1t.get('bar_chart', ''), w=HALF - 6)
        if pie and bar:
            ct = Table([[pie, bar]], colWidths=[HALF, HALF])
            ct.setStyle(TableStyle([
                ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING',  (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            story += [ct, _sp(10)]
        elif pie or bar:
            story += [(pie or bar), _sp(10)]

        g1pr = g1t.get('pause_rows', [])
        g1cr = g1t.get('conc_rows',  [])
        if g1pr and g1cr:
            pt  = _tbl([['Metric', 'Value']] + [[r['label'], r['value']] for r in g1pr], [HALF * 0.55, HALF * 0.45])
            ct_ = _tbl([['Metric', 'Value']] + [[r['label'], r['value']] for r in g1cr], [HALF * 0.55, HALF * 0.45])
            pair = Table(
                [[Paragraph('Pause Time', SH3),     '', Paragraph('Concurrent Time', SH3)],
                 [pt,                               '',  ct_]],
                colWidths=[HALF, 12, HALF],
            )
            pair.setStyle(TableStyle([
                ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING',   (0, 0), (-1, -1), 0),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
                ('TOPPADDING',    (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story += [pair, _sp(12)]
        elif g1pr:
            pt = _tbl([['Metric', 'Value']] + [[r['label'], r['value']] for r in g1pr], [CW * 0.5, CW * 0.5])
            story += [_sub('Pause Time'), pt, _sp(12)]
        elif g1cr:
            ct_ = _tbl([['Metric', 'Value']] + [[r['label'], r['value']] for r in g1cr], [CW * 0.5, CW * 0.5])
            story += [_sub('Concurrent Time'), ct_, _sp(12)]

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.read()


# ── Throughput image analysis ─────────────────────────────────────────────────

_THROUGHPUT_FILES = {
    "30-day":  "Throughput_30days.png",
    "7-day":   "Throughput_7days.png",
    "24-hour": "Throughput_24hours.png",
}


def _extract_line_series(image_path: Path):
    """
    Extract a normalized 0–100 throughput series from a Management Console graph PNG.

    Strategy:
      1. Crop a generous interior region to skip title, axes, legend.
      2. Identify "graph line" pixels by colour saturation (excludes white background,
         gridlines, axis labels, and most text).
      3. For each x-column, take the mean Y position of detected line pixels.
      4. Linearly interpolate over columns with no detections.
      5. Invert (image Y grows downward) and normalize to 0–100.

    Returns (series_np_array, width_px, height_px) or (None, 0, 0) if extraction failed.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape

    # Generous interior crop (works for typical web dashboard screenshots).
    top, bot   = int(h * 0.10), int(h * 0.88)
    left, right = int(w * 0.08), int(w * 0.98)
    if bot - top < 20 or right - left < 40:
        return None, 0, 0
    plot = arr[top:bot, left:right]
    ph, pw, _ = plot.shape

    r = plot[:, :, 0].astype(np.int16)
    g = plot[:, :, 1].astype(np.int16)
    b = plot[:, :, 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat  = maxc - minc  # cheap saturation proxy

    # Saturated coloured pixels — typical Management Console line/area styling.
    line_mask = (sat > 35) & (maxc < 245)

    # If colour-based detection found very little, fall back to dark non-white pixels
    # (covers black/dark-grey line styles).
    if line_mask.sum() < pw * 2:
        line_mask = (maxc < 170) & (sat <= 35)

    if line_mask.sum() < pw:
        return None, w, h

    ys_per_col = []
    for x in range(pw):
        ys = np.where(line_mask[:, x])[0]
        ys_per_col.append(float(ys.mean()) if ys.size else np.nan)
    ys_per_col = np.array(ys_per_col, dtype=float)

    valid = ~np.isnan(ys_per_col)
    if valid.sum() < pw * 0.3:
        return None, w, h
    xs = np.arange(pw)
    ys_per_col = np.interp(xs, xs[valid], ys_per_col[valid])

    # Invert so higher value = higher throughput, then normalize to 0–100 of
    # plot-area height. (Normalizing to data min/max would stretch tiny pixel
    # jitter from anti-aliasing into apparent large fluctuations.)
    raw  = ph - ys_per_col
    norm = np.clip(raw / ph * 100.0, 0.0, 100.0)
    return norm, w, h


def _edge_safe_moving_avg(v: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average that shrinks the window near the edges instead of
    zero-padding (which would produce phantom drops/spikes at the boundaries)."""
    n = v.size
    win = max(1, min(win, n))
    csum = np.cumsum(np.insert(v.astype(float), 0, 0.0))
    half = win // 2
    idx  = np.arange(n)
    los  = np.maximum(0, idx - half)
    his  = np.minimum(n, idx + half + 1)
    return (csum[his] - csum[los]) / (his - los)


def _classify_trend(values: np.ndarray):
    """Return (trend_label, metrics_dict)."""
    n     = int(values.size)
    mean  = float(values.mean())
    std   = float(values.std())
    cov   = std / mean if mean > 1e-6 else 0.0

    x     = np.linspace(0.0, 1.0, n)
    slope = float(np.polyfit(x, values, 1)[0])  # pts of normalized throughput across full window

    # Smooth aggressively, then look for window-over-window changes that exceed
    # a meaningful fraction of the data's value range. This avoids flagging
    # ordinary noise as drops/spikes the way a std-based threshold would.
    smooth_win = max(5, n // 30)
    smoothed   = _edge_safe_moving_avg(values, smooth_win)

    short_w  = max(3, n // 40)
    if short_w < smoothed.size:
        window_change = smoothed[short_w:] - smoothed[:-short_w]
    else:
        window_change = np.zeros(0)

    vrange    = float(values.max() - values.min())
    threshold = max(12.0, 0.18 * vrange)  # in normalized 0–100 points

    drop_idx, spike_idx = [], []
    i = 0
    while i < window_change.size:
        if window_change[i] < -threshold:
            drop_idx.append(int(i + short_w // 2))
            i += short_w
        elif window_change[i] > threshold:
            spike_idx.append(int(i + short_w // 2))
            i += short_w
        else:
            i += 1

    # Periodicity via autocorrelation.
    periodicity = None
    if n > 20:
        v0 = values - mean
        denom = float(np.dot(v0, v0))
        if denom > 0:
            ac = np.correlate(v0, v0, mode="full")[n - 1:] / denom
            search_start = max(3, n // 30)
            search_end   = max(search_start + 1, n // 2)
            window = ac[search_start:search_end]
            if window.size and window.max() > 0.4:
                lag = int(np.argmax(window)) + search_start
                periodicity = (lag / n, float(window.max()))

    if drop_idx or spike_idx:
        trend = "Showing sudden drops or spikes"
    elif slope > 5.0:
        trend = "Increasing"
    elif slope < -5.0:
        trend = "Decreasing"
    elif cov > 0.30:
        trend = "Fluctuating"
    else:
        trend = "Stable"

    metrics = {
        "samples":              n,
        "mean_norm":            round(mean, 1),
        "stdev_norm":           round(std, 1),
        "cov":                  round(cov, 3),
        "slope_per_timeline":   round(slope, 2),
        "drops_detected":       len(drop_idx),
        "spikes_detected":      len(spike_idx),
        "highest_position_pct": round(float(np.argmax(smoothed)) / n * 100, 1),
        "lowest_position_pct":  round(float(np.argmin(smoothed))  / n * 100, 1),
        "min_norm":             round(float(values.min()), 1),
        "max_norm":             round(float(values.max()), 1),
        "periodicity":          (
            {"period_fraction": round(periodicity[0], 3),
             "strength":        round(periodicity[1], 3)}
            if periodicity else None
        ),
    }
    return trend, metrics


def _throughput_observations(period_label, trend, metrics):
    """Build severity-tagged observation cards for one image."""
    obs = []
    obs.append({"level": "INFO",
                "text": f"Analyzing the {period_label} throughput graph "
                        f"({metrics['samples']} sampled columns)."})
    obs.append({"level": "INFO", "text": f"Overall throughput trend: {trend}."})

    high_pos = metrics["highest_position_pct"]
    low_pos  = metrics["lowest_position_pct"]
    obs.append({"level": "OK",
                "text": f"Highest throughput observed at approximately {high_pos:.0f}% "
                        f"into the {period_label} window."})

    low_level = "WARN" if metrics["min_norm"] < 25 else "INFO"
    obs.append({"level": low_level,
                "text": f"Lowest throughput observed at approximately {low_pos:.0f}% "
                        f"into the {period_label} window."})

    if metrics["drops_detected"]:
        n = metrics["drops_detected"]
        obs.append({"level": "WARN",
                    "text": f"{n} visible throughput drop{'s' if n != 1 else ''} detected — "
                            "investigate concurrent events (long GC pauses, deployments, dependency issues) "
                            "during these periods."})

    if metrics["spikes_detected"]:
        n = metrics["spikes_detected"]
        obs.append({"level": "INFO",
                    "text": f"{n} sudden throughput spike{'s' if n != 1 else ''} detected — "
                            "may indicate burst traffic or queued work being processed after a stall."})

    if metrics["periodicity"]:
        pf = metrics["periodicity"]["period_fraction"] * 100
        sg = metrics["periodicity"]["strength"]
        obs.append({"level": "INFO",
                    "text": f"Repeating pattern detected with approximate period ~{pf:.0f}% "
                            f"of the {period_label} timeline (correlation {sg:.2f}) — "
                            "likely a recurring workload cycle (scheduled jobs or traffic-of-day)."})

    cov = metrics["cov"]
    if cov > 0.35 and not metrics["periodicity"] and metrics["drops_detected"] == 0:
        obs.append({"level": "WARN",
                    "text": f"High throughput variability (coefficient of variation {cov:.2f}) "
                            "without a clear repeating pattern — workload may be unstable."})

    if (metrics["drops_detected"] == 0 and metrics["spikes_detected"] == 0
            and cov < 0.15 and abs(metrics["slope_per_timeline"]) < 5):
        obs.append({"level": "OK",
                    "text": "Throughput remained consistent across the window with no anomalies detected."})

    return obs


def _throughput_interpretation(trend, metrics):
    lines = []
    if trend == "Stable":
        lines.append("Stable throughput suggests the application is processing a consistent workload "
                     "without significant GC interference, memory pressure, or CPU saturation.")
        lines.append("From a throughput standpoint, the application appears healthy.")
    elif trend == "Increasing":
        lines.append("An upward throughput trend suggests improving processing capacity or growing "
                     "successful request handling.")
        lines.append("Typically positive — but confirm the increase is not a side effect of reduced load "
                     "(fewer requests served).")
    elif trend == "Decreasing":
        lines.append("A downward throughput trend often indicates degrading performance.")
        lines.append("Common causes: rising GC pause times, memory leaks, growing heap occupancy, "
                     "CPU saturation, thread-pool contention, or slower downstream dependencies.")
        lines.append("Further investigation is recommended.")
    elif trend == "Fluctuating":
        lines.append("Variable throughput without a clear direction may indicate inconsistent load patterns, "
                     "intermittent GC activity, or contention with other workloads on the host.")
        lines.append("Correlating with traffic volume and GC activity is recommended.")
    else:
        if metrics["drops_detected"]:
            lines.append("Visible drops typically correspond to long Stop-the-World GC pauses, brief outages, "
                         "deployments, or downstream-dependency failures.")
        if metrics["spikes_detected"]:
            lines.append("Sudden spikes can indicate post-GC throughput recovery, burst traffic, "
                         "or queued work being processed after a stall.")
        lines.append("These anomalies warrant correlation with the GC log timeline and infrastructure events.")

    if metrics["periodicity"]:
        lines.append("The repeating pattern suggests cyclical load — common drivers are scheduled batch jobs, "
                     "traffic-of-day patterns, or recurring background tasks.")

    return lines


def _throughput_recommendations(trend, metrics):
    recs = []
    if trend == "Decreasing" or metrics["drops_detected"]:
        recs.append("Correlate timing with the GC log: check for long Full GC pauses, "
                    "Mixed/Old generation activity, and rising heap occupancy.")
        recs.append("Review heap utilization (before/after GC) and reclaimed memory trends "
                    "from the GC analysis above.")
        recs.append("Check CPU utilization, thread-pool saturation, and request-queue depth "
                    "during the affected period.")

    if trend == "Fluctuating" or metrics["cov"] > 0.30:
        recs.append("Compare throughput variability with traffic volume — variability driven by load "
                    "is expected; variability under steady load is not.")
        recs.append("Review GC pause distribution (p95/p99) and verify pauses are within target SLO.")

    if metrics["spikes_detected"]:
        recs.append("Verify spikes are not artefacts of queued work being processed after a stall — "
                    "check for preceding throughput drops.")

    if metrics["periodicity"]:
        recs.append("Identify the source of the recurring cycle (scheduled jobs, traffic patterns) "
                    "and confirm it matches expected workload behaviour.")

    if trend == "Increasing":
        recs.append("Confirm the increase is genuine throughput gain and not a side effect of reduced load "
                    "(fewer incoming requests).")

    if not recs:
        recs.append("No immediate action required — continue monitoring throughput together with GC activity.")

    recs.append("Review throughput alongside: GC pause percentiles (p95/p99), heap-reclaim trends, "
                "CPU usage, memory-pool occupancy, and downstream-service latency.")
    return recs


_CONCERNING_TRENDS = ("Decreasing", "Showing sudden drops or spikes")


def _compare_periods(thirty, seven, twentyfour):
    """Cross-window comparison of 30-day, 7-day, and 24-hour analyses.

    Returns a small set of high-signal observations, in order:
      1. Trend coherence across all available windows
      2. Mean throughput trajectory (longest → shortest)
      3. Drop emergence in shorter windows
      4. Verdict on the most-recent window
    Periods that were not provided are skipped automatically, so this works
    with 2 or 3 windows.
    """
    ordered   = [("30-day", thirty), ("7-day", seven), ("24-hour", twentyfour)]
    available = [(label, a) for label, a in ordered if a]
    if len(available) < 2:
        return []

    items   = []
    labels  = [l for l, _ in available]
    long_label, long_a   = available[0]
    short_label, short_a = available[-1]

    # 1. Trend coherence
    trends_seen = [(l, a["trend"]) for l, a in available]
    if len({t for _, t in trends_seen}) == 1:
        the_trend = trends_seen[0][1]
        level     = "OK" if the_trend == "Stable" else "INFO"
        items.append({"level": level,
                      "text": f"All available views ({', '.join(labels)}) show the same overall trend "
                              f"({the_trend}) — behaviour is consistent across time horizons."})
    else:
        short_concerning = short_a["trend"] in _CONCERNING_TRENDS
        long_ok          = long_a["trend"]  not in _CONCERNING_TRENDS
        trend_summary    = ", ".join(f"{l}: {t}" for l, t in trends_seen)
        items.append({"level": "WARN" if (short_concerning and long_ok) else "INFO",
                      "text": f"Trends differ across windows ({trend_summary}) — "
                              "behaviour has changed at some point in the recent past, "
                              "with the shorter window reflecting the more recent state."})

    # 2. Mean throughput trajectory (longest → shortest)
    means        = [(l, a["metrics"]["mean_norm"]) for l, a in available]
    d_endpoints  = means[-1][1] - means[0][1]
    progressive  = (len(means) > 2 and (
        all(means[i + 1][1] >= means[i][1] - 2 for i in range(len(means) - 1)) or
        all(means[i + 1][1] <= means[i][1] + 2 for i in range(len(means) - 1))
    ))
    series_text  = ", ".join(f"{l}: {v:.0f}" for l, v in means)

    if abs(d_endpoints) <= 8:
        items.append({"level": "OK",
                      "text": f"Average throughput is consistent across windows ({series_text}) — "
                              "no significant short-term shift."})
    elif d_endpoints > 0:
        prog_note = " The improvement is progressive across windows." if progressive else ""
        items.append({"level": "INFO",
                      "text": f"Average throughput is higher in {short_label} than {long_label} "
                              f"({series_text}; {d_endpoints:+.0f} pts normalized) — "
                              f"recent improvement.{prog_note}"})
    else:
        prog_note = " The degradation is progressive across windows." if progressive else ""
        items.append({"level": "WARN",
                      "text": f"Average throughput is lower in {short_label} than {long_label} "
                              f"({series_text}; {d_endpoints:+.0f} pts normalized) — "
                              f"recent degradation.{prog_note}"})

    # 3. Drop emergence in shorter windows
    drops_short = short_a["metrics"]["drops_detected"]
    drops_long  = long_a["metrics"]["drops_detected"]
    if drops_short > drops_long and drops_short > 0:
        items.append({"level": "WARN",
                      "text": f"More throughput drops visible in the {short_label} view ({drops_short}) "
                              f"than the {long_label} view ({drops_long}) — short-term instability that "
                              "is not yet apparent in the longer window."})

    # 4. Verdict on the most-recent window
    recent_trend = short_a["trend"]
    if recent_trend == "Decreasing":
        items.append({"level": "WARN",
                      "text": f"Most recent view ({short_label}) is decreasing — investigate immediately. "
                              "Check GC activity, memory pressure, CPU, and downstream dependencies "
                              "for the corresponding time period."})
    elif recent_trend == "Showing sudden drops or spikes":
        items.append({"level": "WARN",
                      "text": f"Most recent view ({short_label}) shows sudden drops or spikes — "
                              "correlate the affected timestamps with GC log events and infrastructure changes."})
    elif recent_trend == "Fluctuating":
        items.append({"level": "INFO",
                      "text": f"Most recent view ({short_label}) is fluctuating — verify the workload "
                              "and traffic patterns are within their expected envelope."})
    else:
        items.append({"level": "OK",
                      "text": f"Most recent view ({short_label}) is {recent_trend.lower()} — "
                              "short-term health appears good."})

    return items


def _analyze_throughput_image(image_path: Path, period_label: str):
    """Full per-image pipeline. Returns dict with all sections, or None on extraction failure."""
    series, w, h = _extract_line_series(image_path)
    if series is None:
        return None
    trend, metrics = _classify_trend(series)
    return {
        "period_label":     period_label,
        "filename":         image_path.name,
        "image_width":      w,
        "image_height":     h,
        "trend":            trend,
        "metrics":          metrics,
        "observations":     _throughput_observations(period_label, trend, metrics),
        "interpretation":   _throughput_interpretation(trend, metrics),
        "recommendations":  _throughput_recommendations(trend, metrics),
    }


# ── Heap memory image analysis ────────────────────────────────────────────────
#
# Heap memory graphs are analyzed with the same pixel-extraction and statistical
# classification primitives as throughput graphs, but the semantic interpretation
# is different:
#   - Rising heap is the classic memory-leak signature        (WARN)
#   - "Drops" usually correspond to Major/Full GC reclaim     (INFO, not WARN)
#   - "Spikes" indicate burst allocations                     (INFO)
#   - Sawtooth / fluctuating pattern is normal                (INFO/OK)
#   - Stable at high level signals memory pressure            (WARN)

_HEAP_FILES = {
    "30-day":  "HeapMemory_30days.png",
    "7-day":   "HeapMemory_7days.png",
    "24-hour": "HeapMemory_24hours.png",
}

_HEAP_CONCERNING_TRENDS = ("Increasing",)
_HEAP_HIGH_LEVEL_THRESHOLD = 75.0  # normalized 0–100


def _heap_observations(period_label, trend, metrics):
    obs = []
    obs.append({"level": "INFO",
                "text": f"Analyzing the {period_label} heap memory graph "
                        f"({metrics['samples']} sampled columns)."})
    obs.append({"level": "INFO", "text": f"Overall heap memory trend: {trend}."})

    high_pos = metrics["highest_position_pct"]
    low_pos  = metrics["lowest_position_pct"]
    obs.append({"level": "INFO",
                "text": f"Highest heap usage observed at approximately {high_pos:.0f}% "
                        f"into the {period_label} window."})
    obs.append({"level": "OK",
                "text": f"Lowest heap usage observed at approximately {low_pos:.0f}% "
                        f"into the {period_label} window."})

    if metrics["drops_detected"]:
        n = metrics["drops_detected"]
        obs.append({"level": "INFO",
                    "text": f"{n} significant drop{'s' if n != 1 else ''} in heap usage detected — "
                            "typically corresponds to Major or Full GC reclaim events. "
                            "Correlate with the GC log to confirm."})

    if metrics["spikes_detected"]:
        n = metrics["spikes_detected"]
        obs.append({"level": "INFO",
                    "text": f"{n} sudden heap allocation spike{'s' if n != 1 else ''} detected — "
                            "indicates burst allocations or large-object creation."})

    if metrics["periodicity"]:
        pf = metrics["periodicity"]["period_fraction"] * 100
        sg = metrics["periodicity"]["strength"]
        obs.append({"level": "INFO",
                    "text": f"Repeating pattern detected with approximate period ~{pf:.0f}% "
                            f"of the {period_label} timeline (correlation {sg:.2f}) — "
                            "characteristic allocate/reclaim sawtooth, or a recurring workload cycle."})

    if trend == "Increasing":
        obs.append({"level": "WARN",
                    "text": "Heap usage is trending upward across the window — investigate for potential "
                            "memory leaks, growing caches, or insufficient old-generation reclaim."})
    elif trend == "Stable":
        if metrics["mean_norm"] > _HEAP_HIGH_LEVEL_THRESHOLD:
            obs.append({"level": "WARN",
                        "text": f"Heap is stable at a high level (mean {metrics['mean_norm']:.0f}/100 of the "
                                "visible range) — limited headroom for allocations or burst load."})
        else:
            obs.append({"level": "OK",
                        "text": "Heap is stable at a moderate level — GC appears to reclaim consistently "
                                "and the allocation rate is in balance with the reclaim rate."})

    cov = metrics["cov"]
    if cov > 0.35 and not metrics["periodicity"] and metrics["drops_detected"] == 0:
        obs.append({"level": "INFO",
                    "text": f"Heap usage varies considerably (coefficient of variation {cov:.2f}) "
                            "without a clear repeating pattern — workload allocation may be irregular."})

    return obs


def _heap_interpretation(trend, metrics):
    lines = []
    if trend == "Stable":
        if metrics["mean_norm"] > _HEAP_HIGH_LEVEL_THRESHOLD:
            lines.append("Stable heap usage at a high level usually indicates sustained memory pressure "
                         "with little headroom for new allocations.")
            lines.append("While the trend is steady, the JVM may be running close to its upper bound, "
                         "which can lead to longer GC pauses or OutOfMemoryError under load spikes.")
        else:
            lines.append("Stable heap at a moderate level usually means GC is reclaiming memory "
                         "consistently and the allocation rate is in balance with the reclaim rate.")
            lines.append("From a memory standpoint, the application appears healthy.")
    elif trend == "Increasing":
        lines.append("A rising heap-usage trend is the classic signature of a potential memory leak — "
                     "objects are being retained longer than expected, or unbounded data structures are "
                     "growing without being reclaimed.")
        lines.append("It can also indicate growing caches, accumulating session state, or insufficient "
                     "Full GC frequency to clear long-lived garbage.")
        lines.append("Further investigation is strongly recommended.")
    elif trend == "Decreasing":
        lines.append("A declining heap-usage trend is unusual but generally not concerning — it can "
                     "indicate a recent Major/Full GC reclaim, reduced workload, or improved GC tuning.")
        lines.append("Verify the decrease is not masking a real issue (e.g., reduced traffic hiding a "
                     "leak that would otherwise be visible).")
    elif trend == "Fluctuating":
        lines.append("Variable heap usage usually reflects the normal allocation/GC sawtooth pattern — "
                     "memory accumulates as objects are allocated, then drops sharply when GC reclaims.")
        lines.append("The amplitude and frequency of these swings are what matter: very large swings "
                     "point to large allocations; very frequent swings indicate high GC pressure.")
    else:
        if metrics["drops_detected"]:
            lines.append("Sudden drops in heap usage typically correspond to Major or Full GC events that "
                         "reclaimed a significant portion of the heap — this is expected behaviour, but "
                         "verify the frequency does not impact pause-time SLO.")
        if metrics["spikes_detected"]:
            lines.append("Sudden spikes in heap usage indicate burst allocations — large object creation, "
                         "request bursts, or batch jobs starting up.")
        lines.append("Correlate these events with the GC log to confirm cause and assess pause impact.")

    if metrics["periodicity"]:
        lines.append("The repeating pattern is characteristic of a steady allocate-and-reclaim cycle "
                     "or a recurring workload (scheduled jobs, traffic-of-day).")

    return lines


def _heap_recommendations(trend, metrics):
    recs = []
    if trend == "Increasing":
        recs.append("Take a heap dump for analysis — use a tool such as Eclipse MAT or VisualVM to identify "
                    "retained-size outliers and dominator hierarchies.")
        recs.append("Review the GC log: rising heap with infrequent Full GCs may indicate insufficient "
                    "old-generation pressure detection. Check Full GC frequency, duration, and reclaim ratio.")
        recs.append("Check common leak sources: unbounded static collections, ThreadLocal references not "
                    "cleaned, listener/callback registrations not removed, growing session or cache state.")

    if trend == "Stable" and metrics["mean_norm"] > _HEAP_HIGH_LEVEL_THRESHOLD:
        recs.append("Consider increasing the maximum heap (-Xmx) if the application consistently runs close "
                    "to its limit under normal load.")
        recs.append("Review object retention — a stable-but-high heap suggests significant long-lived state "
                    "that could potentially be reduced or moved off-heap.")

    if trend == "Fluctuating" or metrics["cov"] > 0.30:
        recs.append("Examine the amplitude of the swings: large drops indicate Major/Full GC events — "
                    "review the GC log for their frequency and pause-time impact.")

    if metrics["drops_detected"]:
        recs.append("Verify the timing of heap drops aligns with Full/Mixed GC entries in the GC log to "
                    "confirm they are reclaim events rather than data anomalies.")
        recs.append("If Major/Full GC frequency is high, consider tuning -XX:NewRatio, -Xmn, or migrating "
                    "to G1/ZGC for better old-generation reclaim characteristics.")

    if metrics["periodicity"]:
        recs.append("If the cycle period matches a known workload (batch job, traffic peaks), this is "
                    "expected. Otherwise, investigate what is driving the recurring allocation pattern.")

    if trend == "Decreasing":
        recs.append("Confirm reduced heap usage is genuine — not a side effect of significantly reduced "
                    "load that could hide an underlying retention issue.")

    if not recs:
        recs.append("No immediate action required — continue monitoring heap behaviour over a longer "
                    "window to confirm the trend.")

    recs.append("Review heap memory alongside: GC pause percentiles (p95/p99), Full GC frequency, "
                "reclaimed memory per event, allocation rate, and overall application throughput.")
    return recs


def _compare_heap_periods(thirty, seven, twentyfour):
    """Cross-window heap-memory comparison across 30-day, 7-day, and 24-hour analyses."""
    ordered   = [("30-day", thirty), ("7-day", seven), ("24-hour", twentyfour)]
    available = [(label, a) for label, a in ordered if a]
    if len(available) < 2:
        return []

    items   = []
    labels  = [l for l, _ in available]
    long_label, long_a   = available[0]
    short_label, short_a = available[-1]

    # 1. Trend coherence
    trends_seen = [(l, a["trend"]) for l, a in available]
    if len({t for _, t in trends_seen}) == 1:
        the_trend = trends_seen[0][1]
        if the_trend == "Stable":
            level = "OK"
        elif the_trend in _HEAP_CONCERNING_TRENDS:
            level = "WARN"
        else:
            level = "INFO"
        items.append({"level": level,
                      "text": f"All available views ({', '.join(labels)}) show the same overall heap trend "
                              f"({the_trend}) — behaviour is consistent across time horizons."})
    else:
        short_concerning = short_a["trend"] in _HEAP_CONCERNING_TRENDS
        long_ok          = long_a["trend"]  not in _HEAP_CONCERNING_TRENDS
        trend_summary    = ", ".join(f"{l}: {t}" for l, t in trends_seen)
        items.append({"level": "WARN" if (short_concerning and long_ok) else "INFO",
                      "text": f"Heap trends differ across windows ({trend_summary}) — behaviour has "
                              "changed in the recent past, with the shorter window reflecting the more "
                              "recent state."})

    # 2. Mean heap usage trajectory — rising = concerning, falling = relief.
    means        = [(l, a["metrics"]["mean_norm"]) for l, a in available]
    d_endpoints  = means[-1][1] - means[0][1]
    progressive  = (len(means) > 2 and (
        all(means[i + 1][1] >= means[i][1] - 2 for i in range(len(means) - 1)) or
        all(means[i + 1][1] <= means[i][1] + 2 for i in range(len(means) - 1))
    ))
    series_text  = ", ".join(f"{l}: {v:.0f}" for l, v in means)

    if abs(d_endpoints) <= 8:
        items.append({"level": "OK",
                      "text": f"Average heap usage is consistent across windows ({series_text}) — "
                              "no significant short-term shift in memory pressure."})
    elif d_endpoints > 0:
        prog_note = " The increase is progressive across windows, suggesting ongoing accumulation." if progressive else ""
        items.append({"level": "WARN",
                      "text": f"Average heap usage is higher in {short_label} than {long_label} "
                              f"({series_text}; {d_endpoints:+.0f} pts normalized) — increasing memory "
                              f"pressure that may indicate a memory leak or growing retained state.{prog_note}"})
    else:
        prog_note = " The decrease is progressive across windows." if progressive else ""
        items.append({"level": "INFO",
                      "text": f"Average heap usage is lower in {short_label} than {long_label} "
                              f"({series_text}; {d_endpoints:+.0f} pts normalized) — "
                              f"reduced memory pressure.{prog_note}"})

    # 3. Drop emergence — more drops in the shorter window means more recent Full/Major GC activity.
    drops_short = short_a["metrics"]["drops_detected"]
    drops_long  = long_a["metrics"]["drops_detected"]
    if drops_short > drops_long and drops_short > 0:
        items.append({"level": "INFO",
                      "text": f"More heap drops visible in the {short_label} view ({drops_short}) than the "
                              f"{long_label} view ({drops_long}) — likely increased Major/Full GC activity "
                              "in the recent period. Verify GC pause impact in the GC log."})

    # 4. Verdict on the most-recent window
    recent_trend = short_a["trend"]
    recent_mean  = short_a["metrics"]["mean_norm"]
    if recent_trend == "Increasing":
        items.append({"level": "WARN",
                      "text": f"Most recent view ({short_label}) shows increasing heap usage — investigate "
                              "immediately for potential memory leaks. Take a heap dump, review GC log for "
                              "Full GC frequency, and check for unbounded growth in caches, sessions, or "
                              "static collections."})
    elif recent_trend == "Stable" and recent_mean > _HEAP_HIGH_LEVEL_THRESHOLD:
        items.append({"level": "WARN",
                      "text": f"Most recent view ({short_label}) is stable at a high level "
                              f"(mean {recent_mean:.0f}/100) — limited headroom. Consider raising max heap "
                              "or investigating retention to free memory."})
    elif recent_trend == "Showing sudden drops or spikes":
        items.append({"level": "INFO",
                      "text": f"Most recent view ({short_label}) shows sudden drops or spikes — typically "
                              "indicates Major/Full GC events (drops) and burst allocations (spikes). "
                              "Correlate with GC log timestamps to confirm."})
    elif recent_trend == "Fluctuating":
        items.append({"level": "INFO",
                      "text": f"Most recent view ({short_label}) is fluctuating — normal allocation/reclaim "
                              "sawtooth pattern. Verify the swing amplitude is within expectations."})
    else:
        items.append({"level": "OK",
                      "text": f"Most recent view ({short_label}) is {recent_trend.lower()} — short-term "
                              "heap behaviour appears healthy."})

    return items


def _analyze_heap_image(image_path: Path, period_label: str):
    """Full per-image pipeline for heap memory. Returns dict, or None on extraction failure."""
    series, w, h = _extract_line_series(image_path)
    if series is None:
        return None
    trend, metrics = _classify_trend(series)
    return {
        "period_label":     period_label,
        "filename":         image_path.name,
        "image_width":      w,
        "image_height":     h,
        "trend":            trend,
        "metrics":          metrics,
        "observations":     _heap_observations(period_label, trend, metrics),
        "interpretation":   _heap_interpretation(trend, metrics),
        "recommendations":  _heap_recommendations(trend, metrics),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "gclog" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["gclog"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".log", delete=False) as tmp:
        f.save(tmp)
        tmp_path = tmp.name
    try:
        events, unmatched = parse_log(Path(tmp_path))
    finally:
        os.unlink(tmp_path)

    if not events:
        return jsonify({
            "error": "No GC events could be parsed. "
                     "Verify this is a valid JVM GC log (JDK 8 PrintGCDetails or JDK 9+ Xlog:gc*)."
        }), 400

    stats = compute_stats(events)
    obs   = build_observations(events, stats)

    has_data  = any(e.heap_before_kb and e.heap_after_kb for e in events)
    charts          = {}
    charts_plotly   = {}
    chart_obs       = {"before": [], "after": []}
    if has_data:
        charts["before"]        = _heap_chart_b64(events, "before")
        charts["after"]         = _heap_chart_b64(events, "after")
        charts_plotly["before"] = _heap_chart_plotly(events, "before")
        charts_plotly["after"]  = _heap_chart_plotly(events, "after")
        chart_obs["before"]     = _chart_obs(events, "before")
        chart_obs["after"]      = _chart_obs(events, "after")

    has_duration = any(e.pause_ms is not None for e in events)
    gc_duration_chart        = _gc_duration_chart_b64(events)    if has_duration else ""
    gc_duration_chart_plotly = _gc_duration_chart_plotly(events) if has_duration else None
    gc_duration_chart_obs    = _gc_duration_chart_obs(events)    if has_duration else []

    has_reclaimed = any(
        e.heap_before_kb and e.heap_after_kb and e.heap_before_kb >= e.heap_after_kb
        for e in events
    )
    reclaimed_chart        = _reclaimed_chart_b64(events)    if has_reclaimed else ""
    reclaimed_chart_plotly = _reclaimed_chart_plotly(events) if has_reclaimed else None
    reclaimed_chart_obs    = _reclaimed_chart_obs(events)    if has_reclaimed else []

    has_pause_gc = any(
        e.pause_ms is not None and e.gc_type not in ("Concurrent", "Unknown")
        for e in events
    )
    pause_gc_chart        = _pause_gc_chart_b64(events)    if has_pause_gc else ""
    pause_gc_chart_plotly = _pause_gc_chart_plotly(events) if has_pause_gc else None
    pause_gc_chart_obs    = _pause_gc_chart_obs(events)    if has_pause_gc else []

    jvm_memory_chart = _jvm_memory_chart_b64(events)
    g1_time   = _g1_gc_time(events)
    g1_phases = _g1_collection_phases(events)
    cpu_stats = _cpu_stats(events)

    all_kpi_keys     = ["throughput", "cpu_time", "latency", "avg_pause", "max_pause", "duration_range"]
    kpi_charts       = {"duration_range": _duration_range_chart_b64(events)}
    kpi_obs          = {k: _kpi_obs(events, stats, k) for k in all_kpi_keys}
    kpi_values       = _kpi_values(events, stats)
    duration_buckets = _pause_duration_buckets(events)  # default 1 sec — matches GCeasy
    kpi_summary_chart = _kpi_summary_chart_b64(duration_buckets)

    # Cache STW pauses so /rebucket can recompute without re-parsing the log.
    _last_stw_pauses_ms.clear()
    _last_stw_pauses_ms.extend(
        e.pause_ms for e in events
        if e.pause_ms is not None and e.gc_type != "Concurrent"
    )

    _payload = {
        "filename":   f.filename,
        "charts":     charts,
        "charts_plotly": charts_plotly,
        "chart_obs":  chart_obs,
        "gc_duration_chart":        gc_duration_chart,
        "gc_duration_chart_plotly": gc_duration_chart_plotly,
        "gc_duration_chart_obs":    gc_duration_chart_obs,
        "reclaimed_chart":        reclaimed_chart,
        "reclaimed_chart_plotly": reclaimed_chart_plotly,
        "reclaimed_chart_obs":    reclaimed_chart_obs,
        "pause_gc_chart":        pause_gc_chart,
        "pause_gc_chart_plotly": pause_gc_chart_plotly,
        "pause_gc_chart_obs":    pause_gc_chart_obs,
        "jvm_memory_chart":      jvm_memory_chart,
        "g1_time":               g1_time,
        "cpu_stats":             cpu_stats,
        "g1_phases":             g1_phases,
        "kpi_charts": kpi_charts,
        "kpi_obs":    kpi_obs,
        "kpi_values": kpi_values,
        "overview": [
            {"label": "GC Events Parsed", "value": str(stats["total"])},
            {"label": "Log Duration",     "value": _fmt_dhms(stats['duration_s'])},
            {"label": "Total Pause Time", "value": f"{stats['total_pause_s']:.3f} s"},
            {"label": "App Throughput",   "value": _pct(stats["throughput_pct"])},
            {"label": "GC Rate",          "value": f"{stats['gc_rate_per_min']:.1f} /min" if stats["gc_rate_per_min"] else "N/A"},
            {"label": "Total Reclaimed",  "value": _mb(stats["reclaimed_total_kb"])},
        ],
        "by_type": [
            {"type": k, "count": v, "pct": round(v / stats["total"] * 100, 1)}
            for k, v in sorted(stats["by_type"].items(), key=lambda x: -x[1])
        ],
        "pause_rows": [{"label": l, "value": v} for l, v in [
            ("Minimum",  _ms(stats["pause_min"])),
            ("Mean",     _ms(stats["pause_mean"])),
            ("Median",   _ms(stats["pause_median"])),
            ("p95",      _ms(stats["pause_p95"])),
            ("p99",      _ms(stats["pause_p99"])),
            ("Maximum",  _ms(stats["pause_max"])),
        ]],
        "heap_rows": [{"label": l, "value": v} for l, v in [
            ("Before GC  max",        _mb(stats["heap_before_max_kb"])),
            ("Before GC  min",        _mb(stats["heap_before_min_kb"])),
            ("After GC   max",        _mb(stats["heap_after_max_kb"])),
            ("After GC   min",        _mb(stats["heap_after_min_kb"])),
            ("Reclaimed  mean/event", _mb(stats["reclaimed_mean_kb"])),
            ("Reclaimed  total",      _mb(stats["reclaimed_total_kb"])),
        ]],
        "observations":     obs,
        "unmatched_count":  len(unmatched),
        "duration_buckets": duration_buckets,
        "kpi_summary_chart": kpi_summary_chart,
    }
    _last_analysis.clear()
    _last_analysis.update(_payload)
    return jsonify(_payload)


@app.route("/rebucket", methods=["POST"])
def rebucket():
    """Recompute the GC Pause Duration Time Range table + chart with a new
    bucket size, using the STW pauses cached from the last /analyze call."""
    if not _last_stw_pauses_ms:
        return jsonify({"error": "No analysis available. Run an analysis first."}), 400

    data = request.get_json(silent=True) or {}
    try:
        size_ms = float(data.get("bucket_ms", 1000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid bucket size."}), 400
    if size_ms <= 0:
        return jsonify({"error": "Bucket size must be greater than zero."}), 400

    bucket_data = _bucket_pauses(_last_stw_pauses_ms, size_ms)
    chart       = _kpi_summary_chart_b64(bucket_data)

    # Keep _last_analysis in sync so subsequent PDF downloads reflect the
    # bucket size the user last picked.
    if _last_analysis:
        _last_analysis["duration_buckets"]  = bucket_data
        _last_analysis["kpi_summary_chart"] = chart

    return jsonify({
        "duration_buckets":  bucket_data,
        "kpi_summary_chart": chart,
    })


@app.route("/analyze-throughput", methods=["POST"])
def analyze_throughput():
    data = request.get_json(silent=True) or {}
    folder_raw = (data.get("folder_path") or "").strip().strip('"').strip("'")
    if not folder_raw:
        return jsonify({"error": "No folder path provided."}), 400

    folder = Path(os.path.expandvars(os.path.expanduser(folder_raw)))
    if not folder.exists():
        return jsonify({"error": f"Folder does not exist: {folder}"}), 400
    if not folder.is_dir():
        return jsonify({"error": f"Path is not a directory: {folder}"}), 400

    expected = {label: folder / name for label, name in _THROUGHPUT_FILES.items()}
    found    = {label: p for label, p in expected.items() if p.is_file()}
    missing  = [p.name for label, p in expected.items() if not p.is_file()]

    if not found:
        return jsonify({
            "error": (
                "None of the expected throughput image files were found in this folder. "
                f"Expected: {', '.join(p.name for p in expected.values())}."
            )
        }), 400

    analyses = {}
    images   = {}
    for label, path in found.items():
        result = _analyze_throughput_image(path, label)
        if result is None:
            return jsonify({
                "error": (
                    f"Could not extract a throughput line from {path.name}. "
                    "The image may not be a recognised Management Console throughput graph, "
                    "or it has an unusual colour scheme."
                )
            }), 400
        analyses[label] = result
        with open(path, "rb") as fh:
            images[label] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()

    comparison = _compare_periods(
        analyses.get("30-day"),
        analyses.get("7-day"),
        analyses.get("24-hour"),
    )

    payload = {
        "folder":     str(folder),
        "analyses":   analyses,
        "images":     images,
        "comparison": comparison,
        "missing":    missing,
    }
    _last_throughput.clear()
    _last_throughput.update(payload)
    return jsonify(payload)


@app.route("/download-report")
def download_report():
    if not _last_analysis:
        return "No analysis available. Run an analysis first.", 400
    try:
        pdf_bytes = _generate_pdf_report(_last_analysis)
    except Exception as exc:
        return f"PDF generation failed: {exc}", 500
    fname = _last_analysis.get('filename', 'gc_report').rsplit('.', 1)[0]
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}_report.pdf"'},
    )


@app.route("/download-throughput-report")
def download_throughput_report():
    if not _last_throughput:
        return "No throughput analysis available. Run a throughput image analysis first.", 400
    try:
        pdf_bytes = _generate_throughput_pdf_report(_last_throughput)
    except Exception as exc:
        return f"PDF generation failed: {exc}", 500
    from datetime import datetime
    fname = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_Throughput_Image_Analysis.pdf"
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


@app.route("/analyze-heap", methods=["POST"])
def analyze_heap():
    data = request.get_json(silent=True) or {}
    folder_raw = (data.get("folder_path") or "").strip().strip('"').strip("'")
    if not folder_raw:
        return jsonify({"error": "No folder path provided."}), 400

    folder = Path(os.path.expandvars(os.path.expanduser(folder_raw)))
    if not folder.exists():
        return jsonify({"error": f"Folder does not exist: {folder}"}), 400
    if not folder.is_dir():
        return jsonify({"error": f"Path is not a directory: {folder}"}), 400

    expected = {label: folder / name for label, name in _HEAP_FILES.items()}
    found    = {label: p for label, p in expected.items() if p.is_file()}
    missing  = [p.name for label, p in expected.items() if not p.is_file()]

    if not found:
        return jsonify({
            "error": (
                "None of the expected heap memory image files were found in this folder. "
                f"Expected: {', '.join(p.name for p in expected.values())}."
            )
        }), 400

    analyses = {}
    images   = {}
    for label, path in found.items():
        result = _analyze_heap_image(path, label)
        if result is None:
            return jsonify({
                "error": (
                    f"Could not extract a heap memory line from {path.name}. "
                    "The image may not be a recognised Management Console heap memory graph, "
                    "or it has an unusual colour scheme."
                )
            }), 400
        analyses[label] = result
        with open(path, "rb") as fh:
            images[label] = "data:image/png;base64," + base64.b64encode(fh.read()).decode()

    comparison = _compare_heap_periods(
        analyses.get("30-day"),
        analyses.get("7-day"),
        analyses.get("24-hour"),
    )

    payload = {
        "folder":     str(folder),
        "analyses":   analyses,
        "images":     images,
        "comparison": comparison,
        "missing":    missing,
    }
    _last_heap.clear()
    _last_heap.update(payload)
    return jsonify(payload)


@app.route("/download-heap-report")
def download_heap_report():
    if not _last_heap:
        return "No heap memory analysis available. Run a heap memory image analysis first.", 400
    try:
        pdf_bytes = _generate_heap_pdf_report(_last_heap)
    except Exception as exc:
        return f"PDF generation failed: {exc}", 500
    from datetime import datetime
    fname = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_Heap_Memory_Analysis.pdf"
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


if __name__ == "__main__":
    print("Starting GC Analyzer at http://localhost:5000")
    app.run(debug=False, port=5000)
