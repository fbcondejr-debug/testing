# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JVM Garbage Collection (GC) Log Analyzer — a Python tool with two interfaces:
- **CLI** (`gc_analyzer.py`): batch parsing, CSV export, static PNG charts
- **Web** (`gc_web.py`): interactive Flask app with upload, charts, KPIs, and observations

## Running the Project

```bash
# Install dependencies
pip install -r requirements_gc.txt

# Web interface (runs on http://localhost:5000)
python gc_web.py

# CLI analysis
python gc_analyzer.py <log_file>

# Generate synthetic test logs
python generate_sample_gc_log.py
```

## Architecture

### Core modules

- **`gc_analyzer.py`** — parsing engine and CLI entry point. Defines the `GCEvent` dataclass, multi-format regex parsers (JDK 8 PrintGCDetails, JDK 9+ unified logging, G1GC, ZGC/Shenandoah), statistics computation, and matplotlib-based static chart export.

- **`gc_web.py`** — Flask app (1,600+ lines). Handles file upload, re-parses the log using the same regex logic as `gc_analyzer.py` (duplicated inline), computes extended KPIs (throughput, CPU time, latency percentiles p50/p95/p99, pause distribution buckets, heap reclaim trends), generates severity-tagged observations, and returns base64-encoded charts in the JSON response.

- **`templates/index.html`** — single-page UI. Drag-and-drop upload, tab-based results view, dynamic chart rendering from base64 PNG, observation cards with severity styling.

### Important design notes

- `gc_web.py` contains its own copy of the parsing logic rather than importing from `gc_analyzer.py`. Changes to parsing behavior must be applied in **both** files to stay consistent.
- Chart generation in the web path encodes matplotlib figures as base64 strings (no files written); the CLI path writes PNG files to disk.
- Observations are rule-based heuristics with three severity levels: `OK`, `INFO`, `WARN`.

## Test Data

- `TestLogs/test_wrapper.log` — real-world wrapper-format log (Tanuki/jsvc style with wall-clock timestamps)
- `sample_gc.log` — synthetic log; regenerate with `generate_sample_gc_log.py`
