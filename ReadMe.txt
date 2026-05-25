================================================================================
 JVM Garbage Collection (GC) Log Analyzer
================================================================================

A Python tool for parsing, analyzing, and visualizing JVM garbage collection
logs. Provides two interfaces:

  1. Web UI  (gc_web.py)      - Flask app with upload, charts, KPIs, and
                                rule-based observations.
  2. CLI     (gc_analyzer.py) - Batch parsing, CSV export, static PNG charts.

Supported log formats:
  - JDK 8  -XX:+PrintGC / +PrintGCDetails / +PrintGCDateStamps
  - JDK 8  G1GC pause-style entries
  - JDK 9+ -Xlog:gc* unified logging
  - ZGC / Shenandoah concurrent entries
  - Wrapper-format logs (Tanuki / jsvc with wall-clock timestamps)


--------------------------------------------------------------------------------
 1. PREREQUISITES
--------------------------------------------------------------------------------

  - Python 3.10 or newer (tested on 3.13)
  - pip
  - A modern web browser (for the web UI)


--------------------------------------------------------------------------------
 2. INSTALL DEPENDENCIES
--------------------------------------------------------------------------------

Open PowerShell or a terminal in the project folder
(c:\DOCS\AI_Claude\CoCode_Playground) and run:

    pip install -r requirements_gc.txt

This installs:
    flask        - web framework
    matplotlib   - chart rendering
    reportlab    - PDF report generation
    Pillow       - image handling


--------------------------------------------------------------------------------
 3. RUN THE WEB INTERFACE  (recommended)
--------------------------------------------------------------------------------

From the project folder:

    python gc_web.py

Then open a browser at:

    http://localhost:5000

Usage in the browser:
  1. Drag and drop a GC log file onto the upload area (or click to browse).
  2. Wait while the file is parsed and KPIs/charts are generated.
  3. Use the tabs to view Overview, Heap charts, KPIs, and Observations.
  4. Optionally export a PDF report.

Sample logs you can try:
    sample_gc.log
    TestLogs\test_wrapper.log
    TestLogs\Sample_GC.log

To stop the server, press Ctrl+C in the terminal.


--------------------------------------------------------------------------------
 4. RUN THE COMMAND-LINE INTERFACE
--------------------------------------------------------------------------------

Basic analysis (prints a report and writes chart PNGs to the current folder):

    python gc_analyzer.py <path-to-log-file>

Examples:

    python gc_analyzer.py sample_gc.log
    python gc_analyzer.py TestLogs\test_wrapper.log

Options:

    --out-dir DIR    Directory where chart images are written
                     (default: current folder)

    --csv FILE       Also export every parsed GC event to a CSV file

Full example:

    python gc_analyzer.py TestLogs\test_wrapper.log --out-dir reports --csv gc_events.csv

Get help:

    python gc_analyzer.py --help


--------------------------------------------------------------------------------
 5. GENERATE A SYNTHETIC SAMPLE LOG
--------------------------------------------------------------------------------

If you do not have a GC log handy, create one:

    python generate_sample_gc_log.py

This produces sample_gc.log in the current folder, which you can feed to
either gc_web.py or gc_analyzer.py.


--------------------------------------------------------------------------------
 6. PROJECT LAYOUT
--------------------------------------------------------------------------------

    gc_analyzer.py              Parsing engine + CLI entry point
    gc_web.py                   Flask web application
    generate_sample_gc_log.py   Synthetic GC log generator
    templates\index.html        Web UI (single page)
    requirements_gc.txt         Python dependencies
    TestLogs\                   Real and sample GC log files
    JVM_GC_Log_Analyzer_User_Guide.pdf   Detailed user guide


--------------------------------------------------------------------------------
 7. TROUBLESHOOTING
--------------------------------------------------------------------------------

  - "ModuleNotFoundError: No module named 'flask'"
      -> Run:  pip install -r requirements_gc.txt

  - "Port 5000 already in use"
      -> Edit the last line of gc_web.py and change port=5000 to a free
         port, e.g. port=5050.

  - "No GC events could be parsed"
      -> The log file is empty or in an unsupported format. Verify the JVM
         flags used to produce it (see Supported log formats above).

  - Charts not appearing in the web UI
      -> Make sure matplotlib installed cleanly. Try:
             pip install --upgrade matplotlib Pillow

================================================================================
