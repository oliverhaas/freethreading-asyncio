# Free-Threading + Asyncio Benchmarks

Benchmarks for testing Python 3.14's free-threading (no-GIL) mode with asyncio.

## Requirements

- Python 3.14+ with free-threading build (`python3.14t`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Running the Benchmarks

### Core benchmark (threading vs multiprocessing)

```bash
uv run python benchmark.py
```

### Django benchmark

```bash
uv run python django_benchmark.py
```

## What it tests

### benchmark.py

Measures whether Python's free-threading mode allows true parallelism:

**Threading:**
- Raw threads with/without asyncio event loops
- `ThreadPoolExecutor`
- `asyncio.to_thread()`

**Multiprocessing:**
- Raw processes with/without asyncio event loops
- `ProcessPoolExecutor`
- `asyncio` with `ProcessPoolExecutor`

### django_benchmark.py

Tests Django with a threaded WSGI server handling CPU-bound views concurrently.

## Expected Output

With free-threading enabled (GIL disabled), you should see speedups close to 4x with 4 workers:

```
Benchmark Results:
----------------------------------------------------------------------
  Single thread (4 sequential tasks): 0.661s (baseline)

  Threading:
  4 threads (no asyncio): 0.230s (speedup: 2.87x, efficiency: 71.8%)
  ThreadPoolExecutor (4 workers): 0.215s (speedup: 3.07x, efficiency: 76.7%)
  ...

  Multiprocessing:
  4 processes (no asyncio): 0.322s (speedup: 2.05x, efficiency: 51.3%)
  ProcessPoolExecutor (4 workers): 0.232s (speedup: 2.85x, efficiency: 71.1%)
  ...

Analysis:
  ✓ True parallelism achieved with threads! GIL-free threading works.
  ✓ Threading is comparable to multiprocessing (108% efficiency)
```
