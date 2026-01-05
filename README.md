# Free-Threading + Asyncio Benchmarks

Benchmarks for testing Python 3.14's free-threading (no-GIL) mode with asyncio.

## Requirements

- Python 3.14+ with free-threading build (`python3.14t`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Running the Benchmark

```bash
uv run python benchmark.py
```

## What it tests

This benchmark measures whether Python's free-threading mode allows true parallelism when:

1. Running CPU-bound work in multiple threads
2. Running asyncio event loops in separate threads
3. Using `asyncio.to_thread()` for parallel execution
4. Using `ThreadPoolExecutor`

With the GIL disabled, we expect ~2x speedup when running 2 threads on CPU-bound work.
With the GIL enabled, we expect ~1x speedup (no parallelism for CPU-bound work).

## Expected Output

With free-threading enabled (GIL disabled), you should see speedups close to 2x:

```
Benchmark Results:
----------------------------------------------------------------------
  Single thread (2 sequential tasks): 1.234s (baseline)
  Two threads (no asyncio): 0.650s (speedup: 1.90x, efficiency: 95.0%)
  Two threads (with asyncio event loops): 0.660s (speedup: 1.87x, efficiency: 93.5%)
  ...
```
