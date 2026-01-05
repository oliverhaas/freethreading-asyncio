"""
Benchmark for Python 3.14 free-threading with asyncio.

This script tests whether multiple threads running asyncio event loops
can achieve true parallelism with the GIL disabled.
"""

import asyncio
import multiprocessing
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def cpu_intensive_work(iterations: int = 10_000_000) -> float:
    """Pure Python CPU-bound work that would normally be blocked by the GIL."""
    total = 0.0
    for i in range(iterations):
        total += i * i / (i + 1)
    return total


async def async_cpu_work(iterations: int = 10_000_000) -> float:
    """Run CPU-intensive work within an async context."""
    return cpu_intensive_work(iterations)


def run_asyncio_in_thread(iterations: int, results: list, index: int) -> None:
    """Run an asyncio event loop in a thread with CPU-bound work."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_cpu_work(iterations))
        results[index] = result
    finally:
        loop.close()


NUM_THREADS = 4


def benchmark_single_thread(iterations: int) -> tuple[float, float]:
    """Run CPU work sequentially in a single thread."""
    start = time.perf_counter()
    results = [cpu_intensive_work(iterations) for _ in range(NUM_THREADS)]
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def benchmark_threads_with_asyncio(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using threads with asyncio."""
    results = [0.0] * NUM_THREADS
    threads = [
        threading.Thread(target=run_asyncio_in_thread, args=(iterations, results, i))
        for i in range(NUM_THREADS)
    ]

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    return elapsed, sum(results)


def benchmark_threads_no_asyncio(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using threads without asyncio."""
    results = [0.0] * NUM_THREADS

    def worker(idx: int) -> None:
        results[idx] = cpu_intensive_work(iterations)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    return elapsed, sum(results)


def benchmark_thread_pool(iterations: int) -> tuple[float, float]:
    """Run CPU work using ThreadPoolExecutor."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(cpu_intensive_work, iterations) for _ in range(NUM_THREADS)]
        result = sum(f.result() for f in futures)
    elapsed = time.perf_counter() - start
    return elapsed, result


async def benchmark_asyncio_to_thread(iterations: int) -> tuple[float, float]:
    """Use asyncio.to_thread to run CPU work in parallel."""
    start = time.perf_counter()
    results = await asyncio.gather(
        *[asyncio.to_thread(cpu_intensive_work, iterations) for _ in range(NUM_THREADS)]
    )
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def run_asyncio_in_process(iterations: int) -> float:
    """Run an asyncio event loop in a process with CPU-bound work."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_cpu_work(iterations))
    finally:
        loop.close()


def benchmark_processes_with_asyncio(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using processes with asyncio."""
    start = time.perf_counter()
    with multiprocessing.Pool(NUM_THREADS) as pool:
        results = pool.map(run_asyncio_in_process, [iterations] * NUM_THREADS)
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def benchmark_processes_no_asyncio(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using processes without asyncio."""
    start = time.perf_counter()
    with multiprocessing.Pool(NUM_THREADS) as pool:
        results = pool.map(cpu_intensive_work, [iterations] * NUM_THREADS)
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def benchmark_process_pool(iterations: int) -> tuple[float, float]:
    """Run CPU work using ProcessPoolExecutor."""
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(cpu_intensive_work, iterations) for _ in range(NUM_THREADS)]
        result = sum(f.result() for f in futures)
    elapsed = time.perf_counter() - start
    return elapsed, result


async def benchmark_asyncio_run_in_executor_process(iterations: int) -> tuple[float, float]:
    """Use asyncio with ProcessPoolExecutor."""
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [loop.run_in_executor(executor, cpu_intensive_work, iterations) for _ in range(NUM_THREADS)]
        results = await asyncio.gather(*futures)
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def print_result(name: str, elapsed: float, baseline: float | None = None) -> None:
    """Print benchmark result with optional speedup calculation."""
    if baseline is not None:
        speedup = baseline / elapsed
        efficiency = (speedup / NUM_THREADS) * 100
        print(f"  {name}: {elapsed:.3f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
    else:
        print(f"  {name}: {elapsed:.3f}s (baseline)")


def main() -> None:
    """Run all benchmarks and report results."""
    print("=" * 70)
    print("Python Free-Threading + Asyncio Benchmark")
    print("=" * 70)
    print()

    # System info
    print("System Information:")
    print(f"  Python version: {sys.version}")
    gil_enabled = sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else "N/A"
    print(f"  GIL enabled: {gil_enabled}")
    print(f"  CPU count: {threading.active_count()} active threads")

    import os
    cpu_count = os.cpu_count()
    print(f"  Available CPUs: {cpu_count}")
    print()

    if gil_enabled is True:
        print("WARNING: GIL is enabled! Parallel execution will be limited.")
        print("         Run with PYTHON_GIL=0 or use a free-threading build.")
        print()

    iterations = 5_000_000
    print(f"Running benchmarks with {iterations:,} iterations per task, {NUM_THREADS} workers...")
    print()

    # Warmup
    print("Warming up...")
    cpu_intensive_work(100_000)
    print()

    # Run benchmarks
    print("Benchmark Results:")
    print("-" * 70)

    # Single thread baseline
    single_time, _ = benchmark_single_thread(iterations)
    print_result(f"Single thread ({NUM_THREADS} sequential tasks)", single_time)

    print()
    print("  Threading:")

    # Threads without asyncio
    threads_time, _ = benchmark_threads_no_asyncio(iterations)
    print_result(f"{NUM_THREADS} threads (no asyncio)", threads_time, single_time)

    # Threads with asyncio
    threads_asyncio_time, _ = benchmark_threads_with_asyncio(iterations)
    print_result(f"{NUM_THREADS} threads (with asyncio event loops)", threads_asyncio_time, single_time)

    # ThreadPoolExecutor
    thread_pool_time, _ = benchmark_thread_pool(iterations)
    print_result(f"ThreadPoolExecutor ({NUM_THREADS} workers)", thread_pool_time, single_time)

    # asyncio.to_thread
    to_thread_time, _ = asyncio.run(benchmark_asyncio_to_thread(iterations))
    print_result("asyncio.to_thread + gather", to_thread_time, single_time)

    print()
    print("  Multiprocessing:")

    # Processes without asyncio
    procs_time, _ = benchmark_processes_no_asyncio(iterations)
    print_result(f"{NUM_THREADS} processes (no asyncio)", procs_time, single_time)

    # Processes with asyncio
    procs_asyncio_time, _ = benchmark_processes_with_asyncio(iterations)
    print_result(f"{NUM_THREADS} processes (with asyncio event loops)", procs_asyncio_time, single_time)

    # ProcessPoolExecutor
    proc_pool_time, _ = benchmark_process_pool(iterations)
    print_result(f"ProcessPoolExecutor ({NUM_THREADS} workers)", proc_pool_time, single_time)

    # asyncio + ProcessPoolExecutor
    asyncio_proc_time, _ = asyncio.run(benchmark_asyncio_run_in_executor_process(iterations))
    print_result("asyncio + ProcessPoolExecutor", asyncio_proc_time, single_time)

    print("-" * 70)
    print()

    # Analysis
    print("Analysis:")

    best_thread_time = min(threads_time, threads_asyncio_time, thread_pool_time, to_thread_time)
    best_proc_time = min(procs_time, procs_asyncio_time, proc_pool_time, asyncio_proc_time)
    best_thread_speedup = single_time / best_thread_time
    best_proc_speedup = single_time / best_proc_time

    if gil_enabled is False:
        if best_thread_speedup > NUM_THREADS * 0.5:
            print("  ✓ True parallelism achieved with threads! GIL-free threading works.")
            print(f"  ✓ Best thread speedup: {best_thread_speedup:.2f}x (theoretical max: {NUM_THREADS}x)")
        else:
            print("  ⚠ Limited thread parallelism despite GIL being disabled.")
    else:
        print("  ⚠ GIL is enabled, so threads cannot run Python code in parallel.")
        print("    Expected thread speedup is ~1.0x for CPU-bound work.")

    print(f"  ✓ Best process speedup: {best_proc_speedup:.2f}x (theoretical max: {NUM_THREADS}x)")

    if gil_enabled is False and best_thread_speedup > 0:
        ratio = best_thread_speedup / best_proc_speedup
        if ratio > 0.9:
            print(f"  ✓ Threading is comparable to multiprocessing ({ratio:.0%} efficiency)")
        else:
            print(f"  → Threading achieves {ratio:.0%} of multiprocessing performance")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
