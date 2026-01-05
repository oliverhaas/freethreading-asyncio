"""
Benchmark for Python 3.14 free-threading with asyncio.

This script tests whether multiple threads running asyncio event loops
can achieve true parallelism with the GIL disabled.
"""

import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


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


def benchmark_single_thread(iterations: int) -> tuple[float, float]:
    """Run CPU work sequentially in a single thread."""
    start = time.perf_counter()
    result1 = cpu_intensive_work(iterations)
    result2 = cpu_intensive_work(iterations)
    elapsed = time.perf_counter() - start
    return elapsed, result1 + result2


def benchmark_two_threads(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using two threads with asyncio."""
    results = [0.0, 0.0]

    thread1 = threading.Thread(target=run_asyncio_in_thread, args=(iterations, results, 0))
    thread2 = threading.Thread(target=run_asyncio_in_thread, args=(iterations, results, 1))

    start = time.perf_counter()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    elapsed = time.perf_counter() - start

    return elapsed, sum(results)


def benchmark_two_threads_no_asyncio(iterations: int) -> tuple[float, float]:
    """Run CPU work in parallel using two threads without asyncio."""
    results = [0.0, 0.0]

    def worker(idx: int) -> None:
        results[idx] = cpu_intensive_work(iterations)

    thread1 = threading.Thread(target=worker, args=(0,))
    thread2 = threading.Thread(target=worker, args=(1,))

    start = time.perf_counter()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    elapsed = time.perf_counter() - start

    return elapsed, sum(results)


def benchmark_thread_pool(iterations: int) -> tuple[float, float]:
    """Run CPU work using ThreadPoolExecutor."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(cpu_intensive_work, iterations)
        future2 = executor.submit(cpu_intensive_work, iterations)
        result = future1.result() + future2.result()
    elapsed = time.perf_counter() - start
    return elapsed, result


async def benchmark_asyncio_gather_threads(iterations: int) -> tuple[float, float]:
    """Use asyncio.to_thread to run CPU work in parallel."""
    start = time.perf_counter()
    results = await asyncio.gather(
        asyncio.to_thread(cpu_intensive_work, iterations),
        asyncio.to_thread(cpu_intensive_work, iterations),
    )
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


def print_result(name: str, elapsed: float, baseline: float | None = None) -> None:
    """Print benchmark result with optional speedup calculation."""
    if baseline is not None:
        speedup = baseline / elapsed
        efficiency = (speedup / 2) * 100  # 2 threads, so max speedup is 2x
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
    print(f"Running benchmarks with {iterations:,} iterations per task...")
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
    print_result("Single thread (2 sequential tasks)", single_time)

    # Two threads without asyncio
    two_threads_time, _ = benchmark_two_threads_no_asyncio(iterations)
    print_result("Two threads (no asyncio)", two_threads_time, single_time)

    # Two threads with asyncio
    two_asyncio_time, _ = benchmark_two_threads(iterations)
    print_result("Two threads (with asyncio event loops)", two_asyncio_time, single_time)

    # ThreadPoolExecutor
    pool_time, _ = benchmark_thread_pool(iterations)
    print_result("ThreadPoolExecutor (2 workers)", pool_time, single_time)

    # asyncio.to_thread
    gather_time, _ = asyncio.run(benchmark_asyncio_gather_threads(iterations))
    print_result("asyncio.to_thread + gather", gather_time, single_time)

    print("-" * 70)
    print()

    # Analysis
    print("Analysis:")
    if gil_enabled is False:
        if two_threads_time < single_time * 0.7:
            print("  ✓ True parallelism achieved! Threads are running on multiple CPUs.")
            print(f"  ✓ Best speedup: {single_time / min(two_threads_time, two_asyncio_time, pool_time, gather_time):.2f}x")
        else:
            print("  ⚠ Limited parallelism detected despite GIL being disabled.")
            print("    This could be due to other bottlenecks.")
    else:
        print("  ⚠ GIL is enabled, so threads cannot run Python code in parallel.")
        print("    Expected speedup is ~1.0x for CPU-bound work.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
