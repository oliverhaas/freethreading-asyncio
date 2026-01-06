"""
Minimal Django benchmark for Python 3.14 free-threading.

Uses Granian server with WSGI interface and blocking threads.
"""

import os
import sys
import threading
import time
import subprocess
import socket

# Configure Django settings before importing anything else
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_benchmark")

SECRET_KEY = "benchmark-secret-key-not-for-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]
ROOT_URLCONF = "django_benchmark"
INSTALLED_APPS = [
    "django.contrib.contenttypes",
]
DATABASES = {}
USE_TZ = True

# Now import Django
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY=SECRET_KEY,
        DEBUG=DEBUG,
        ALLOWED_HOSTS=ALLOWED_HOSTS,
        ROOT_URLCONF=ROOT_URLCONF,
        INSTALLED_APPS=INSTALLED_APPS,
        DATABASES=DATABASES,
        USE_TZ=USE_TZ,
    )

django.setup()

from django.http import JsonResponse
from django.urls import path
from django.core.wsgi import get_wsgi_application

NUM_WORKERS = 4
ITERATIONS = 5_000_000  # CPU work per request


def cpu_intensive_work(iterations: int) -> float:
    """Pure Python CPU-bound work."""
    total = 0.0
    for i in range(iterations):
        total += i * i / (i + 1)
    return total


def cpu_bound_view(request):
    """A sync view that does CPU-intensive work."""
    start = time.perf_counter()
    result = cpu_intensive_work(ITERATIONS)
    elapsed = time.perf_counter() - start
    return JsonResponse({
        "result": result,
        "elapsed": elapsed,
        "thread": threading.current_thread().name,
    })


def health_view(request):
    """Simple health check."""
    return JsonResponse({"status": "ok"})


# URL patterns
urlpatterns = [
    path("cpu/", cpu_bound_view),
    path("health/", health_view),
]

# WSGI application
application = get_wsgi_application()


def find_free_port() -> int:
    """Find a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_concurrent_requests_sync(url: str, num_requests: int) -> list[dict]:
    """Make concurrent HTTP requests using threads."""
    import httpx
    from concurrent.futures import ThreadPoolExecutor

    def make_request():
        with httpx.Client() as client:
            response = client.get(url, timeout=120.0)
            return response.json()

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        return [f.result() for f in futures]


def run_benchmark(port: int):
    """Run the benchmark against the server."""
    import httpx

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    for _ in range(50):
        try:
            response = httpx.get(f"{base_url}/health/", timeout=1.0)
            if response.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        print("Server failed to start")
        return

    print(f"Granian server running on {base_url}")
    print()

    # Sequential baseline
    print(f"Making {NUM_WORKERS} sequential requests...")
    start = time.perf_counter()
    for _ in range(NUM_WORKERS):
        httpx.get(f"{base_url}/cpu/", timeout=120.0)
    sequential_time = time.perf_counter() - start
    print(f"  Sequential: {sequential_time:.3f}s")

    # Concurrent requests
    print(f"Making {NUM_WORKERS} concurrent requests...")
    start = time.perf_counter()
    concurrent_results = run_concurrent_requests_sync(f"{base_url}/cpu/", NUM_WORKERS)
    concurrent_time = time.perf_counter() - start
    print(f"  Concurrent: {concurrent_time:.3f}s")

    # Results
    print()
    speedup = sequential_time / concurrent_time
    efficiency = (speedup / NUM_WORKERS) * 100
    print(f"Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")

    threads_used = set(r["thread"] for r in concurrent_results)
    print(f"Threads used: {len(threads_used)} - {sorted(threads_used)}")

    # Analysis
    gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else "N/A"
    print()
    if gil_enabled is False and speedup > 1.5:
        print("✓ Django + Granian with free-threading achieves parallel request handling!")
    elif gil_enabled is False:
        print("⚠ Limited parallelism detected")
    else:
        print("⚠ GIL enabled - requests processed sequentially")


def main():
    print("=" * 70)
    print("Django + Granian Free-Threading Benchmark")
    print("=" * 70)
    print()

    print("System Information:")
    print(f"  Python: {sys.version}")
    gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else "N/A"
    print(f"  GIL enabled: {gil_enabled}")

    import django
    print(f"  Django: {django.__version__}")

    try:
        import granian
        print(f"  Granian: {granian.__version__}")
    except Exception:
        print("  Granian: installed")
    print()

    port = find_free_port()

    # Start granian server with WSGI interface
    # With free-threading, workers become threads in a single process
    # Use more workers than requests to ensure all requests can be handled in parallel
    cmd = [
        sys.executable, "-m", "granian",
        "--interface", "wsgi",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--workers", str(NUM_WORKERS * 2),  # Extra workers to handle concurrent requests
        "--backpressure", str(NUM_WORKERS * 2),
        "django_benchmark:application",
    ]

    print(f"Starting Granian with {NUM_WORKERS * 2} workers...")
    server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        run_benchmark(port)
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
