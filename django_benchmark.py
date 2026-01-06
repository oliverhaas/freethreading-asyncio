"""
Minimal Django benchmark for Python 3.14 free-threading.

Spins up a threaded WSGI server with Django and makes concurrent requests.
"""

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer

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

from django.core.handlers.wsgi import WSGIHandler
from django.http import JsonResponse
from django.urls import path

NUM_WORKERS = 4
ITERATIONS = 2_000_000


def cpu_intensive_work(iterations: int) -> float:
    """Pure Python CPU-bound work."""
    total = 0.0
    for i in range(iterations):
        total += i * i / (i + 1)
    return total


def cpu_bound_view(request):
    """A view that does CPU-intensive work."""
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


class QuietWSGIRequestHandler(WSGIRequestHandler):
    """WSGI handler that doesn't log requests."""
    def log_message(self, format, *args):
        pass


class ThreadPoolWSGIServer(WSGIServer):
    """WSGI server that handles requests in a thread pool."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread, request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


async def run_concurrent_requests(url: str, num_requests: int) -> list[dict]:
    """Make concurrent HTTP requests."""
    import httpx

    async with httpx.AsyncClient() as client:
        tasks = [client.get(url, timeout=60.0) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]


def run_server_and_benchmark():
    """Run threaded WSGI server and make concurrent requests."""
    import socket

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    base_url = f"http://127.0.0.1:{port}"

    # Create and start server
    application = WSGIHandler()
    server = ThreadPoolWSGIServer(("127.0.0.1", port), QuietWSGIRequestHandler)
    server.set_app(application)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(0.2)

    # Test health endpoint first
    import httpx
    try:
        response = httpx.get(f"{base_url}/health/", timeout=5.0)
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"Could not connect to server: {e}")
        return

    print(f"Threaded WSGI server running on {base_url}")
    print(f"Thread pool size: {NUM_WORKERS}")
    print()

    # Sequential baseline
    print(f"Making {NUM_WORKERS} sequential requests...")
    start = time.perf_counter()
    sequential_results = []
    for _ in range(NUM_WORKERS):
        response = httpx.get(f"{base_url}/cpu/", timeout=60.0)
        sequential_results.append(response.json())
    sequential_time = time.perf_counter() - start
    print(f"  Sequential: {sequential_time:.3f}s")

    # Concurrent requests
    print(f"Making {NUM_WORKERS} concurrent requests...")
    start = time.perf_counter()
    concurrent_results = asyncio.run(run_concurrent_requests(f"{base_url}/cpu/", NUM_WORKERS))
    concurrent_time = time.perf_counter() - start
    print(f"  Concurrent: {concurrent_time:.3f}s")

    # Results
    print()
    speedup = sequential_time / concurrent_time
    efficiency = (speedup / NUM_WORKERS) * 100
    print(f"Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")

    # Show which threads handled requests
    threads_used = set(r["thread"] for r in concurrent_results)
    print(f"Threads used: {len(threads_used)} ({', '.join(sorted(threads_used))})")

    gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else "N/A"
    print()
    if gil_enabled is False and speedup > 1.5:
        print("✓ Django with free-threading achieves parallel CPU-bound request handling!")
    elif gil_enabled is False:
        print("⚠ Limited parallelism - server may have bottlenecks")
    else:
        print("⚠ GIL enabled - concurrent requests processed sequentially")

    # Shutdown
    server.shutdown()


def main():
    print("=" * 70)
    print("Django + Free-Threading Benchmark")
    print("=" * 70)
    print()

    print("System Information:")
    print(f"  Python: {sys.version}")
    gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else "N/A"
    print(f"  GIL enabled: {gil_enabled}")

    import django
    print(f"  Django: {django.__version__}")
    print()

    run_server_and_benchmark()

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
