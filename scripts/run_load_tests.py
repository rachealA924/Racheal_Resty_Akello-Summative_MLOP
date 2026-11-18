"""
run_load_tests.py: runs the same Locust flood against 1, 2 and 3 Docker containers
and writes a comparison table and chart.

Needs Docker Desktop running and the dev requirements installed:
    pip install -r requirements-dev.txt
    python scripts/run_load_tests.py                      # 1, 2, 3 containers, 50 users, 60 s each
    python scripts/run_load_tests.py --containers 1 2 4 --users 100 --duration 90

Results go to results/locust/:
    <n>_containers_stats.csv      raw Locust statistics per run
    summary.csv / summary.md      one row per container count
    latency_vs_containers.png     chart for the README
"""

import argparse
import csv
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "locust"
HOST = "http://localhost:8080"


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_until_ready(n, timeout=300):
    """Wait until nginx answers and every container has loaded the model."""
    seen, start = set(), time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(HOST + "/health", timeout=5) as r:
                body = r.read().decode()
                if '"model_loaded":true' in body.replace(" ", ""):
                    seen.add(body.split('"container":"')[1].split('"')[0])
        except Exception:
            pass
        if len(seen) >= n:
            print(f"ready: {len(seen)} container(s) answering: {sorted(seen)}")
            return
        time.sleep(2)
    print(f"warning: only saw {len(seen)} of {n} containers answer; continuing anyway")


def read_stats(path):
    with open(path, newline="") as f:
        rows = {r["Name"]: r for r in csv.DictReader(f)}
    r = rows.get("POST /predict") or rows["Aggregated"]
    num = lambda k: float(r[k]) if r.get(k) not in (None, "", "N/A") else float("nan")  # noqa: E731
    return {
        "requests": int(num("Request Count")),
        "failures": int(num("Failure Count")),
        "avg_ms": round(num("Average Response Time"), 1),
        "median_ms": round(num("Median Response Time"), 1),
        "p95_ms": round(num("95%"), 1),
        "p99_ms": round(num("99%"), 1),
        "max_ms": round(num("Max Response Time"), 1),
        "rps": round(num("Requests/s"), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--containers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--spawn-rate", type=int, default=10)
    ap.add_argument("--duration", default="60s")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sh(["docker", "compose", "up", "-d", "--build"])
    summary = []
    for n in args.containers:
        print(f"\n===== {n} container(s) =====")
        sh(["docker", "compose", "up", "-d", "--scale", f"api={n}"])
        sh(["docker", "compose", "restart", "nginx"])
        wait_until_ready(n)
        prefix = OUT / f"{n}_containers"
        sh([sys.executable, "-m", "locust", "-f", "locustfile.py", "--host", HOST, "--headless",
            "-u", str(args.users), "-r", str(args.spawn_rate), "-t", args.duration,
            "--csv", str(prefix), "--only-summary"])
        row = {"containers": n, "users": args.users, "duration": args.duration,
               **read_stats(f"{prefix}_stats.csv")}
        summary.append(row)
        print(row)

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    head = ("| Containers | Users | Requests | Failures | Avg (ms) | Median (ms) | 95th pct (ms) "
            "| 99th pct (ms) | Max (ms) | Requests/s |\n|---|---|---|---|---|---|---|---|---|---|\n")
    lines = "".join(f"| {r['containers']} | {r['users']} | {r['requests']} | {r['failures']} | {r['avg_ms']} | "
                    f"{r['median_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['max_ms']} | {r['rps']} |\n"
                    for r in summary)
    (OUT / "summary.md").write_text(head + lines)
    print("\n" + head + lines)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        xs = [r["containers"] for r in summary]
        ax1.plot(xs, [r["avg_ms"] for r in summary], "o-", label="average latency")
        ax1.plot(xs, [r["p95_ms"] for r in summary], "s--", label="95th percentile latency")
        ax1.set(xlabel="number of API containers", ylabel="response time (ms)", xticks=xs,
                title=f"/predict under {args.users} concurrent users")
        ax2 = ax1.twinx()
        ax2.bar(xs, [r["rps"] for r in summary], alpha=0.2, color="green", label="requests/s")
        ax2.set_ylabel("requests per second")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")
        plt.tight_layout()
        plt.savefig(OUT / "latency_vs_containers.png", dpi=150)
        print("chart:", OUT / "latency_vs_containers.png")
    except ImportError:
        print("matplotlib not installed, skipping the chart")


if __name__ == "__main__":
    main()
