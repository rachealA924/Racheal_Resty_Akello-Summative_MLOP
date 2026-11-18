"""
locustfile.py: simulates a flood of users sending leaf photos to the model.

Each simulated user keeps sending a random test image to POST /predict, and now
and then checks GET /health. Run it with the web UI:

    locust -f locustfile.py --host http://localhost:8080

or headless (no browser), the way scripts/run_load_tests.py does:

    locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s --csv results/locust/run
"""

import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent / "data" / "test"
IMAGES = [p.read_bytes() for p in sorted(TEST_DIR.rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if not IMAGES:
    raise SystemExit("No images in data/test. Run the notebook first (or copy some test images there).")


class LeafUser(HttpUser):
    wait_time = between(0.2, 1.0)   # each user pauses 0.2 to 1 s between requests

    @task(10)
    def predict(self):
        img = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": ("leaf.jpg", img, "image/jpeg")},
                              name="POST /predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "label" not in r.json():
                r.failure("no label in response")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
