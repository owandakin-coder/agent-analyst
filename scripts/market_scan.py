"""Run the read-only, account-scoped market screener; never handles broker keys."""
import json
import os
import urllib.request
import urllib.error

BASE = "https://sofowpweliticltlbxrj.supabase.co/functions/v1/api"


def call(path, body):
    token = os.environ.get("ATZMA_WORKER_SHARED_TOKEN")
    if not token:
        raise RuntimeError("Worker token is unavailable")
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-atzma-worker-token": token})
    try:
        with urllib.request.urlopen(req, timeout=145) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Scanner API returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Scanner API connection failed") from None


def main():
    after = None
    totals = {"completed": 0, "skipped": 0, "failed": 0}
    while True:
        rows = call("/worker/scanner/connections", {"after": after})["connections"]
        for row in rows:
            try:
                result = call("/worker/scanner/run", {"connection_id": row["id"]})
                totals[result["status"]] += 1
            except RuntimeError as exc:
                totals["failed"] += 1
                print(str(exc))  # No response bodies, account identifiers or credentials.
        if len(rows) < 100:
            break
        after = rows[-1]["id"]
    print(json.dumps(totals))
    if totals["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
