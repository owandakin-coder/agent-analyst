"""Deploy and inspect the autonomous runtime without printing credentials."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "sofowpweliticltlbxrj"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "deploy-db", "deploy-api", "verify-scheduler", "restart-worker"])
    args = parser.parse_args()
    if args.action == "restart-worker":
        restart_worker()
        return
    config = dotenv_values(ROOT / ".env")
    token = os.environ.get("SUPABASE_ACCESS_TOKEN") or config.get("SUPABASE_ACCESS_TOKEN")
    if not token:
        raise SystemExit("SUPABASE_ACCESS_TOKEN is required")

    def query(sql):
        request = urllib.request.Request(
            f"https://api.supabase.com/v1/projects/{PROJECT}/database/query",
            data=json.dumps({"query": sql}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)

    if args.action == "deploy-db":
        sql = (ROOT / "supabase/migrations/20261015210000_autonomous_paper_scheduler.sql").read_text(encoding="utf-8")
        query("begin;\n" + sql + "\ncommit;")
        print("Autonomous scheduler migration applied")
    elif args.action == "deploy-api":
        env = dict(os.environ, SUPABASE_ACCESS_TOKEN=token)
        subprocess.run(["npx.cmd" if os.name == "nt" else "npx", "--yes", "supabase", "functions", "deploy", "api", "--project-ref", PROJECT, "--no-verify-jwt", "--use-api"], env=env, cwd=ROOT, check=True)
    elif args.action == "status":
        print(json.dumps(query("select public.autonomous_runtime_status() as runtime"), indent=2))
    elif args.action == "verify-scheduler":
        # Roll back every change, including control state and test requests.
        sql = (ROOT / "tests/autonomous_scheduler.sql").read_text(encoding="utf-8")
        print(query(sql))


def restart_worker() -> None:
    """Restart a broken chain, without duplicating an active or queued run."""
    api = "https://api.github.com/repos/owandakin-coder/agent-analyst/actions/workflows/execution_poll.yml"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ATZMA-operations"}
    with urllib.request.urlopen(urllib.request.Request(api + "/runs?per_page=10", headers=headers), timeout=20) as response:
        runs = json.load(response)["workflow_runs"]
    if any(run["status"] in {"queued", "in_progress", "waiting", "requested", "pending"} for run in runs):
        print("Worker workflow already active; no restart needed")
        return
    github_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True,
                                env=dict(os.environ, GCM_INTERACTIVE="Never"), cwd=ROOT, check=True)
        credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        github_token = credential.get("password")
    if not github_token:
        raise SystemExit("No GitHub credential available for recovery")
    headers["Authorization"] = "Bearer " + github_token
    headers["Content-Type"] = "application/json"
    request = urllib.request.Request(api + "/dispatches", data=b'{"ref":"main"}', headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        print("Worker restart dispatched:", response.status)


if __name__ == "__main__":
    main()
