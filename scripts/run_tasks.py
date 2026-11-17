"""Run a task list (JSON) serially, optionally one shard of it: run_tasks.py tasks.json out.jsonl [shard nshards].
Parallelism is by independent processes (one per shard), which avoids inter-process pipes."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sgw import experiments as EX

if __name__ == "__main__":
    tasks = json.load(open(sys.argv[1])); out = sys.argv[2]
    k, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    tasks = [t for i, t in enumerate(tasks) if i % n == k]
    done = set()
    if Path(out).exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    with open(out, "a", encoding="utf-8") as fh:
        for i, t in enumerate(todo):
            r = EX.run_task(t)
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
            print(f"{i + 1}/{len(todo)} {t['id']} {r.get('status')} {r.get('secs', 0):.0f}s", flush=True)
