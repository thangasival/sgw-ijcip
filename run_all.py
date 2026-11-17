"""Reproduce the IJCIP results from a clean copy of this package.

    python run_all.py           # tests, recomputation of G2 from raw run records, figures, manuscript (minutes)
    python run_all.py --rerun   # additionally re-solve every experiment (about 1,370 LPs; several hours on 8 cores)

Exit code 0 means every step succeeded."""
import subprocess, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PY = sys.executable

def step(name, args):
    print(f"== {name}", flush=True)
    r = subprocess.run([PY] + args, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAILED: {name}"); sys.exit(r.returncode)

def rerun():
    stages = ["stageA", "stageB", "stageC0", "stageC1", "stageC2", "stageD0", "stageS", "stageP", "stageP2", "stageP3"]
    out = ROOT / "outputs" / "rerun"; out.mkdir(parents=True, exist_ok=True)
    for st in stages:
        tasks = ROOT / "outputs" / "runs" / f"{st}_tasks.json"
        if tasks.exists():
            step(f"re-solve {st}", ["scripts/run_tasks.py", str(tasks), str(out / f"{st}.jsonl")])

if __name__ == "__main__":
    for t in sorted((ROOT / "tests").glob("test_*.py")):
        step(t.name, [str(t)])
    step("recompute gate G2 from run records", ["scripts/summarise_runs.py"])
    step("figures", ["-c", "import sys; sys.path.insert(0, '.'); from sgw import figures; figures.all_figures()"])
    step("manuscript", ["manuscript/build_manuscript.py"])
    if "--rerun" in sys.argv:
        rerun()
    print("run_all: all steps completed")
