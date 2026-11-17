"""Rewrite MANIFEST.sha256 for every file in the package except git metadata and caches.

Run only after PREREG.md is byte-identical to the version hashed on 13 Nov 2026
(SHA-256 ef5253d2...6614, outputs/prereg_hash.txt); the script refuses otherwise."""
import hashlib, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
EXPECT = (ROOT / "outputs" / "prereg_hash.txt").read_text().split()[0]
SKIP = {".git", "__pycache__", ".pytest_cache"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

if sha(ROOT / "PREREG.md") != EXPECT:
    sys.exit(f"PREREG.md does not match the pre-registered hash {EXPECT[:8]}...; restore the original first.")
files = sorted(p for p in ROOT.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(ROOT).parts)
               and p.name not in {"MANIFEST.sha256", ".gitattributes"})
(ROOT / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files))
print(f"MANIFEST.sha256: {len(files)} files")
