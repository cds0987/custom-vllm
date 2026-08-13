#!/usr/bin/env python3
"""
Snapshot / restore the whole installed Python environment via a HF repo.

WHY
---
`pip install vllm` is the single most expensive step of a Colab rebuild (~12
min, dominated by torch at ~2.5 GB) and it is *identical every time*. After
nine runtime wipes that is well over an hour spent reinstalling byte-for-byte
the same tree. Tarring `dist-packages` once and pulling it back on the next
wipe turns that install into a download + untar (~4-5 min with hf_transfer).

    # once, on a runtime where setup_env.sh has already succeeded:
    python scripts/env_snapshot.py save --repo gunnybd01/qwen35-9b-env

    # on every later wipe, INSTEAD of setup_env.sh's install section:
    python scripts/env_snapshot.py restore --repo gunnybd01/qwen35-9b-env

WHEN IT IS *NOT* SAFE
---------------------
A snapshot is only valid on a machine whose Python and CUDA match the one that
produced it -- compiled extensions (torch, flashinfer, _C_gguf) are built
against both. Colab rotates its base image every few weeks, and a silently
mismatched restore is far worse than a slow install: it fails at import time,
or worse, at kernel-launch time deep inside a benchmark.

So `save` writes a manifest and `restore` refuses on mismatch rather than
"trying anyway". The guard is deliberately strict; a refused restore costs 12
minutes, a bad restore costs a debugging session.

The patches from setup_env.sh are baked into the snapshot (they edit files
inside dist-packages), so a restored environment is already patched. Re-running
the patch loop afterwards is harmless -- every patch script is idempotent.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from pathlib import Path

MANIFEST_NAME = "env_manifest.json"
TARBALL_NAME = "dist-packages.tar.zst"
# Caches and build detritus: pure bloat inside a snapshot.
EXCLUDE_DIRS = {"__pycache__", ".cache", "pip", "uv"}


def cuda_driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "none"
    except Exception:
        return "none"


def pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "absent"


def build_manifest() -> dict:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.platform(),
        "cuda_driver": cuda_driver_version(),
        "torch": pkg_version("torch"),
        "vllm": pkg_version("vllm"),
        "site_packages": sysconfig.get_paths()["purelib"],
    }


def compatible(saved: dict, current: dict) -> tuple[bool, str]:
    """Strict on what breaks compiled extensions, lax on cosmetics."""
    if saved["python"] != current["python"]:
        return False, f"Python {saved['python']} != {current['python']}"
    # Driver major version: a newer driver runs older CUDA, not vice versa.
    def major(v: str) -> int:
        try:
            return int(v.split(".")[0])
        except Exception:
            return -1
    if major(current["cuda_driver"]) < major(saved["cuda_driver"]):
        return False, (
            f"CUDA driver {current['cuda_driver']} older than snapshot's "
            f"{saved['cuda_driver']}"
        )
    return True, "ok"


def _filter(info: tarfile.TarInfo):
    parts = Path(info.name).parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return None
    return info


def do_save(repo: str, dry_run: bool) -> int:
    from huggingface_hub import HfApi

    manifest = build_manifest()
    src = Path(manifest["site_packages"])
    if not src.is_dir():
        print(f"ERROR: site-packages not found at {src}", file=sys.stderr)
        return 1
    print(f"Snapshotting {src}")
    print(json.dumps(manifest, indent=2))
    if dry_run:
        print("--dry-run: stopping before tar/upload")
        return 0

    tmp = Path(tempfile.mkdtemp())
    tar_path = tmp / TARBALL_NAME.replace(".zst", "")  # plain tar.gz fallback
    tar_path = tmp / "dist-packages.tar.gz"
    print(f"Creating {tar_path} (this takes a few minutes)")
    with tarfile.open(tar_path, "w:gz", compresslevel=1) as tf:
        tf.add(src, arcname=".", filter=_filter)
    size_gb = tar_path.stat().st_size / 1e9
    print(f"Tarball: {size_gb:.1f} GB")

    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"Uploading to {repo}")
    api.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="model")
    print("Saved. Restore on a fresh runtime with:")
    print(f"  python scripts/env_snapshot.py restore --repo {repo}")
    return 0


def do_restore(repo: str, force: bool, dry_run: bool) -> int:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        mpath = hf_hub_download(repo, MANIFEST_NAME, repo_type="model")
    except Exception as e:
        print(f"No snapshot available in {repo} ({e}); build the environment normally.")
        return 2  # distinct code so callers can fall back without treating it as failure

    saved = json.loads(Path(mpath).read_text(encoding="utf-8"))
    current = build_manifest()
    ok, why = compatible(saved, current)
    print(f"snapshot: {saved['python']=} {saved['cuda_driver']=} {saved['torch']=}")
    print(f"current : {current['python']=} {current['cuda_driver']=}")
    if not ok:
        print(f"INCOMPATIBLE: {why}")
        if not force:
            print("Refusing to restore. Build normally, then re-run `save` to "
                  "refresh the snapshot for this image. (--force to override.)")
            return 2
        print("--force given; restoring anyway")

    if dry_run:
        print("--dry-run: compatible, stopping before download")
        return 0

    print("Downloading tarball")
    tar_path = hf_hub_download(repo, "dist-packages.tar.gz", repo_type="model")
    dest = Path(current["site_packages"])
    print(f"Extracting into {dest}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    print("Restored. Re-run the patch loop if you changed any patch script "
          "(they are idempotent).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["save", "restore"])
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. user/qwen35-9b-env")
    ap.add_argument("--force", action="store_true",
                    help="restore even if the manifest says the image differs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.action == "save":
        return do_save(a.repo, a.dry_run)
    return do_restore(a.repo, a.force, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
