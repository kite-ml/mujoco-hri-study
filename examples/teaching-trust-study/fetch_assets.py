#!/usr/bin/env python3
"""Fetch the SO-ARM100 arm the bundled study scenes ``<include>``.

The scenes in ``scenes/`` describe only the *task* — table, blocks, bins, and the
region sites the scorer reads. The arm itself comes from MuJoCo Menagerie, which we
do not vendor: it is ~4 MB of meshes with its own upstream that we would rather not
fork. This script drops it next to the scenes so they load.

    python examples/teaching-trust-study/fetch_assets.py

Downloads ``trs_so_arm100`` (Apache-2.0, © The Robot Studio / Google DeepMind) from
github.com/google-deepmind/mujoco_menagerie into ``scenes/``. Stdlib only; re-running
is a no-op. If you already have Menagerie locally, point at it instead:

    python examples/teaching-trust-study/fetch_assets.py --from /path/to/mujoco_menagerie
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "google-deepmind/mujoco_menagerie"
MODEL = "trs_so_arm100"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{MODEL}"
API = f"https://api.github.com/repos/{REPO}/contents/{MODEL}"
DEST = Path(__file__).parent / "scenes"
TIMEOUT = 60


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mjhri-fetch-assets"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def _copy_local(src_root: Path) -> int:
    src = src_root / MODEL if (src_root / MODEL).is_dir() else src_root
    xml = src / "so_arm100.xml"
    if not xml.is_file():
        sys.exit(f"error: {xml} not found — pass the mujoco_menagerie checkout root")
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy(xml, DEST / "so_arm100.xml")
    shutil.copytree(src / "assets", DEST / "assets", dirs_exist_ok=True)
    n = len(list((DEST / "assets").glob("*")))
    for extra in ("LICENSE", "README.md"):
        if (src / extra).is_file():
            shutil.copy(src / extra, DEST / f"so_arm100_{extra}")
    return n


def _download() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "assets").mkdir(exist_ok=True)
    print(f"fetching {MODEL} from {REPO} …")
    (DEST / "so_arm100.xml").write_bytes(_get(f"{RAW}/so_arm100.xml"))
    try:
        (DEST / "so_arm100_LICENSE").write_bytes(_get(f"{RAW}/LICENSE"))
    except urllib.error.HTTPError:
        pass
    listing = json.loads(_get(f"{API}/assets").decode("utf-8"))
    meshes = [e for e in listing if e.get("type") == "file"]
    for i, entry in enumerate(meshes, 1):
        out = DEST / "assets" / entry["name"]
        if not out.exists():
            out.write_bytes(_get(entry["download_url"]))
        print(f"  [{i}/{len(meshes)}] {entry['name']}", end="\r", flush=True)
    print()
    return len(meshes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", metavar="DIR",
                    help="copy from a local mujoco_menagerie checkout instead of downloading")
    args = ap.parse_args()

    if (DEST / "so_arm100.xml").is_file() and (DEST / "assets").is_dir() and not args.src:
        print(f"already present in {DEST} — nothing to do")
    else:
        n = _copy_local(Path(args.src).expanduser()) if args.src else _download()
        print(f"installed so_arm100.xml + {n} meshes into {DEST}")

    try:
        import mujoco
    except ImportError:
        print("note: `pip install mujoco` to verify the scenes load")
        return
    ok = 0
    for scene in sorted(DEST.glob("scene_*.xml")):
        try:
            mujoco.MjModel.from_xml_path(str(scene))
            ok += 1
        except Exception as exc:
            print(f"  FAILED {scene.name}: {exc}")
    print(f"verified: {ok}/{len(list(DEST.glob('scene_*.xml')))} study scenes load")


if __name__ == "__main__":
    main()
