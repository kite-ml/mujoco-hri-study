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
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = "google-deepmind/mujoco_menagerie"
MODEL = "trs_so_arm100"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/{MODEL}"
DEST = Path(__file__).parent / "scenes"
TIMEOUT = 60


def _get(url: str, retries: int = 3) -> bytes:
    """GET with a short backoff, so one flaky response does not fail the whole fetch."""
    headers = {"User-Agent": "mjhri-fetch-assets"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (403, 429):        # rate limited — backing off will not help much
                raise
            if exc.code < 500:
                raise
        except urllib.error.URLError as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise last  # type: ignore[misc]


def _mesh_names(xml_text: str) -> list[str]:
    """Mesh filenames referenced by the model, read from the model itself.

    Listing the directory would mean calling the GitHub *API*, which is rate-limited
    per IP and unauthenticated by default — that fails on shared/NAT'd networks
    (university proxies, CI runners) with a 403 that has nothing to do with the user.
    raw.githubusercontent.com is a CDN with no such limit, and the XML already names
    every file it needs, so parse it and fetch those directly.
    """
    names: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return names
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and f not in names:
            names.append(f)
    return names


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
    xml = _get(f"{RAW}/so_arm100.xml")
    (DEST / "so_arm100.xml").write_bytes(xml)
    try:
        (DEST / "so_arm100_LICENSE").write_bytes(_get(f"{RAW}/LICENSE"))
    except urllib.error.HTTPError:
        pass

    meshes = _mesh_names(xml.decode("utf-8"))
    if not meshes:
        sys.exit("error: could not read any mesh names from so_arm100.xml — "
                 "pass --from <mujoco_menagerie checkout> instead")
    for i, name in enumerate(meshes, 1):
        out = DEST / "assets" / name
        if not out.exists():
            out.write_bytes(_get(f"{RAW}/assets/{name}"))
        print(f"  [{i}/{len(meshes)}] {name}", end="\r", flush=True)
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
