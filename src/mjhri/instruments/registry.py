"""Load survey instruments from JSON — built-in ones ship in ``builtin/``.

Instruments are pure data validated into ``Instrument`` models, so a host renders
and a store persists responses without importing any study code. Studies can load
their own instrument JSONs from a directory with ``load_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas.instruments import Instrument

_BUILTIN_DIR = Path(__file__).parent / "builtin"


def builtin_ids() -> list[str]:
    return sorted(p.stem for p in _BUILTIN_DIR.glob("*.json"))


def load_instrument(path: str | Path) -> Instrument:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Instrument.model_validate(data)


def get_instrument(instrument_id: str) -> Instrument:
    """Load a built-in instrument by id (its filename stem)."""
    path = _BUILTIN_DIR / f"{instrument_id}.json"
    if not path.exists():
        raise KeyError(f"no built-in instrument {instrument_id!r}; have {builtin_ids()}")
    return load_instrument(path)


def load_dir(path: str | Path) -> dict[str, Instrument]:
    """Load every ``*.json`` in a directory into ``{id: Instrument}``."""
    out: dict[str, Instrument] = {}
    for p in sorted(Path(path).glob("*.json")):
        inst = load_instrument(p)
        out[inst.id] = inst
    return out
