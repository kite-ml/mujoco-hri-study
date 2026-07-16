"""Survey instrument schema — renderer-agnostic questionnaire definitions.

Instruments are data (see ``mjhri/instruments/builtin/*.json``); any UI can render
them and any store can persist the ``{item_id: value}`` responses. ``validate_response``
is the one behaviour the schema owns, so hosts validate identically.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ItemKind = Literal["likert", "slider", "choice", "text"]


class InstrumentItem(BaseModel):
    id: str
    prompt: str
    kind: ItemKind = "likert"
    # likert / slider
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    min_label: Optional[str] = None
    max_label: Optional[str] = None
    reverse: bool = Field(False, description="reverse-scored item (for subscale aggregation)")
    subscale: Optional[str] = Field(None, description="grouping key, e.g. a NASA-TLX dimension")
    # choice
    options: Optional[list[str]] = None
    required: bool = True


class Instrument(BaseModel):
    id: str
    title: str
    description: str = ""
    items: list[InstrumentItem]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def item(self, item_id: str) -> Optional[InstrumentItem]:
        return next((it for it in self.items if it.id == item_id), None)

    def validate_response(self, response: dict[str, Any]) -> list[str]:
        """Return human-readable validation errors for a ``{item_id: value}`` map
        (empty list = valid)."""
        errors: list[str] = []
        for it in self.items:
            if it.id not in response or response[it.id] is None:
                if it.required:
                    errors.append(f"missing required item '{it.id}'")
                continue
            val = response[it.id]
            if it.kind in ("likert", "slider"):
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    errors.append(f"item '{it.id}' expects a number")
                    continue
                if it.min is not None and val < it.min:
                    errors.append(f"item '{it.id}' = {val} below min {it.min}")
                if it.max is not None and val > it.max:
                    errors.append(f"item '{it.id}' = {val} above max {it.max}")
            elif it.kind == "choice":
                if it.options is not None and val not in it.options:
                    errors.append(f"item '{it.id}' = {val!r} not in {list(it.options)}")
            elif it.kind == "text":
                if not isinstance(val, str):
                    errors.append(f"item '{it.id}' expects text")
        return errors
