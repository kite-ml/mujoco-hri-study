"""Every built-in instrument must say where it came from.

Two different problems this guards against:

*Attribution.* Some published scales are free to use **on the condition that the
source is acknowledged** — SUS is the case in point. Shipping the items without
that acknowledgement puts every downstream user out of compliance without them
knowing, and it is invisible until someone checks.

*Validity.* ``builtin_ids()`` lists a validated 10-item scale and a two-item
measure written for one study side by side, looking equally official. A researcher
choosing between them has to be able to tell which is which, and an adapted scale
has to say so — psychometric properties do not transfer just because the items
look familiar.
"""

import json

import pytest

from mjhri.instruments import builtin_ids, get_instrument
from mjhri.instruments.registry import _BUILTIN_DIR

#: Instruments derived from published work: must carry a full reference.
PUBLISHED = {"sus", "nasa_tlx", "trust_scale", "attention_check_1", "attention_check_2"}

#: Instruments written for this study: must say so, so nobody cites them as a scale.
STUDY_DEFINED = {"control_steer_effort", "deploy_confidence"}


def _meta(instrument_id):
    return json.loads((_BUILTIN_DIR / f"{instrument_id}.json").read_text())["metadata"]


def test_the_two_groups_cover_every_builtin():
    """A new instrument fails here until it is classified as published or study-defined."""
    assert PUBLISHED | STUDY_DEFINED == set(builtin_ids())


@pytest.mark.parametrize("instrument_id", sorted(PUBLISHED | STUDY_DEFINED))
def test_every_instrument_declares_provenance(instrument_id):
    meta = _meta(instrument_id)
    assert meta, f"{instrument_id} has no metadata"
    assert meta.get("instrument"), f"{instrument_id} does not describe what it is"


@pytest.mark.parametrize("instrument_id", sorted(PUBLISHED | STUDY_DEFINED))
def test_numeric_instruments_state_their_response_format(instrument_id):
    """A rated instrument has to say what the numbers mean, or a stored 4 is
    uninterpretable later. Purely categorical instruments (the attention checks)
    carry their options on the items themselves and need no scale line."""
    inst = get_instrument(instrument_id)
    if not any(it.kind in ("likert", "slider") for it in inst.items):
        pytest.skip("no rated items")
    assert _meta(instrument_id).get("scale"), f"{instrument_id} has no scale description"


@pytest.mark.parametrize("instrument_id", sorted(PUBLISHED))
def test_published_instruments_carry_a_citation(instrument_id):
    citation = _meta(instrument_id).get("citation", "")
    assert citation, f"{instrument_id} derives from published work but has no citation"
    # A usable reference names an author and a year.
    assert any(ch.isdigit() for ch in citation), f"{instrument_id} citation has no year"
    assert len(citation) > 40, f"{instrument_id} citation is too thin to cite from"


@pytest.mark.parametrize("instrument_id", sorted(STUDY_DEFINED))
def test_study_defined_instruments_do_not_pose_as_scales(instrument_id):
    desc = _meta(instrument_id)["instrument"].lower()
    assert "study-defined" in desc or "not a published scale" in desc, (
        f"{instrument_id} is ad-hoc but does not say so"
    )


def test_sus_acknowledges_its_source():
    """SUS is free to use *provided the source is acknowledged* — so the notice ships."""
    meta = _meta("sus")
    attribution = meta.get("attribution", "")
    assert "Digital Equipment Corporation" in attribution
    assert "1986" in attribution
    assert "Brooke" in meta["citation"]


@pytest.mark.parametrize("instrument_id", ["sus", "trust_scale"])
def test_adapted_instruments_disclose_the_adaptation(instrument_id):
    """Both reword a published scale, so both must say what changed."""
    meta = _meta(instrument_id)
    assert "ADAPTED" in meta["instrument"], f"{instrument_id} is adapted but not flagged"
    adaptation = meta.get("adaptation", "")
    assert len(adaptation) > 60, f"{instrument_id} does not explain what it changed"


def test_reverse_scored_items_are_flagged_on_sus():
    """SUS scoring depends on the even items being reverse-coded; a missing flag
    silently corrupts the composite rather than raising."""
    sus = get_instrument("sus")
    assert len(sus.items) == 10
    for i, item in enumerate(sus.items, start=1):
        assert item.reverse == (i % 2 == 0), f"sus_{i} reverse flag is wrong"
