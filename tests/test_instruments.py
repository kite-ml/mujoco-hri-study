"""Built-in instruments load and validate responses correctly."""

import pytest

from mjhri.instruments import builtin_ids, get_instrument


def test_all_builtins_load():
    ids = builtin_ids()
    assert {"trust_scale", "nasa_tlx", "control_steer_effort", "deploy_confidence"} <= set(ids)
    for iid in ids:
        inst = get_instrument(iid)
        assert inst.items, f"{iid} has no items"


def test_trust_scale_shape():
    inst = get_instrument("trust_scale")
    # TiA (Körber 2019) "Trust in Automation" subscale, items 9 & 14 only: trust + rely.
    assert len(inst.items) == 2
    assert {it.id for it in inst.items} == {"trust", "rely"}
    assert all(it.min == 1 and it.max == 5 for it in inst.items)  # 5-point Likert
    assert not any(it.reverse for it in inst.items)  # neither item is reverse-coded


def test_validate_response_flags_missing_and_out_of_range():
    inst = get_instrument("nasa_tlx")
    # complete, valid
    good = {it.id: 50 for it in inst.items}
    assert inst.validate_response(good) == []
    # missing one required item
    missing = dict(good)
    del missing["effort"]
    assert any("effort" in e for e in inst.validate_response(missing))
    # out of range
    bad = dict(good)
    bad["mental"] = 999
    assert any("mental" in e for e in inst.validate_response(bad))


def test_choice_validation():
    inst = get_instrument("deploy_confidence")
    assert inst.validate_response({"deploy": "Deploy", "confidence": 80}) == []
    errs = inst.validate_response({"deploy": "Maybe", "confidence": 80})
    assert any("deploy" in e for e in errs)
