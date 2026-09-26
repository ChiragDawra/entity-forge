import pytest

from entity_forge.pipeline import STAGES, _stage_range
from entity_forge.stages import forced


def test_stage_ranges_and_notebook_aliases():
    assert _stage_range(None, None) == list(STAGES)
    assert _stage_range("features", "stage2") == ["features", "stage1", "stage2"]
    assert _stage_range("02", "02") == ["candidates", "prune"]
    assert _stage_range("04", None)[0] == "stage1"
    assert _stage_range(None, "03")[-1] == "features"


def test_bad_stage_names_fail_clearly():
    with pytest.raises(SystemExit):
        _stage_range("nope", None)
    with pytest.raises(SystemExit):
        _stage_range("predict", "normalize")


def test_force_env(monkeypatch):
    monkeypatch.setenv("EF_FORCE", "prune, features")
    assert forced("prune") and forced("features") and not forced("stage1")
    monkeypatch.setenv("EF_FORCE", "all")
    assert forced("candidates")
    monkeypatch.setenv("EF_FORCE", "")
    assert not forced("prune")
