import json

import polars as pl
import pytest

from entity_forge import checkpoints as ck
from entity_forge.resources import PROFILES, auto_profile, resolve_profile
from entity_forge.settings import Settings


def _frame(n=5):
    return pl.DataFrame({"q": list(range(n)), "t": list(range(n))})


def test_file_manifest_roundtrip_and_size_check(tmp_path):
    f = tmp_path / "a.parquet"
    ck.atomic_write_parquet(_frame(), f)
    assert not ck.is_complete(f, "h1")  # no manifest yet
    m = ck.write_manifest(f, "stage", "k", "h1")
    assert m["rows"] == 5 and m["schema"] == {"q": "Int64", "t": "Int64"}
    assert ck.check(f, "h1") == (True, "ok")
    assert not ck.is_complete(f, "h2")  # config changed
    ck.atomic_write_parquet(_frame(50), f)  # file replaced behind the manifest's back
    ok, reason = ck.check(f, "h1")
    assert not ok and "size" in reason


def test_dir_manifest_and_invalidate_keeps_data(tmp_path):
    d = tmp_path / "part_dir"
    for i in range(3):
        ck.atomic_write_parquet(_frame(), ck.part_path(d, i))
    ck.write_manifest(d, "stage", "k", "h")
    assert ck.read_manifest(d)["rows"] == 15
    assert ck.invalidate(d)
    assert not ck.is_complete(d, "h")
    assert len(ck.data_files(d)) == 3  # data untouched


def test_missing_manifest_or_file(tmp_path):
    d = tmp_path / "x"
    ck.atomic_write_parquet(_frame(), ck.part_path(d, 0))
    ck.write_manifest(d, "s", "k", "h")
    ck.part_path(d, 0).unlink()
    assert not ck.is_complete(d, "h")
    ck.manifest_path(d).write_text("{not json")
    assert ck.read_manifest(d) is None


def test_plan_is_reused_and_stale_parts_removed(tmp_path):
    d = tmp_path / "p"
    plan = ck.load_or_create_plan(d, "h1", lambda: [[0, 10], [10, 20]])
    ck.atomic_write_parquet(_frame(), ck.part_path(d, 0))
    # Resume with the same config keeps the stored plan even if a new plan would differ.
    assert ck.load_or_create_plan(d, "h1", lambda: [[0, 5]]) == plan
    assert ck.part_path(d, 0).exists()
    # A config change discards this partition's stale parts only.
    assert ck.load_or_create_plan(d, "h2", lambda: [[0, 5]]) == [[0, 5]]
    assert not ck.part_path(d, 0).exists()
    assert json.loads((d / ck.PLAN).read_text())["config_hash"] == "h2"


def test_fingerprint_is_order_independent():
    assert ck.fingerprint({"a": 1, "b": [1, 2]}) == ck.fingerprint({"b": [1, 2], "a": 1})
    assert ck.fingerprint({"a": 1}) != ck.fingerprint({"a": 2})


def test_profiles_and_settings(monkeypatch):
    assert auto_profile(15) == "16gb" and auto_profile(31) == "32gb" and auto_profile(128) == "64gb"
    monkeypatch.setenv("EF_PROFILE", "32gb")
    monkeypatch.setenv("EF_FEATURE_CHUNK", "12345")
    monkeypatch.setenv("EF_N_THREADS", "4")
    s = Settings.from_env()
    assert s.profile == "32gb" and s.n_threads == 4
    assert s.feature_chunk == 12345  # explicit override wins
    assert s.stage1_max_rows == PROFILES["32gb"].stage1_max_rows  # rest from profile
    with pytest.raises(ValueError):
        resolve_profile("7gb")


def test_disk_guard(tmp_path):
    with pytest.raises(OSError):
        ck.require_disk(tmp_path, need_gb=10**6, margin_gb=1, what="test")
    ck.require_disk(tmp_path, need_gb=0, margin_gb=0, what="test")
