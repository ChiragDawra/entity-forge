import polars as pl

from entity_forge.decision import resolve_exclusive, select_expected_f, select_threshold, tune_threshold


def scored(rows):
    return pl.DataFrame(rows, schema={"s1_id": pl.String, "t_id": pl.String, "p": pl.Float64}, orient="row")


def test_exclusive_keeps_best_s1_per_target():
    pred = scored([("A", "x", 0.9), ("B", "x", 0.6), ("B", "y", 0.8), ("A", "z", 0.3), ("C", "z", 0.3)])
    out = resolve_exclusive(pred).sort("t_id")
    assert out.select("s1_id", "t_id").rows() == [("A", "x"), ("B", "y"), ("A", "z")]


def test_threshold_allows_empty_sets():
    pred = scored([("A", "x", 0.9), ("B", "y", 0.2)])
    assert select_threshold(pred, 0.5)["s1_id"].to_list() == ["A"]


def test_expected_f_prefers_empty_when_all_unlikely():
    pred = scored([("A", "x", 0.05), ("A", "y", 0.04)])
    assert select_expected_f(pred).height == 0


def test_expected_f_takes_confident_prefix():
    pred = scored([("A", "x", 0.97), ("A", "y", 0.95), ("A", "z", 0.10)])
    assert sorted(select_expected_f(pred)["t_id"].to_list()) == ["x", "y"]


def test_tune_threshold_finds_separating_cut():
    pred = scored([("A", "x", 0.9), ("A", "n", 0.3), ("B", "m", 0.35)])
    truth = pl.DataFrame({"s1_id": ["A"], "t_id": ["x"]})
    tau, table = tune_threshold(pred, truth, pl.Series(["A", "B"]))
    assert 0.35 < tau <= 0.9
    assert table["macro_f05"].max() == 1.0
