import polars as pl
import pytest

from entity_forge.metrics import fbeta, macro_fbeta, per_entity_scores, summary


def pairs(rows):
    return pl.DataFrame(rows, schema={"s1_id": pl.String, "t_id": pl.String}, orient="row")


EMPTY = pairs([])


def test_scalar_cases():
    assert fbeta(0, 0, 0) == 1.0  # true empty, predicted empty
    assert fbeta(0, 2, 0) == 0.0  # true empty, predicted non-empty
    assert fbeta(0, 0, 3) == 0.0  # true non-empty, predicted empty
    assert fbeta(2, 2, 2) == 1.0  # exact overlap
    # Problem-statement example: P = 2/3, R = 1 -> 0.714
    assert fbeta(2, 3, 2) == pytest.approx(0.7142857, rel=1e-6)


def test_problem_statement_example_vectorized():
    pred = pairs([("S1-1", "S2-47"), ("S1-1", "S2-193"), ("S1-1", "S3-812")])
    truth = pairs([("S1-1", "S2-47"), ("S1-1", "S3-812")])
    assert macro_fbeta(pl.Series(["S1-1"]), pred, truth) == pytest.approx(0.7142857, rel=1e-6)


def test_singletons_and_missing_rows_count():
    ids = pl.Series(["A", "B", "C", "D"])
    truth = pairs([("A", "x"), ("C", "y")])  # B and D are singletons
    pred = pairs([("A", "x"), ("B", "z")])  # C missed entirely, D correctly empty
    scores = per_entity_scores(ids, pred, truth).sort("s1_id")["f"].to_list()
    assert scores == [1.0, 0.0, 0.0, 1.0]
    assert macro_fbeta(ids, pred, truth) == pytest.approx(0.5)


def test_duplicate_predictions_are_ignored():
    ids = pl.Series(["A"])
    truth = pairs([("A", "x")])
    pred = pairs([("A", "x"), ("A", "x")])
    assert macro_fbeta(ids, pred, truth) == 1.0


def test_partial_overlap_prefers_precision():
    ids = pl.Series(["A"])
    truth = pairs([("A", "x"), ("A", "y"), ("A", "z"), ("A", "w")])
    one_right = pairs([("A", "x")])  # P=1, R=0.25
    all_plus_noise = pairs([("A", t) for t in ["x", "y", "z", "w", "n1", "n2", "n3", "n4"]])
    assert macro_fbeta(ids, one_right, truth) == pytest.approx(fbeta(1, 1, 4))
    assert macro_fbeta(ids, all_plus_noise, truth) == pytest.approx(fbeta(4, 8, 4))


def test_summary_fields():
    ids = pl.Series(["A", "B"])
    out = summary(ids, EMPTY, pairs([("A", "x")]))
    assert out["macro_f05"] == 0.5
    assert out["singleton_accuracy"] == 1.0
    assert out["micro_recall"] == 0.0
