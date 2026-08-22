"""Tests for deterministic weighted-XGBoost search behavior."""

from road_accidents.tuning import generate_candidates, select_winner


def test_candidate_generation_is_deterministic_and_in_range():
    first = generate_candidates(12)
    second = generate_candidates(12)
    assert first == second
    assert len(first) == 12
    assert len({tuple(sorted(candidate.items())) for candidate in first}) == 12
    for candidate in first:
        assert 3 <= candidate["max_depth"] <= 8
        assert 0.015 <= candidate["learning_rate"] <= 0.15
        assert 0.6 <= candidate["subsample"] <= 1.0
        assert 0.6 <= candidate["colsample_bytree"] <= 1.0


def _result(candidate, macro, fatal, std):
    return {
        "candidate": candidate,
        "summary": {
            "macro_f1": {"mean": macro, "std": std},
            "per_class": {"Fatal": {"f1": {"mean": fatal}}},
        },
    }


def test_winner_uses_macro_f1_before_fatal_f1():
    results = [_result(1, 0.35, 0.20, 0.01), _result(2, 0.36, 0.05, 0.02)]
    assert select_winner(results)["candidate"] == 2


def test_winner_tie_breaks_on_fatal_then_stability_then_order():
    results = [
        _result(1, 0.36, 0.08, 0.02),
        _result(2, 0.36, 0.09, 0.03),
        _result(3, 0.36, 0.09, 0.01),
        _result(4, 0.36, 0.09, 0.01),
    ]
    assert select_winner(results)["candidate"] == 3


def test_learning_rate_upper_bound_is_not_exceeded():
    assert max(c["learning_rate"] for c in generate_candidates(100)) <= 0.15
