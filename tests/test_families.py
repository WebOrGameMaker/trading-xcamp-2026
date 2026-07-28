"""Tests for feature family maps and Experiment 2 subset helpers."""

import pytest

from src.features.families import (
    ALL_STANDARD_FEATURES,
    FEATURE_FAMILIES,
    STAGE_A_ARMS,
    cumulative_importance_features,
    normalize_importances,
    resolve_feature_arm,
    resolve_prune_arm,
    top_k_features,
)


def test_all_standard_features_cover_families_without_duplicates() -> None:
    flat = [col for cols in FEATURE_FAMILIES.values() for col in cols]
    assert len(flat) == len(set(flat))
    assert tuple(flat) == ALL_STANDARD_FEATURES
    assert len(ALL_STANDARD_FEATURES) == 19


def test_resolve_feature_arm_full_and_families() -> None:
    assert resolve_feature_arm("full") == list(ALL_STANDARD_FEATURES)
    assert resolve_feature_arm("returns") == list(FEATURE_FAMILIES["returns"])
    assert resolve_feature_arm("returns_volatility") == (
        list(FEATURE_FAMILIES["returns"]) + list(FEATURE_FAMILIES["volatility"])
    )
    for arm in STAGE_A_ARMS:
        cols = resolve_feature_arm(arm)
        assert cols
        assert len(cols) == len(set(cols))


def test_resolve_feature_arm_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown feature arm"):
        resolve_feature_arm("not_an_arm")


def test_top_k_features_orders_by_importance() -> None:
    importance = {"a": 0.1, "b": 0.5, "c": 0.4}
    assert top_k_features(importance, 2) == ["b", "c"]
    assert top_k_features(importance, 10) == ["b", "c", "a"]


def test_cumulative_importance_features_covers_threshold() -> None:
    importance = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert cumulative_importance_features(importance, 0.8) == ["a", "b"]
    assert cumulative_importance_features(importance, 1.0) == ["a", "b", "c"]


def test_normalize_importances_sums_to_one() -> None:
    normalized = normalize_importances({"x": 2.0, "y": 2.0})
    assert normalized["x"] == pytest.approx(0.5)
    assert sum(normalized.values()) == pytest.approx(1.0)


def test_resolve_prune_arm() -> None:
    importance = {f"f{i}": float(20 - i) for i in range(12)}
    assert len(resolve_prune_arm("top5", importance)) == 5
    assert len(resolve_prune_arm("top10", importance)) == 10
    cum80 = resolve_prune_arm("cum80", importance)
    assert 1 <= len(cum80) <= 12
