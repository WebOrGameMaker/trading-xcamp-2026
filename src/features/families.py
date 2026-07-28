"""Feature family maps and subset helpers for Experiment 2.

Groups the standard 19 technical indicators into economic families and
resolves named experiment arms (family ablation + importance pruning).
"""

from __future__ import annotations

from typing import Mapping

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "returns": (
        "return_1d",
        "return_5d",
        "return_20d",
    ),
    "trend": (
        "price_sma10_ratio",
        "price_sma20_ratio",
        "price_sma50_ratio",
        "price_ema12_ratio",
        "price_ema26_ratio",
        "macd_pct",
        "macd_signal_pct",
        "macd_hist_pct",
    ),
    "momentum": (
        "rsi_14",
        "stoch_k",
        "stoch_d",
    ),
    "volatility": (
        "bb_width",
        "atr_pct",
        "volatility_20d",
    ),
    "volume": (
        "obv_zscore_60",
        "volume_sma_ratio",
    ),
}

STAGE_A_ARMS: tuple[str, ...] = (
    "full",
    "returns",
    "trend",
    "momentum",
    "volatility",
    "volume",
    "returns_volatility",
)

STAGE_B_ARMS: tuple[str, ...] = (
    "top5",
    "top10",
    "cum80",
)

ALL_STANDARD_FEATURES: tuple[str, ...] = tuple(
    col for family in FEATURE_FAMILIES.values() for col in family
)


def resolve_feature_arm(arm_name: str) -> list[str]:
    """Resolve a Stage A arm name to an ordered feature column list.

    Args:
        arm_name: One of ``STAGE_A_ARMS``.

    Returns:
        Feature column names for the arm.

    Raises:
        ValueError: If ``arm_name`` is unknown.
    """
    if arm_name == "full":
        return list(ALL_STANDARD_FEATURES)
    if arm_name == "returns_volatility":
        return list(FEATURE_FAMILIES["returns"]) + list(FEATURE_FAMILIES["volatility"])
    if arm_name in FEATURE_FAMILIES:
        return list(FEATURE_FAMILIES[arm_name])
    raise ValueError(
        f"Unknown feature arm: {arm_name!r}. "
        f"Expected one of {STAGE_A_ARMS}."
    )


def normalize_importances(importance: Mapping[str, float]) -> dict[str, float]:
    """L1-normalize feature importances so they sum to 1 (or all zeros)."""
    total = float(sum(float(v) for v in importance.values()))
    if total <= 0:
        return {str(k): 0.0 for k in importance}
    return {str(k): float(v) / total for k, v in importance.items()}


def top_k_features(importance: Mapping[str, float], k: int) -> list[str]:
    """Return the top-``k`` features by importance (descending).

    Args:
        importance: Feature name → importance score.
        k: Number of features to keep (must be >= 1).

    Returns:
        Ordered list of feature names (highest importance first).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranked = sorted(importance.items(), key=lambda kv: float(kv[1]), reverse=True)
    return [str(name) for name, _ in ranked[:k]]


def cumulative_importance_features(
    importance: Mapping[str, float],
    threshold: float = 0.8,
) -> list[str]:
    """Smallest prefix of ranked features whose normalized importance sums to threshold.

    Args:
        importance: Feature name → importance score.
        threshold: Cumulative normalized mass to cover (in (0, 1]).

    Returns:
        Ordered list of feature names (highest importance first).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if not importance:
        return []

    normalized = normalize_importances(importance)
    ranked = sorted(normalized.items(), key=lambda kv: float(kv[1]), reverse=True)
    selected: list[str] = []
    cumulative = 0.0
    for name, weight in ranked:
        selected.append(str(name))
        cumulative += float(weight)
        if cumulative >= threshold:
            break
    return selected


def resolve_prune_arm(
    arm_name: str,
    importance: Mapping[str, float],
) -> list[str]:
    """Resolve a Stage B prune arm from Full-model importance.

    Args:
        arm_name: One of ``STAGE_B_ARMS``.
        importance: Feature importances from the Full arm.

    Returns:
        Feature column list for the prune arm.
    """
    if arm_name == "top5":
        return top_k_features(importance, 5)
    if arm_name == "top10":
        return top_k_features(importance, 10)
    if arm_name == "cum80":
        return cumulative_importance_features(importance, threshold=0.8)
    raise ValueError(
        f"Unknown prune arm: {arm_name!r}. Expected one of {STAGE_B_ARMS}."
    )
