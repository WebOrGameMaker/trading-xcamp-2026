"""Model persistence and registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.utils.paths import MODEL_DIR

POOLED_SYMBOL = "pooled"


@dataclass
class ModelArtifact:
    """Metadata for a saved model artifact."""

    symbol: str
    model_path: str
    metadata_path: str
    model_type: str
    feature_columns: list[str]
    train_rows: int
    val_rows: int
    test_rows: int
    created_at: str
    metrics: dict[str, Any]


def save_model(
    pipeline: Any,
    symbol: str,
    model_type: str,
    feature_columns: list[str],
    train_rows: int,
    val_rows: int,
    test_rows: int,
    metrics: dict[str, Any],
) -> ModelArtifact:
    """Persist trained pipeline and metadata to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        symbol: Ticker whose observations trained the model, or ``pooled``.
        model_type: Model identifier string.
        feature_columns: List of feature column names used in training.
        train_rows: Number of training samples.
        val_rows: Number of validation samples.
        test_rows: Number of test samples.
        metrics: Evaluation metrics dictionary.

    Returns:
        ModelArtifact with paths and metadata.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    symbol_dir = MODEL_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    model_path = symbol_dir / f"{model_type}_{timestamp}.joblib"
    metadata_path = symbol_dir / f"{model_type}_{timestamp}.json"

    joblib.dump(pipeline, model_path)

    artifact = ModelArtifact(
        symbol=symbol,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        model_type=model_type,
        feature_columns=feature_columns,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        created_at=timestamp,
        metrics=metrics,
    )

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifact), handle, indent=2)

    return artifact


def save_model_manifest(
    artifacts: list[ModelArtifact],
    model_type: str,
    run_id: str,
    metrics: dict[str, Any],
    scope: str = "pooled",
) -> Path:
    """Persist the manifest for a training run.

    Args:
        artifacts: Saved model artifacts for the run.
        model_type: Classifier family used for the run.
        run_id: Unique identifier shared by all models in the run.
        metrics: Aggregate evaluation metrics.
        scope: Training scope (``pooled`` or ``per_symbol``).

    Returns:
        Path to the latest-run manifest.
    """
    if not artifacts:
        raise ValueError("Cannot save an empty model manifest")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": scope,
        "model_type": model_type,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol_count": len(artifacts) if scope == "per_symbol" else 1,
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "metrics": metrics,
    }
    latest_path = MODEL_DIR / "latest.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return latest_path


def load_latest_model() -> tuple[Any, ModelArtifact]:
    """Load the latest pooled (or single) model artifact.

    Returns:
        Fitted pipeline and artifact metadata.

    Raises:
        FileNotFoundError: If no model has been saved.
        ValueError: If the manifest has no artifacts.
    """
    latest_path = MODEL_DIR / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError("No trained model found. Run 'python main.py train' first.")

    with latest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    artifacts = raw.get("artifacts", [])
    if not artifacts:
        raise ValueError("Latest model manifest contains no model artifacts")

    # Prefer the pooled artifact when present; otherwise take the first entry.
    artifact_data = next(
        (item for item in artifacts if item.get("symbol") == POOLED_SYMBOL),
        artifacts[0],
    )
    artifact = ModelArtifact(**artifact_data)
    return joblib.load(artifact.model_path), artifact


def load_latest_models() -> dict[str, tuple[Any, ModelArtifact]]:
    """Load the latest model collection.

    For pooled runs returns a single-entry mapping keyed by ``pooled``.
    For legacy per-symbol runs returns one entry per ticker.

    Returns:
        Mapping from ticker (or ``pooled``) to fitted pipeline and metadata.

    Raises:
        FileNotFoundError: If no model has been saved.
    """
    latest_path = MODEL_DIR / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError("No trained model found. Run 'python main.py train' first.")

    with latest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    models: dict[str, tuple[Any, ModelArtifact]] = {}
    for artifact_data in raw.get("artifacts", []):
        artifact = ModelArtifact(**artifact_data)
        models[artifact.symbol] = (joblib.load(artifact.model_path), artifact)

    if not models:
        raise ValueError("Latest model manifest contains no model artifacts")
    return models
