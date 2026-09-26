from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and thresholds used by the entity-resolution pipeline."""

    project_root: Path = Path(__file__).resolve().parents[2]
    dataset_dir: Path | None = None
    output_dir: Path | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset_dir is None:
            object.__setattr__(self, "dataset_dir", self.project_root / "dataset")
        if self.output_dir is None:
            object.__setattr__(self, "output_dir", self.project_root / "output")

    @property
    def train_path(self) -> Path:
        return self.dataset_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.dataset_dir / "test.csv"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.joblib"

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / "predictions.csv"
