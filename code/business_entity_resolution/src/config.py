from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and thresholds used by the entity-resolution pipeline."""

    project_root: Path = Path(__file__).resolve().parents[3]
    dataset_dir: Path | None = None
    output_dir: Path | None = None
    threshold: float = 0.80

    def __post_init__(self) -> None:
        if self.dataset_dir is None:
            # Check primary dataset dir first, fallback to student_resource dataset dir
            primary = self.project_root / "dataset"
            fallback = (
                self.project_root
                / "6ab10eb3b23ba_student_resource"
                / "student_resource"
                / "dataset"
            )
            if (primary / "train").exists() or (primary / "test").exists():
                object.__setattr__(self, "dataset_dir", primary)
            elif (fallback / "train").exists() or (fallback / "test").exists():
                object.__setattr__(self, "dataset_dir", fallback)
            else:
                object.__setattr__(self, "dataset_dir", primary)

        if self.output_dir is None:
            object.__setattr__(self, "output_dir", self.project_root / "output")

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "test"

    @property
    def train_s1_path(self) -> Path:
        return self.train_dir / "train_source1.tsv"

    @property
    def train_s2_path(self) -> Path:
        return self.train_dir / "train_source2.tsv"

    @property
    def train_s3_path(self) -> Path:
        return self.train_dir / "train_source3.tsv"

    @property
    def train_gt_path(self) -> Path:
        return self.train_dir / "train_ground_truth.tsv"

    @property
    def test_s1_path(self) -> Path:
        return self.test_dir / "test_source1.tsv"

    @property
    def test_s2_path(self) -> Path:
        return self.test_dir / "test_source2.tsv"

    @property
    def test_s3_path(self) -> Path:
        return self.test_dir / "test_source3.tsv"

    @property
    def matching_output_path(self) -> Path:
        return self.output_dir / "matching_results.tsv"

    @property
    def candidate_output_path(self) -> Path:
        return self.output_dir / "candidate_pairs.tsv"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.joblib"

