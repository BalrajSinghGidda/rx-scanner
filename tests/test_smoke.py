from pathlib import Path


def test_project_structure_exists() -> None:
    """Check that the initial project directories are present."""
    root = Path(__file__).parents[1]
    for path in ("src", "train", "data", "models", "reports"):
        assert (root / path).is_dir()
