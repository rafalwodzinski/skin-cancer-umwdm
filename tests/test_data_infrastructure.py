from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.utils.device_check import get_device
from src.data.download_data import download_raw_data, PROJECT_ROOT
from src.utils.artifact_sync import download_artifacts, upload_artifacts


def test_get_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "mps", "cpu"]


def test_download_raw_data_skips_if_already_present(tmp_path):
    # Utworzenie folderu z plikiem testowym
    test_dir = tmp_path / "raw"
    test_dir.mkdir(parents=True)
    (test_dir / "test_file.csv").write_text("dummy")

    with patch("src.data.download_data.init_kaggle_api") as mock_init:
        result = download_raw_data(download_dir=str(test_dir), force=False)
        assert result == test_dir
        mock_init.assert_not_called()


def test_download_raw_data_calls_kaggle_cli(tmp_path):
    test_dir = tmp_path / "raw"
    test_dir.mkdir(parents=True)
    # Plik .gitkeep nie powinien blokować pobierania
    (test_dir / ".gitkeep").write_text("")

    mock_api = MagicMock()
    with patch("src.data.download_data.init_kaggle_api", return_value=mock_api):
        result = download_raw_data(dataset="test-team/test-dataset", download_dir=str(test_dir), force=False)
        assert result == test_dir
        mock_api.dataset_download_cli.assert_called_once_with(
            dataset="test-team/test-dataset",
            path=str(test_dir),
            unzip=True,
            force=False,
            quiet=False,
        )


def test_download_artifacts_calls_kaggle_cli(tmp_path):
    test_dir = tmp_path / "models"
    test_dir.mkdir(parents=True)

    mock_api = MagicMock()
    with patch("src.utils.artifact_sync.init_kaggle_api", return_value=mock_api):
        result = download_artifacts(download_dir=str(test_dir), dataset="team/models")
        assert result == test_dir
        mock_api.dataset_download_cli.assert_called_once_with(
            dataset="team/models",
            path=str(test_dir),
            unzip=True,
            force=True,
            quiet=False,
        )


def test_upload_artifacts_creates_metadata_and_calls_version(tmp_path):
    test_dir = tmp_path / "models"
    test_dir.mkdir(parents=True)
    (test_dir / "best_model.pth").write_text("weights")

    mock_api = MagicMock()
    with patch("src.utils.artifact_sync.init_kaggle_api", return_value=mock_api):
        upload_artifacts(folder_to_upload=str(test_dir), version_notes="Test version", dataset="team/model-repo")

        # Plik dataset-metadata.json powinien zostać utworzony
        meta_file = test_dir / "dataset-metadata.json"
        assert meta_file.is_file()
        assert "team/model-repo" in meta_file.read_text()

        mock_api.dataset_create_version.assert_called_once_with(
            str(test_dir),
            version_notes="Test version",
            dir_mode="zip",
            quiet=False,
        )


def test_upload_artifacts_raises_when_empty_folder(tmp_path):
    test_dir = tmp_path / "empty_models"
    test_dir.mkdir(parents=True)
    (test_dir / ".gitkeep").write_text("")

    with pytest.raises(ValueError, match="nie zawiera żadnych plików"):
        upload_artifacts(folder_to_upload=str(test_dir))
