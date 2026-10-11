import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.data.dataset import ISICDataset, get_val_transforms
from src.models.densenet import get_model, NUM_CLASSES
from src.models.trainer import Trainer, get_loss


def test_get_loss_weights_and_label_smoothing():
    # Przygotowanie ramki danych z niezbalansowanymi klasami
    targets = [0, 0, 0, 0, 1, 1, 2]
    df = pd.DataFrame({'target': targets})
    device = torch.device('cpu')

    # Test dla 3 klas
    criterion = get_loss(df, device, num_classes=3)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1

    # Sprawdzenie obliczeń wag: w = 1 / sqrt(count), normalizowane przez średnią
    counts = np.array([4, 2, 1])
    raw_weights = 1.0 / np.sqrt(counts)
    expected_weights = raw_weights / np.mean(raw_weights)

    actual_weights = criterion.weight.cpu().numpy()
    assert np.allclose(actual_weights, expected_weights, atol=1e-5)


def test_trainer_setup_stages():
    model = get_model(num_classes=NUM_CLASSES, weights=None)
    dummy_loader = DataLoader([1], batch_size=1)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        criterion=criterion,
        device="cpu"
    )

    # Etap 1: Trening klasyfikatora
    opt1, sched1, epochs1, pat1 = trainer.setup_stage('trening klasyfikatora')
    assert sched1 is None
    for param in model.features.parameters():
        assert not param.requires_grad
    for param in model.classifier.parameters():
        assert param.requires_grad

    # Etap 2: Dostrajanie modelu
    opt2, sched2, epochs2, pat2 = trainer.setup_stage('dostrajanie modelu')
    assert sched2 is not None
    assert any(p.requires_grad for p in model.features.denseblock4.parameters())
    assert any(p.requires_grad for p in model.features.denseblock3.parameters())


def test_trainer_train_and_evaluate_step(mock_dataset_dir, tmp_path):
    img_dir, df = mock_dataset_dir
    transform = get_val_transforms(img_size=100)

    dataset = ISICDataset(dataframe=df, root_dir=str(img_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = get_model(num_classes=NUM_CLASSES, weights=None)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        num_classes=NUM_CLASSES,
        save_dir=str(tmp_path),
        warmup_epochs=1,
        finetune_epochs=1,
        use_amp=False
    )

    opt, _, _, _ = trainer.setup_stage('trening klasyfikatora')
    loss = trainer.train_one_epoch(opt)
    assert isinstance(loss, float)
    assert loss > 0.0

    val_metrics = trainer.evaluate()
    assert 'strata' in val_metrics
    assert 'mcc' in val_metrics
    assert 'dokładność' in val_metrics


def test_cli_train_dry_run(mock_dataset_dir, tmp_path):
    """
    Weryfikuje uruchomienie skryptu train.py z poziomu interpretera CLI w trybie dry-run.
    """
    import subprocess
    import sys

    img_dir, df = mock_dataset_dir
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "results"

    cmd = [
        sys.executable,
        "train.py",
        "--metadata-path", str(csv_path),
        "--images-dir", str(img_dir),
        "--output-dir", str(out_dir),
        "--fold", "0",
        "--batch-size", "2",
        "--num-workers", "0",
        "--dry-run",
        "--no-pretrained",
        "--device", "cpu"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Błąd wykonania CLI train.py: {result.stderr}"

    # Weryfikacja czy powstały pliki wynikowe
    history_file = out_dir / "DenseNet-121_wyniki_treningu_fold_0.json"
    preds_file = out_dir / "DenseNet-121_predykcje_fold_0.json"
    assert history_file.exists()
    assert preds_file.exists()

