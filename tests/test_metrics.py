import os
import csv
from pathlib import Path
import pytest
import numpy as np

from src.utils.metrics import (
    calculate_specificity,
    calculate_metrics,
    compute_medical_metrics,
    plot_confusion_matrix,
    CSVLogger,
)


def test_calculate_specificity_exact_math():
    """
    Sprawdza dokładność matematyczną funkcji calculate_specificity
    na ręcznie wyznaczonym przykładzie z 3 klasami.
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    num_classes = 3

    # Z analitycznych obliczeń:
    # Klasa 0: TP=1, FP=1, FN=1, TN=3 -> Spec = 3/(3+1) = 0.75
    # Klasa 1: TP=2, FP=1, FN=0, TN=3 -> Spec = 3/(3+1) = 0.75
    # Klasa 2: TP=1, FP=0, FN=1, TN=4 -> Spec = 4/(4+0) = 1.00
    # Średnia = (0.75 + 0.75 + 1.0) / 3 = 2.5 / 3 ≈ 0.8333333333
    expected_mean_spec = (0.75 + 0.75 + 1.0) / 3.0

    calculated = calculate_specificity(y_true, y_pred, num_classes=num_classes)
    assert pytest.approx(calculated, rel=1e-5) == expected_mean_spec


def test_calculate_specificity_perfect_predictions():
    """
    Dla perfekcyjnych predykcji specyficzność każdej klasy i średnia powinna wynosić dokładnie 1.0.
    """
    y_true = [0, 1, 2, 3, 4, 5, 6, 7]
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7]
    num_classes = 8

    calculated = calculate_specificity(y_true, y_pred, num_classes=num_classes)
    assert calculated == 1.0


def test_calculate_metrics_structure_and_bounds():
    """
    Sprawdza, czy calculate_metrics zwraca kompletny zestaw metryk diagnostycznych
    mieszczących się w poprawnych przedziałach matematycznych.
    """
    y_true = [0, 1, 2, 0, 1, 2, 3, 4]
    y_pred = [0, 1, 1, 0, 1, 2, 2, 4]
    num_classes = 8

    metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)

    required_keys = {'dokładność', 'mcc', 'czułość', 'precyzja', 'specyficzność'}
    assert required_keys.issubset(metrics.keys())

    assert 0.0 <= metrics['dokładność'] <= 1.0
    assert -1.0 <= metrics['mcc'] <= 1.0
    assert 0.0 <= metrics['czułość'] <= 1.0
    assert 0.0 <= metrics['precyzja'] <= 1.0
    assert 0.0 <= metrics['specyficzność'] <= 1.0


def test_calculate_metrics_perfect_score():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]

    metrics = calculate_metrics(y_true, y_pred, num_classes=4)
    assert metrics['dokładność'] == 1.0
    assert metrics['mcc'] == 1.0
    assert metrics['czułość'] == 1.0
    assert metrics['precyzja'] == 1.0
    assert metrics['specyficzność'] == 1.0


def test_compute_medical_metrics_dod():
    """
    Weryfikacja Definition of Done (DoD) dla issue #2:
    Uruchomienie testu na sztucznych predykcjach zwraca słownik z 8 kluczami metryk,
    a na dysku w folderze logs/ pojawia się wygenerowany metrics_history.csv oraz obraz macierzy pomyłek.
    """
    class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    num_classes = len(class_names)
    np.random.seed(42)

    # Sztuczne predykcje
    n_samples = 48
    y_true = np.array([i % num_classes for i in range(n_samples)])
    y_prob = np.random.dirichlet(np.ones(num_classes), size=n_samples)

    # 1. Zwrócenie słownika z 8 kluczami metryk
    metrics = compute_medical_metrics(y_true, y_prob, class_names=class_names)

    assert len(metrics) == 8
    expected_keys = {
        'accuracy',
        'sensitivity_macro',
        'specificity_macro',
        'sensitivity_MEL',
        'f1_macro',
        'mcc',
        'roc_auc_ovr',
        'pr_auc_macro',
    }
    assert set(metrics.keys()) == expected_keys

    # Weryfikacja zakresów wartości
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['sensitivity_macro'] <= 1.0
    assert 0.0 <= metrics['specificity_macro'] <= 1.0
    assert 0.0 <= metrics['sensitivity_MEL'] <= 1.0
    assert 0.0 <= metrics['f1_macro'] <= 1.0
    assert -1.0 <= metrics['mcc'] <= 1.0
    assert 0.0 <= metrics['roc_auc_ovr'] <= 1.0
    assert 0.0 <= metrics['pr_auc_macro'] <= 1.0

    # Obsługa aliasów
    assert metrics['pr_auc'] == metrics['pr_auc_macro']
    assert metrics['recall_macro'] == metrics['sensitivity_macro']

    # 2. Generowanie obrazu macierzy pomyłek w katalogu logs/
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    cm_path = logs_dir / "confusion_matrix.png"

    saved_cm = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_prob,
        class_names=class_names,
        save_path=cm_path
    )
    assert Path(saved_cm).exists()
    assert Path(saved_cm).stat().st_size > 0

    # 3. CSVLogger zapisujący do logs/metrics_history.csv w trybie dopisywania
    csv_path = logs_dir / "metrics_history.csv"
    if csv_path.exists():
        csv_path.unlink()

    logger = CSVLogger(filename=csv_path)
    logger.log(metrics, epoch=1)

    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    # Sprawdzenie trybu append (mode='a')
    logger.log(metrics, epoch=2)
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3  # Nagłówek + 2 wiersze epok
        header = reader[0]
        assert "epoch" in header
        assert "sensitivity_macro" in header
        assert "sensitivity_MEL" in header


def test_sensitivity_mel_calculation_accuracy():
    """
    Testuje wyliczanie czułości specyficznie dla czerniaka (MEL) na kontrolowanych danych.
    """
    class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    mel_idx = class_names.index('MEL')

    # 4 przypadki czerniaka: 3 poprawnie sklasyfikowane, 1 błędnie
    # 2 przypadki innej klasy: poprawnie sklasyfikowane
    y_true = [mel_idx, mel_idx, mel_idx, mel_idx, 0, 1]
    y_pred = [mel_idx, mel_idx, mel_idx, 0,       0, 1]

    # Konstruujemy dummy prawdopodobieństwa
    num_classes = len(class_names)
    y_prob = np.zeros((len(y_true), num_classes))
    for i, pred in enumerate(y_pred):
        y_prob[i, pred] = 1.0

    metrics = compute_medical_metrics(y_true, y_prob, class_names=class_names)

    # TP = 3, FN = 1 -> Sensitivity_MEL = 3 / 4 = 0.75
    assert pytest.approx(metrics['sensitivity_MEL'], rel=1e-5) == 0.75


def test_csv_logger_json_export(tmp_path):
    """
    Weryfikuje zapis metryk do pliku JSON.
    """
    csv_path = tmp_path / "metrics_history.csv"
    logger = CSVLogger(filename=csv_path)

    sample_data = {"epoch": 1, "mcc": 0.85, "sensitivity_MEL": 0.92}
    json_path = tmp_path / "metrics_summary.json"
    logger.log_json(sample_data, filepath=json_path)

    assert json_path.exists()
    assert json_path.stat().st_size > 0
