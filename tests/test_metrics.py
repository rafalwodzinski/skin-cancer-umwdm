import pytest
import numpy as np
from src.utils.metrics import calculate_specificity, calculate_metrics


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
