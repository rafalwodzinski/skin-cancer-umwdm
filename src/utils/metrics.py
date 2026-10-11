from typing import Dict, Union, Sequence
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
)


def calculate_specificity(
    y_true: Union[Sequence, np.ndarray],
    y_pred: Union[Sequence, np.ndarray],
    num_classes: int
) -> float:
    """
    Oblicza średnią specyficzność dla klasyfikacji wieloklasowej.

    Funkcja wykorzystuje strategię One-vs-Rest, obliczając specyficzność niezależnie
    dla każdej klasy (binarna klasyfikacja: dana klasa vs reszta), a następnie zwraca
    ich średnią arytmetyczną. Jest to kluczowa metryka w diagnostyce,
    określająca zdolność modelu do poprawnego wykluczania danej choroby.

    Args:
        y_true (array-like): Wektor rzeczywistych etykiet.
        y_pred (array-like): Wektor przewidzianych etykiet.
        num_classes (int): Całkowita liczba klas w modelu.

    Returns:
        float: Uśredniona wartość specyficzności (zakres 0.0 - 1.0).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_per_class = []

    for i in range(num_classes):
        # dla klasy 'i':
        # TN = suma wszystkich próbek, które nie są 'i' i nie zostały przewidziane jako 'i'
        # FP = próbki, które nie są 'i', ale zostały przewidziane jako 'i'

        # wartości w macierzy pomyłek:
        # cm[i, i] to TP
        # cm[:, i] to kolumna predykcji (suma to TP + FP) -> FP = suma_kol - TP
        # cm[i, :] to wiersz prawdy (suma to TP + FN)
        # suma całkowita to total

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        # Specyficzność = TN / (TN + FP) z mechanizmem uniknięcia dzielenia przez 0
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = 0.0
        specificity_per_class.append(spec)

    return float(np.mean(specificity_per_class))


def calculate_metrics(
    y_true: Union[Sequence, np.ndarray],
    y_pred: Union[Sequence, np.ndarray],
    num_classes: int = 8
) -> Dict[str, float]:
    """
    Oblicza kompletny zestaw metryk diagnostycznych dla ewaluacji klasyfikatora:
    - dokładność (Accuracy)
    - mcc (Matthews Correlation Coefficient)
    - czułość (Sensitivity / Recall macro)
    - precyzja (Precision macro)
    - specyficzność (Specificity One-vs-Rest macro)

    Args:
        y_true (array-like): Wektor rzeczywistych etykiet.
        y_pred (array-like): Wektor przewidzianych etykiet.
        num_classes (int): Liczba klas w zadaniu klasyfikacji.

    Returns:
        dict: Słownik zawierający wartości metryk.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
    sens = recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    prec = precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    spec = calculate_specificity(y_true_arr, y_pred_arr, num_classes)

    return {
        'dokładność': float(acc),
        'mcc': float(mcc),
        'czułość': float(sens),
        'precyzja': float(prec),
        'specyficzność': float(spec),
    }
