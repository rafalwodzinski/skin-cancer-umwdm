import os
import csv
import json
from pathlib import Path
from typing import Dict, Union, Sequence, Optional, Any, List
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


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


class MetricsDict(dict):
    """
    Słownik metryk z obsługą powszechnych aliasów (np. pr_auc, recall_macro).
    """
    _ALIASES = {
        'pr_auc': 'pr_auc_macro',
        'pr_auc_ovr': 'pr_auc_macro',
        'recall_macro': 'sensitivity_macro',
        'roc_auc': 'roc_auc_ovr',
    }

    def __getitem__(self, key: str) -> Any:
        if key not in self and key in self._ALIASES:
            return super().__getitem__(self._ALIASES[key])
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self and key in self._ALIASES:
            return super().get(self._ALIASES[key], default)
        return super().get(key, default)


def compute_medical_metrics(
    y_true: Union[Sequence, np.ndarray],
    y_prob: Union[Sequence, np.ndarray],
    class_names: Sequence[str]
) -> MetricsDict:
    """
    Wyznacza rygorystyczny zestaw metryk diagnostyczno-medycznych dla klasyfikatora zmian skórnych.

    Blokuje ocenę wyłącznie w oparciu o Accuracy, kalkulując zbalansowane metryki
    makro oraz czułość dla kluczowej klinicznie klasy czerniaka (MEL - Melanoma).

    Args:
        y_true (array-like): Wektor rzeczywistych etykiet klas (indeksy 0..C-1 lub nazwy z class_names).
        y_prob (array-like): Macierz predykcji prawdopodobieństw (lub logitów) o kształcie (N, C).
        class_names (Sequence[str]): Lista nazw klas (np. ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']).

    Returns:
        MetricsDict: Słownik zawierający 8 kluczy metryk:
            - 'accuracy': ogólna dokładność klasyfikacji
            - 'sensitivity_macro': średnia czułość (Recall macro)
            - 'specificity_macro': średnia specyficzność One-vs-Rest
            - 'sensitivity_MEL': czułość specyficzna dla klasy czerniaka ('MEL')
            - 'f1_macro': uśredniona miara F1 (macro)
            - 'mcc': współczynnik korelacji Matthewsa
            - 'roc_auc_ovr': pole pod krzywą ROC (One-vs-Rest macro)
            - 'pr_auc_macro': pole pod krzywą Precision-Recall (Average Precision macro)
    """
    class_names_list = list(class_names)
    num_classes = len(class_names_list)

    y_prob_arr = np.asarray(y_prob)
    if y_prob_arr.ndim == 1:
        # Jeśli podano 1D dla binarnej, rozszerz do (N, 2)
        y_prob_arr = np.column_stack([1.0 - y_prob_arr, y_prob_arr])

    # Konwersja y_true do indeksów numerycznych jeśli przekazano napisy
    if len(y_true) > 0 and isinstance(y_true[0], str):
        name_to_idx = {name: i for i, name in enumerate(class_names_list)}
        y_true_arr = np.array([name_to_idx.get(label, 0) for label in y_true], dtype=int)
    else:
        y_true_arr = np.asarray(y_true, dtype=int)

    y_pred_arr = np.argmax(y_prob_arr, axis=1)

    # 1. Dokładność (Accuracy)
    acc = accuracy_score(y_true_arr, y_pred_arr)

    # 2. Czułość (Sensitivity / Recall macro) oraz per-class
    recall_per_class = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    sens_macro = float(np.mean(recall_per_class))

    # 3. Specyficzność (Specificity One-vs-Rest macro)
    spec_macro = calculate_specificity(y_true_arr, y_pred_arr, num_classes=num_classes)

    # 4. Czułość dla czerniaka (MEL)
    if "MEL" in class_names_list:
        mel_idx = class_names_list.index("MEL")
        sens_mel = float(recall_per_class[mel_idx])
    else:
        sens_mel = 0.0

    # 5. F1-macro
    f1 = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0
    )

    # 6. MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

    # 7. ROC-AUC One-vs-Rest & 8. PR-AUC (Average Precision macro)
    y_true_bin = label_binarize(y_true_arr, classes=list(range(num_classes)))
    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    roc_aucs: List[float] = []
    pr_aucs: List[float] = []

    for c in range(num_classes):
        # Sprawdzamy czy w wektorze binarnym występują obie klasy (0 i 1)
        if len(np.unique(y_true_bin[:, c])) > 1:
            roc_aucs.append(float(roc_auc_score(y_true_bin[:, c], y_prob_arr[:, c])))
        if np.sum(y_true_bin[:, c]) > 0:
            pr_aucs.append(float(average_precision_score(y_true_bin[:, c], y_prob_arr[:, c])))

    roc_auc_ovr = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    pr_auc_macro = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    metrics = MetricsDict({
        'accuracy': float(acc),
        'sensitivity_macro': float(sens_macro),
        'specificity_macro': float(spec_macro),
        'sensitivity_MEL': float(sens_mel),
        'f1_macro': float(f1),
        'mcc': float(mcc),
        'roc_auc_ovr': float(roc_auc_ovr),
        'pr_auc_macro': float(pr_auc_macro),
    })

    return metrics


def plot_confusion_matrix(
    y_true: Union[Sequence, np.ndarray],
    y_pred: Union[Sequence, np.ndarray],
    class_names: Sequence[str],
    save_path: Union[str, Path] = "logs/confusion_matrix.png",
    title: str = "Macierz pomyłek (Confusion Matrix)",
    cmap: str = "Blues",
    normalize: bool = False,
) -> str:
    """
    Generuje macierz pomyłek dla wieloklasowej diagnozy zmian skórnych i zapisuje ją jako plik PNG.

    Args:
        y_true (array-like): Wektor rzeczywistych etykiet klas.
        y_pred (array-like): Wektor przewidzianych etykiet (lub macierz prawdopodobieństw shape (N, C)).
        class_names (Sequence[str]): Lista nazw klas do opisania osi.
        save_path (str lub Path): Ścieżka docelowa do zapisu pliku obrazu PNG.
        title (str): Tytuł wyświetlany nad wykresem.
        cmap (str): Paleta barw Seaborn/Matplotlib.
        normalize (bool): Czy wartości mają zostać znormalizowane do proporcji (0.0 - 1.0).

    Returns:
        str: Ścieżka do zapisanego pliku PNG.
    """
    class_names_list = list(class_names)
    num_classes = len(class_names_list)

    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = np.argmax(y_pred_arr, axis=1)

    if len(y_true) > 0 and isinstance(y_true[0], str):
        name_to_idx = {name: i for i, name in enumerate(class_names_list)}
        y_true_arr = np.array([name_to_idx.get(label, 0) for label in y_true], dtype=int)
    else:
        y_true_arr = np.asarray(y_true, dtype=int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        with np.errstate(all='ignore'):
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ax=ax,
        cbar=True
    )

    ax.set_title(title, pad=12, fontsize=12, fontweight='bold')
    ax.set_ylabel("Prawdziwa klasa (True Label)", fontsize=10)
    ax.set_xlabel("Przewidziana klasa (Predicted Label)", fontsize=10)
    plt.tight_layout()

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return str(save_file)


class CSVLogger:
    """
    Lekki moduł logowania metryk treningowych i ewaluacyjnych do pliku CSV (zamiast MLflow).

    Po każdej epoce lub ewaluacji testowej otwiera wskazany plik w trybie dopisywania (mode='a')
    i dodaje nowy wiersz z wynikami.
    """

    def __init__(self, filename: Union[str, Path] = "logs/metrics_history.csv"):
        """
        Inicjalizuje instancję loggera.

        Args:
            filename (str lub Path): Ścieżka do pliku CSV z historią metryk.
        """
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], epoch: Optional[int] = None, **kwargs) -> None:
        """
        Dopisuje nowy wiersz z wynikami do pliku CSV (mode='a').

        Args:
            metrics (dict): Słownik zawierający metryki i ich wartości.
            epoch (int, optional): Numer bieżącej epoki.
            **kwargs: Dodatkowe metadane (np. split='val', learning_rate=1e-4).
        """
        row: Dict[str, Any] = {}
        if epoch is not None and "epoch" not in metrics:
            row["epoch"] = epoch

        row.update(metrics)
        row.update(kwargs)

        file_exists = self.filename.exists() and self.filename.stat().st_size > 0

        # Wczytanie istniejących nagłówków jeśli plik już istnieje
        if file_exists:
            with open(self.filename, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header:
                fieldnames = list(header)
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            else:
                fieldnames = list(row.keys())
        else:
            fieldnames = list(row.keys())

        with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any], **kwargs) -> None:
        """Wygodna metoda logowania konkretnej epoki."""
        self.log(metrics=metrics, epoch=epoch, **kwargs)

    def log_json(self, data: Union[Dict, list], filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Zapisuje strukturę słownika lub listy do pliku .json.

        Args:
            data (dict lub list): Dane do zapisania.
            filepath (str lub Path, optional): Ścieżka docelowa pliku JSON.

        Returns:
            str: Ścieżka do zapisanego pliku JSON.
        """
        if filepath is None:
            filepath = self.filename.with_suffix(".json")
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return str(out_path)
