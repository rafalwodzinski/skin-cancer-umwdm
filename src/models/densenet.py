import os
from typing import Optional, Union
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 8


def get_model(
    num_classes: int = NUM_CLASSES,
    device: Optional[Union[torch.device, str]] = None,
    weights: Optional[Union[str, models.DenseNet121_Weights]] = "DEFAULT",
    drop_rate: float = 0.2,
    dropout_p: float = 0.5
) -> nn.Module:
    """
    Inicjalizuje architekturę DenseNet-121 i dostosowuje ją do zadania klasyfikacji zmian skórnych.

    Funkcja realizuje strategię uczenia transferowego:
    1. Pobiera wagi wstępnie wytrenowane na zbiorze ImageNet (lub inicjalizuje losowo przy weights=None).
    2. Wprowadza mechanizm regularyzacji wewnątrz bloków zagęszczonych (drop_rate).
    3. Zastępuje oryginalny klasyfikator z dodatkowym Dropoutem,
       dostosowany do specyficznej liczby klas zbioru ISIC.

    Args:
        num_classes (int): Liczba klas wyjściowych (domyślnie 8).
        device (torch.device lub str, optional): Urządzenie obliczeniowe, na które zostanie przeniesiony model.
        weights (str lub DenseNet121_Weights, optional): Wagi wstępne ('DEFAULT', None lub instancja wag).
        drop_rate (float): Wskaźnik wyłączania neuronów wewnątrz bloków splotowych (domyślnie 0.2).
        dropout_p (float): Wskaźnik Dropoutu w nowej głowicy klasyfikacyjnej (domyślnie 0.5).

    Returns:
        torch.nn.Module: Skompilowany model PyTorch z nową głowicą klasyfikacyjną.
    """
    # pobieramy model z wytrenowanymi wagami (lub pusty jeśli weights=None)
    # wprowadzamy wyłączanie neuronów w drop_rate
    model = models.densenet121(weights=weights, drop_rate=drop_rate)

    # pobieramy liczbę cech wchodzących do ostatniej warstwy (dla DenseNet121 to 1024)
    in_features = model.classifier.in_features

    # podmieniamy warstwę klasyfikacyjną (dodajemy Dropout)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes)
    )

    # przenosimy model na docelowe urządzenie
    if device is not None:
        model = model.to(device)

    return model


def get_densenet_extractor(
    weights_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    num_classes: int = NUM_CLASSES,
    drop_rate: float = 0.2
) -> nn.Module:
    """
    Wczytuje model DenseNet-121 i konwertuje go na ekstraktor cech.

    Funkcja najpierw odtwarza architekturę użytą podczas treningu (aby poprawnie
    wczytać wagi), a następnie zamienia warstwę klasyfikacyjną na funkcję tożsamościową,
    dzięki czemu model zwraca wektory cech (embeddings) zamiast predykcji klas.

    Args:
        weights_path (str, optional): Ścieżka do pliku .pth z zapisanymi wagami modelu.
        device (torch.device lub str, optional): Urządzenie (CPU/GPU/MPS), na które ma trafić model.
        num_classes (int): Liczba klas użyta przy treningu modelu (domyślnie 8).
        drop_rate (float): Wskaźnik dropoutu użyty w architekturze bazowej.

    Returns:
        torch.nn.Module: Model w trybie ewaluacji, zwracający wektor cech o wymiarze 1024.
    """
    # inicjalizujemy pusty model
    model = models.densenet121(weights=None, drop_rate=drop_rate)
    in_features = model.classifier.in_features

    # odtwarzamy warstwę klasyfikacyjną z notatnika z treningu
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    # wczytujemy wagi, jeśli podano ścieżkę
    if weights_path is not None:
        map_location = device if device is not None else "cpu"
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)

    # modyfikujemy warstwę klasyfikacyjną, aby zwracała ostatni wynik warstw splotowych
    model.classifier = nn.Identity()

    # przenosimy na docelowe urządzenie i przełączamy w tryb ewaluacji
    if device is not None:
        model = model.to(device)
    model.eval()

    return model
