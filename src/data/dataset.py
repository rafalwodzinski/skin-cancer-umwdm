import os
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

# Parametry transformacji obrazów
IMG_SIZE = 300                      # wymiar wejściowy obrazu
NORM_MEAN = [0.485, 0.456, 0.406]   # wartości średniej do normalizacji
NORM_STD = [0.229, 0.224, 0.225]    # wartości odchyleń do normalizacji


class ISICDataset(Dataset):
    """
    Niestandardowy Dataset PyTorch do wczytywania obrazów dermatologicznych ISIC.

    Klasa wczytuje obrazy na podstawie ścieżek z DataFrame, konwertuje je do RGB
    i zwraca w formie gotowej do przetworzenia przez model.
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform=None):
        """
        Inicjalizuje dataset.

        Args:
            dataframe (pd.DataFrame): Ramka danych zawierająca co najmniej kolumny:
                                      - 'nazwa_pliku': nazwa pliku z rozszerzeniem (np. 'IMG_1.jpg')
                                      - 'target': numeryczna etykieta klasy (int).
            root_dir (str): Ścieżka do katalogu głównego, w którym znajdują się obrazy.
            transform (callable, optional): Opcjonalne transformacje (np. augmentacja, normalizacja)
                                            aplikowane na obrazie. Domyślnie None.
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę próbek w zbiorze."""
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Pobiera pojedynczą próbkę danych o podanym indeksie.

        Args:
            idx (int/tensor): Indeks próbki do pobrania.

        Returns:
            tuple: Krotka (image, label), gdzie:
                   - image (PIL.Image lub Tensor): Obraz po transformacjach.
                   - label (int): Numeryczna etykieta klasy.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pobranie nazwy pliku
        img_name = self.df.iloc[idx]['nazwa_pliku']
        img_path = os.path.join(self.root_dir, img_name)

        # wczytanie obrazu i konwersja na RGB (dla bezpieczeństwa)
        image = Image.open(img_path).convert('RGB')

        # pobranie etykiety numerycznej (target)
        label = int(self.df.iloc[idx]['target'])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(img_size: int = IMG_SIZE, mean=NORM_MEAN, std=NORM_STD) -> transforms.Compose:
    """
    Tworzy pipeline transformacji treningowych z augmentacją danych.

    Args:
        img_size (int): Docelowy rozmiar boku obrazu kwadratowego.
        mean (list): Średnie kanałów RGB dla normalizacji.
        std (list): Odchylenia standardowe kanałów RGB dla normalizacji.

    Returns:
        transforms.Compose: Kompozycja transformacji Torchvision.
    """
    working_size = int(img_size * 1.25)

    return transforms.Compose([
        # losujemy wycinek od 80% do 100% powierzchni oryginalnego obrazu
        transforms.RandomResizedCrop(working_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),

        # losowe odbicia lustrzane pionowo i poziomo
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        # delikatna zmiana koloru
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),

        # obrót zdjęcia o kąt z przedziału (-15 stopni, 15 stopni)
        transforms.RandomRotation(15),

        # wycinamy środek obrazu w docelowych wymiarach
        transforms.CenterCrop(img_size),

        # przeniesienie na tensor i normalizacja
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_val_transforms(img_size: int = IMG_SIZE, mean=NORM_MEAN, std=NORM_STD) -> transforms.Compose:
    """
    Tworzy pipeline transformacji walidacyjnych/testowych bez losowej augmentacji.

    Args:
        img_size (int): Docelowy rozmiar boku obrazu kwadratowego.
        mean (list): Średnie kanałów RGB dla normalizacji.
        std (list): Odchylenia standardowe kanałów RGB dla normalizacji.

    Returns:
        transforms.Compose: Kompozycja transformacji Torchvision.
    """
    working_size = int(img_size * 1.25)

    return transforms.Compose([
        # zmiana wymiarów
        transforms.Resize(working_size),
        # wycinamy środek obrazu w docelowych wymiarach
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def denormalize(tensor: torch.Tensor, mean=NORM_MEAN, std=NORM_STD) -> torch.Tensor:
    """
    Cofnięcie procesu normalizacji na tensorze obrazu (przywrócenie oryginalnej skali kolorów).

    W uczeniu głębokim obrazy wejściowe są często standaryzowane według wzoru:
    output = (input - mean) / std. Funkcja ta odwraca tę operację (output = input * std + mean),
    co jest niezbędne do poprawnej wizualizacji obrazów, które zostały wcześniej
    przetworzone przez transformacje PyTorch (np. `transforms.Normalize`).

    Args:
        tensor (torch.Tensor): Tensor obrazu o kształcie (C, H, W).
        mean (tuple/list): Średnie użyte do normalizacji dla każdego kanału (R, G, B).
        std (tuple/list): Odchylenia standardowe użyte do normalizacji dla każdego kanału (R, G, B).

    Returns:
        torch.Tensor: Zdenormalizowany tensor gotowy do konwersji na format obrazu (np. PIL lub NumPy).
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def get_fold_data(
    df: pd.DataFrame,
    fold_idx: int,
    processed_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_transforms=None,
    val_transforms=None
) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
    """
    Inicjalizuje generatory danych (DataLoaders) dla określonego foldu walidacji krzyżowej.

    Funkcja dzieli zbiór danych na podzbiór treningowy (wszystkie foldy poza obecnym)
    i walidacyjny (obecny fold). Następnie tworzy instancje klasy Dataset z odpowiednimi
    transformacjami oraz konfiguruje DataLoadery z obsługą wielowątkowości i transferu
    pamięci do GPU (pin_memory).

    Args:
        df (pd.DataFrame): Główna ramka danych zawierająca metadane i przypisane numery foldów.
        fold_idx (int): Indeks aktualnego foldu, który ma służyć jako zbiór walidacyjny.
        processed_dir (str): Ścieżka do katalogu z przetworzonymi obrazami.
        batch_size (int): Wielkość partii danych.
        num_workers (int): Liczba procesów ładujących dane.
        train_transforms (callable, optional): Obiekt transformacji dla zbioru treningowego.
        val_transforms (callable, optional): Obiekt transformacji dla zbioru walidacyjnego.

    Returns:
        tuple: (train_loader, val_loader, train_df) - Krotka zawierająca gotowe loadery
               oraz ramkę danych treningowych (potrzebną np. do wyliczenia wag klas).
    """
    # filtrujemy zbiór treningowy
    train_df = df[(df['typ_puli'] == 'trening') & (df['nr_podzbioru'] != fold_idx)].reset_index(drop=True)

    # filtrujemy zbiór walidacyjny
    val_df = df[(df['typ_puli'] == 'trening') & (df['nr_podzbioru'] == fold_idx)].reset_index(drop=True)

    if train_transforms is None:
        train_transforms = get_train_transforms()
    if val_transforms is None:
        val_transforms = get_val_transforms()

    # inicjalizujemy instancje klasy
    train_ds = ISICDataset(train_df, processed_dir, transform=train_transforms)
    val_ds = ISICDataset(val_df, processed_dir, transform=val_transforms)

    # konfigurujemy loadery
    pin_memory = torch.cuda.is_available()
    drop_last = len(train_ds) > batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_df
