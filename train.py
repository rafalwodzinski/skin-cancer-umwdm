#!/usr/bin/env python3
"""
Główny skrypt CLI do uruchamiania treningu zadanego foldu modelu DenseNet-121
na zbiorze obrazów dermatologicznych ISIC 2019.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.dataset import get_fold_data, get_train_transforms, get_val_transforms
from src.models.densenet import get_model, NUM_CLASSES
from src.models.trainer import Trainer, get_loss


def set_seed(seed: int = 42) -> None:
    """Ustawia ziarno losowości dla pełnej reprodukowalności wyników."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trening jednego foldu modelu DenseNet-121 dla klasyfikacji zmian skórnych."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Indeks foldu walidacji krzyżowej do wytrenowania (domyślnie: 0)."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="ISIC_metadata_processed.csv",
        help="Ścieżka do przetworzonego pliku CSV z metadanymi i podziałem na foldy."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images_processed",
        help="Ścieżka do katalogu zawierającego przetworzone obrazy."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Katalog do zapisu metryk, predykcji oraz wag modeli."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Rozmiar partii danych (batch size)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Liczba procesów pomocniczych DataLoader (num_workers)."
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Liczba epok rozgrzewki dla głowicy klasyfikacyjnej (etap 1)."
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=60,
        help="Liczba epok dostrajania głębokiego (etap 2)."
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-3,
        help="Współczynnik uczenia w etapie 1 (rozgrzewka)."
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=3e-5,
        help="Współczynnik uczenia w etapie 2 (dostrajanie)."
    )
    parser.add_argument(
        "--warmup-patience",
        type=int,
        default=5,
        help="Cierpliwość wczesnego zatrzymania w etapie 1."
    )
    parser.add_argument(
        "--finetune-patience",
        type=int,
        default=12,
        help="Cierpliwość wczesnego zatrzymania w etapie 2."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Współczynnik kary L2 (weight decay) dla optymalizatora AdamW."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Wartość ziarna losowości (seed)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Wymuszone urządzenie obliczeniowe ('cuda', 'cpu', 'mps'). Domyślnie automatyczne."
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Wyłącza wczytywanie wstępnie wytrenowanych wag ImageNet (inicjalizacja losowa)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tryb testowy: uruchamia po 1 epoce z małą liczbą kroków w celu weryfikacji przepływu."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Wybór urządzenia obliczeniowego
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Urządzenie obliczeniowe: {device}")
    print(f"Rozpoczynanie przygotowania treningu dla foldu: {args.fold}")

    # Weryfikacja istnienia plików metadanych
    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Nie odnaleziono pliku metadanych pod ścieżką: {metadata_path.resolve()}\n"
            f"Upewnij się, że plik istnieje lub wskaż właściwą ścieżkę parametrem --metadata-path."
        )

    df = pd.read_csv(metadata_path)
    required_cols = {'nazwa_pliku', 'target', 'typ_puli', 'nr_podzbioru'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w ramce danych: {missing}")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Nie odnaleziono katalogu z obrazami: {images_dir.resolve()}\n"
            f"Podaj poprawną ścieżkę parametrem --images-dir."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Przygotowanie loaderów
    train_loader, val_loader, train_df = get_fold_data(
        df=df,
        fold_idx=args.fold,
        processed_dir=str(images_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"Dane załadowane: {len(train_loader.dataset)} próbek treningowych, "
          f"{len(val_loader.dataset)} próbek walidacyjnych.")

    # Inicjalizacja modelu
    weights = None if args.no_pretrained else "DEFAULT"
    model = get_model(num_classes=NUM_CLASSES, device=device, weights=weights)

    # Inicjalizacja funkcji straty z ważeniem klas
    criterion = get_loss(train_df, device)

    # Konfiguracja epok (z uwzględnieniem flagi dry-run)
    warmup_epochs = 1 if args.dry_run else args.warmup_epochs
    finetune_epochs = 1 if args.dry_run else args.finetune_epochs

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
        save_dir=args.output_dir,
        model_name="DenseNet-121",
        fold_idx=args.fold,
        warmup_epochs=warmup_epochs,
        warmup_lr=args.warmup_lr,
        warmup_patience=args.warmup_patience,
        finetune_epochs=finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_patience=args.finetune_patience,
        weight_decay=args.weight_decay,
        use_amp=True
    )

    history, fold_predictions = trainer.train()

    # Zapis wyników i predykcji w formacie JSON zgodnym z notatnikami
    results_path = os.path.join(args.output_dir, f"DenseNet-121_wyniki_treningu_fold_{args.fold}.json")
    preds_path = os.path.join(args.output_dir, f"DenseNet-121_predykcje_fold_{args.fold}.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    print(f"Zapisano historię treningu do: {results_path}")

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(fold_predictions, f, indent=4, ensure_ascii=False)
    print(f"Zapisano predykcje do: {preds_path}")

    print(f"\nTrening foldu {args.fold} zakończony sukcesem!")
    print(f"Najlepszy uzyskany wynik MCC: {history.get('najlepsze_mcc', 'N/A')}")


if __name__ == "__main__":
    main()
