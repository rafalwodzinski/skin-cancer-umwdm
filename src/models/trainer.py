import os
import time
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.metrics import calculate_metrics

NUM_CLASSES = 8
WARMUP_EPOCHS = 10
WARMUP_LR = 1e-3
WARMUP_PATIENCE = 5

FINETUNE_EPOCHS = 60
FINETUNE_LR = 3e-5
FINETUNE_PATIENCE = 12

WEIGHT_DECAY = 1e-2


def get_loss(
    train_df: pd.DataFrame,
    device: Union[torch.device, str],
    num_classes: int = NUM_CLASSES
) -> nn.CrossEntropyLoss:
    """
    Konfiguruje funkcję straty (Loss Function) z mechanizmem równoważenia klas.

    Funkcja oblicza wagi dla każdej klasy na podstawie ich liczebności w zbiorze treningowym,
    stosując strategię "odwrotności pierwiastka" (1/sqrt(N)). Jest to podejście łagodniejsze
    niż prosta odwrotność częstości, zapobiegające niestabilności gradientów w przypadku
    bardzo rzadkich klas. Dodatkowo wagi są normalizowane do średniej 1.0, a funkcja straty
    wykorzystuje Label Smoothing (0.1) w celu zapobiegania nadmiernej pewności modelu.

    Args:
        train_df (pd.DataFrame): Ramka danych zbioru treningowego (wymagana kolumna 'target').
        device (torch.device lub str): Urządzenie obliczeniowe (CPU/CUDA/MPS), na którym umieszczone zostaną wagi.
        num_classes (int): Liczba wszystkich klas modelu (domyślnie 8).

    Returns:
        nn.CrossEntropyLoss: Instancja funkcji straty z załadowanymi wagami i wygładzaniem etykiet.
    """
    counts_series = train_df['target'].value_counts()
    class_counts = np.array([counts_series.get(c, 1) for c in range(num_classes)], dtype=float)

    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / np.mean(weights)
    weights_tensor = torch.FloatTensor(weights).to(device)

    return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)


class Trainer:
    """
    Hermetyzuje pętlę treningową, walidacyjną oraz procedurę dwuetapowego uczenia
    transferowego dla modelu klasyfikacji dermatologicznej DenseNet.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        num_classes: int = NUM_CLASSES,
        save_dir: Optional[str] = None,
        model_name: str = "DenseNet-121",
        fold_idx: int = 0,
        warmup_epochs: int = WARMUP_EPOCHS,
        warmup_lr: float = WARMUP_LR,
        warmup_patience: int = WARMUP_PATIENCE,
        finetune_epochs: int = FINETUNE_EPOCHS,
        finetune_lr: float = FINETUNE_LR,
        finetune_patience: int = FINETUNE_PATIENCE,
        weight_decay: float = WEIGHT_DECAY,
        use_amp: bool = True
    ):
        """
        Inicjalizuje obiekt Trainera.

        Args:
            model (nn.Module): Instancja modelu PyTorch.
            train_loader (DataLoader): DataLoader zbioru treningowego.
            val_loader (DataLoader): DataLoader zbioru walidacyjnego.
            criterion (nn.Module): Funkcja straty (np. zbalansowany CrossEntropyLoss).
            device (torch.device lub str, optional): Urządzenie obliczeniowe.
            num_classes (int): Liczba klas w zadaniu.
            save_dir (str, optional): Katalog zapisu wag modelu i wyników.
            model_name (str): Identyfikator nazwy modelu.
            fold_idx (int): Numer foldu walidacji krzyżowej.
            warmup_epochs (int): Liczba epok w etapie treningu samej głowicy.
            warmup_lr (float): Learning rate dla etapu rozgrzewki.
            warmup_patience (int): Cierpliwość wczesnego zatrzymania w etapie 1.
            finetune_epochs (int): Liczba epok w etapie dostrajania głębokiego.
            finetune_lr (float): Learning rate dla etapu dostrajania.
            finetune_patience (int): Cierpliwość wczesnego zatrzymania w etapie 2.
            weight_decay (float): Waga regularyzacji L2 dla optymalizatora AdamW.
            use_amp (bool): Czy korzystać z automatycznej precyzji mieszanej (AMP).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.num_classes = num_classes

        self.save_dir = save_dir
        self.model_name = model_name
        self.fold_idx = fold_idx

        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.warmup_patience = warmup_patience

        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.finetune_patience = finetune_patience

        self.weight_decay = weight_decay
        self.use_amp = use_amp and (self.device.type == "cuda")

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_fold{self.fold_idx}.pth")
        else:
            self.checkpoint_path = None

        self.best_mcc_overall: float = -1.0
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def setup_stage(self, stage: str) -> Tuple[optim.Optimizer, Optional[Any], int, int]:
        """
        Konfiguruje hiperparametry treningu oraz stan zamrożenia warstw dla danego etapu.

        Funkcja realizuje strategię dwuetapowego uczenia transferowego:
        1. 'trening klasyfikatora': Zamraża cały ekstraktor cech, aktualizując wagi jedynie
           w nowej głowicy klasyfikacyjnej.
        2. 'dostrajanie modelu': Odmraża kluczowe bloki konwolucyjne (DenseBlock 3 i 4),
           zmniejsza learning rate i aktywuje scheduler w celu precyzyjnego dopasowania wag.

        Args:
            stage (str): Identyfikator etapu ('trening klasyfikatora' lub 'dostrajanie modelu').

        Returns:
            tuple: Krotka konfiguracyjna (optimizer, scheduler, num_epochs, patience).
        """
        if stage == 'trening klasyfikatora':
            print("\nKonfiguracja etapu 1: Trening klasyfikatora")
            for param in self.model.features.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                self.model.classifier.parameters(),
                lr=self.warmup_lr,
                weight_decay=self.weight_decay
            )
            scheduler = None

            return optimizer, scheduler, self.warmup_epochs, self.warmup_patience

        elif stage == 'dostrajanie modelu':
            print("\nKonfiguracja etapu 2: Dostrajanie modelu")
            # odmrażamy ostatni blok
            for param in self.model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.model.features.norm5.parameters():
                param.requires_grad = True

            # odmrażamy kolejny blok
            for param in self.model.features.transition3.parameters():
                param.requires_grad = True
            for param in self.model.features.denseblock3.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.finetune_lr,
                weight_decay=self.weight_decay
            )

            # definiujemy harmonogramowanie learning_rate w oparciu o poprawę MCC
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

            return optimizer, scheduler, self.finetune_epochs, self.finetune_patience

        else:
            raise ValueError(f"Nieznany etap uczenia: {stage}")

    def train_one_epoch(
        self,
        optimizer: optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None
    ) -> float:
        """
        Realizuje jedną pełną epokę treningową modelu.

        Funkcja iteruje przez wszystkie partie danych treningowych,
        oblicza błędy predykcji, wyznacza gradienty i aktualizuje wagi modelu.

        Args:
            optimizer (optim.Optimizer): Algorytm optymalizujący wagi.
            scaler (GradScaler, optional): Skaler gradientów dla mieszanej precyzji (CUDA).

        Returns:
            float: Średnia wartość funkcji straty dla całej epoki.
        """
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="  Trening", leave=False)

        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            if scaler is not None and self.device.type == "cuda":
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return float(epoch_loss)

    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Przeprowadza proces ewaluacji modelu na przekazanym DataLoaderze (domyślnie val_loader).

        Funkcja przełącza model w tryb inferencji (wyłącza obliczanie gradientów),
        generuje predykcje i wyznacza pełen zestaw metryk diagnostycznych.

        Args:
            dataloader (DataLoader, optional): Loader danych do ewaluacji.

        Returns:
            dict: Słownik zawierający metryki ('strata', 'dokładność', 'mcc', 'czułość', 'precyzja', 'specyficzność').
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = calculate_metrics(all_labels, all_preds, num_classes=self.num_classes)

        return {
            'strata': float(epoch_loss),
            **metrics
        }

    def get_predictions(
        self,
        dataloader: Optional[DataLoader] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Generuje predykcje dla całego DataLoader'a (używane na koniec foldu).

        Args:
            dataloader (DataLoader, optional): Loader danych (domyślnie val_loader).

        Returns:
            tuple: (y_true, y_pred)
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        return all_labels, all_preds

    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Przeprowadza pełny dwuetapowy proces uczenia modelu na zadanym foldzie:
        - Etap 1: Trening głowicy klasyfikacyjnej.
        - Etap 2: Fine-tuning wybranych bloków splotowych.

        Returns:
            tuple: Krotka (history, fold_predictions) zawierająca:
                - history (dict): Szczegółowy przebieg uczenia ze statystykami epok.
                - fold_predictions (dict): Predykcje najlepszego modelu na walidacji.
        """
        start_time = time.time()
        print(f"\n{'='*5} START TRENINGU DLA FOLDU {self.fold_idx} {'='*5}")

        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        history: Dict[str, Any] = {'fold': self.fold_idx, 'metryki': []}
        self.best_mcc_overall = -1.0

        stages = ['trening klasyfikatora', 'dostrajanie modelu']

        for stage_name in stages:
            optimizer, scheduler, n_epochs, patience_limit = self.setup_stage(stage_name)
            patience_cnt = 0
            best_mcc_stage = -1.0

            print(f"Etap - {stage_name}: Liczba epok={n_epochs}, Cierpliwość={patience_limit}")

            for epoch in range(n_epochs):
                t_loss = self.train_one_epoch(optimizer, scaler)
                metrics = self.evaluate(self.val_loader)

                print(
                    f"\tEpoka {epoch + 1}/{n_epochs} | "
                    f"Strata: {t_loss:.4f} | "
                    f"Val MCC: {metrics['mcc']:.4f} | "
                    f"Val Czułość: {metrics['czułość']:.4f} | "
                    f"Val Specyficzność: {metrics['specyficzność']:.4f} | "
                    f"Val Precyzja: {metrics['precyzja']:.4f} | "
                    f"Val Dokładność: {metrics['dokładność']:.4f}"
                )

                record = {
                    'etap': stage_name,
                    'ostatnia_epoka': epoch + 1,
                    'strata_treningowa': t_loss,
                    **metrics
                }
                history['metryki'].append(record)

                if scheduler is not None:
                    scheduler.step(metrics['mcc'])

                # Weryfikacja najlepszego wyniku ogólnego
                if metrics['mcc'] > self.best_mcc_overall:
                    self.best_mcc_overall = metrics['mcc']
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    if self.checkpoint_path:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                        print(f"\t>>> Poprawa modelu! Najlepsze MCC: {self.best_mcc_overall:.4f}. Zapisano wagi.")

                # Logika wczesnego zatrzymania w ramach etapu
                if metrics['mcc'] > best_mcc_stage:
                    best_mcc_stage = metrics['mcc']
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience_limit:
                        print(f"  [Info] Zatrzymano trening w etapie {stage_name} po {epoch + 1} epokach.")
                        break

        duration = time.time() - start_time
        history['czas_trwania_foldu'] = duration
        history['najlepsze_mcc'] = self.best_mcc_overall

        print(f"\nGenerowanie predykcji dla foldu {self.fold_idx}...")
        self.load_best_model()
        final_y_true, final_y_pred = self.get_predictions(self.val_loader)

        fold_predictions = {
            'fold': self.fold_idx,
            'y_true': final_y_true,
            'y_pred': final_y_pred
        }

        print(f"Koniec treningu na foldzie {self.fold_idx}. Czas: {duration / 60:.1f} min. Najlepszy MCC: {self.best_mcc_overall:.4f}")
        return history, fold_predictions

    def load_best_model(self) -> None:
        """
        Przywraca najlepsze wagi modelu (z pamięci podręcznej lub z pliku checkpointu).
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})
            print("Wczytano najlepsze wagi modelu z pamięci.")
        elif self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Wczytano najlepsze wagi z checkpointu: {self.checkpoint_path}")
        else:
            print("[Ostrzeżenie] Brak zapisanego stanu najlepszego modelu, pozostawiono aktualny.")
