# Skin cancer diagnosis using dermatological image analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-B.Eng._Thesis-success)

This repository contains the refactored, modular codebase and methodology for the Engineering Thesis: *"Skin cancer diagnosis using dermatological image analysis"* (Cracow University of Technology).

The project combines deep transfer learning (**DenseNet-121**) with classical Machine Learning classifiers (**SVM**, **Random Forest**) on the **ISIC 2019** dataset, utilizing rigorous statistical validation (**Stratified Group K-Fold**, **Friedman & Nemenyi post-hoc tests**).

---

## 📁 Repository Structure

```text
skin-cancer-umwdm/
├── train.py                  # Główny skrypt CLI do treningu pojedynczego foldu
├── environment.yml           # Definicja środowiska Conda / Mamba
├── requirements.txt          # Zależności dla instalatora pip / venv
├── .env.example              # Szablon konfiguracji zmiennych środowiskowych i Kaggle API
├── .gitignore                # Reguły ignorowania plików dla Git
├── README.md                 # Dokumentacja projektu
│
├── data/                     # Katalog danych lokalnych (ignorowany w git)
│   ├── raw/                  # Surowe wycinki obrazów i metadane ISIC 2024 (.gitkeep)
│   └── processed/            # Przetworzone zbiory danych i cechy (.gitkeep)
│
├── models/                   # Lokalne wagi modeli i artefakty (.pth, .parquet) (.gitkeep)
│
├── src/                      # Kod źródłowy (pakiet modularny)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Klasa ISICDataset, transformacje, get_fold_data
│   │   └── download_data.py  # Pobieranie surowych danych ISIC 2024 z Kaggle API do data/raw/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── densenet.py       # Architektura DenseNet-121, ekstraktor cech
│   │   └── trainer.py        # Klasa Trainer (dwuetapowy trening, early stopping, get_loss)
│   └── utils/
│       ├── __init__.py
│       ├── artifact_sync.py  # Synchronizacja (download/upload) artefaktów modeli z Kaggle
│       ├── device_check.py   # Weryfikacja akceleratora sprzętowego (CUDA / MPS / CPU)
│       └── metrics.py        # Obliczanie metryk (specyficzność, MCC, czułość, precyzja, acc)
│
├── tests/                    # Pakiet testów jednostkowych (pytest)
│   ├── __init__.py
│   ├── conftest.py           # Fixtury (syntetyczne obrazy, ramki metadanych)
│   ├── test_data_infrastructure.py # Testy infrastruktury danych, Kaggle API i akceleratorów
│   ├── test_dataset.py       # Testy kształtu tensorów Dataset/DataLoader i podziału foldów
│   ├── test_densenet.py      # Testy forward pass na dummy tensorze i ekstraktora
│   ├── test_metrics.py       # Weryfikacja obliczeń matematycznych metryk diagnostycznych
│   └── test_trainer.py       # Testy funkcji get_loss, setup_stage oraz pętli CLI
│
├── figures/                  # Wykresy i wizualizacje wygenerowane podczas badań
│   ├── macierz_pomylek_DenseNet-121_test.pdf
│   ├── krzywe_uczenia_glowne.pdf
│   └── ...
│
└── old_notebooks/            # Pierwotne notatniki badawcze (archiwum)
    ├── 01_data_preparation.ipynb
    ├── 02_densente_training.ipynb
    ├── 03_hybrid_models_training.ipynb
    └── 04_models_comparison.ipynb
```

---

## ⚙️ Instalacja środowiska (Environment Setup)

Projekt można zainstalować zarówno za pomocą menedżera **Conda / Mamba**, jak i standardowego **Python venv + pip**.

### Wariant 1: Conda / Mamba (Rekomendowany)

```bash
# 1. Klonowanie repozytorium (jeśli jeszcze nie sklonowano)
git clone https://github.com/rafalwodzinski/skin-cancer-umwdm.git
cd skin-cancer-umwdm

# 2. Utworzenie środowiska z pliku environment.yml
conda env create -f environment.yml
# LUB (znacznie szybsza instalacja przez mamba):
# mamba env create -f environment.yml

# 3. Aktywacja środowiska
conda activate skin-cancer-umwdm
```

### Wariant 2: Python Virtual Environment (`venv` + `pip`)

```bash
# 1. Utworzenie wirtualnego środowiska (Python 3.10+)
python3 -m venv venv

# 2. Aktywacja środowiska
# Na systemach macOS / Linux:
source venv/bin/activate
# Na systemie Windows:
# venv\Scripts\activate

# 3. Aktualizacja pip i instalacja zależności
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Konfiguracja zmiennych środowiskowych (`.env`)

Do komunikacji z Kaggle API (pobieranie surowych danych i artefaktów modeli) wymagane są klucze uwierzytelniające.

1. Utwórz lokalny plik `.env` na podstawie szablonu:
   ```bash
   cp .env.example .env
   ```
2. Pobierz swój token z Kaggle: przejdź do profilu [Kaggle](https://www.kaggle.com/) -> `Settings` -> sekcja `API` -> `Create New Token` (pobierze się plik `kaggle.json`).
3. Uzupełnij w pliku `.env` swoje wartości:
   ```env
   KAGGLE_USERNAME=twoj_login_kaggle
   KAGGLE_KEY=twoj_klucz_api_kaggle

   # Identyfikatory prywatnych zbiorów zespołu:
   KAGGLE_RAW_DATASET="rafalwodzinski/isic2024"
   KAGGLE_ARTIFACTS_DATASET="rafalwodzinski/skin-cancer-models-artifacts"
   ```
> [!NOTE]
> Plik `.env` jest ignorowany przez `.gitignore` i nie zostanie przypadkowo wypchnięty do repozytorium.

---

## ⚡ Weryfikacja akceleracji sprzętowej (`device_check.py`)

Aby upewnić się, że PyTorch poprawnie wykrywa akcelerację sprzętową (**NVIDIA CUDA** lub **Apple Silicon MPS**):

```bash
python src/utils/device_check.py
```

Skrypt weryfikuje dostępność w hierarchii: `cuda` -> `mps` -> `cpu` i wyświetli aktywne urządzenie oraz wersję PyTorch.

---

## 📦 Architektura danych i synchronizacja Kaggle API

Repozytorium GitHub nie przechowuje dużych plików binarnych (obrazów, plików wag `.pth`, wektorów `.parquet`). Wszystkie dane przechowywane są na platformie Kaggle.

> [!IMPORTANT]
> **Zasada podziału zadań w zespole:**
> - **Surowe obrazy (`download_data.py`) pobierają TYLKO osoby trenujące sieci konwolucyjne / wizyjne na mocnym sprzęcie.**
> - **Reszta zespołu NIE pobiera surowych danych! Wszyscy pozostali członkowie zespołu korzystają wyłącznie ze współdzielonych wag modeli i wyekstrahowanych cech przy użyciu `src/utils/artifact_sync.py`.**

### 1. Pobieranie surowych danych ISIC 2024 (Tylko trening wizyjny)

```bash
# Bezpieczne pobranie surowych danych do data/raw/ w korzeniu projektu:
python src/data/download_data.py

# Wymuszenie ponownego pobrania nawet jeśli dane już istnieją:
python src/data/download_data.py --force
```

Skrypt automatycznie sprawdza obecność danych w `data/raw/` (pomijając `.gitkeep`), a jeśli katalog nie jest pusty, informuje o tym i pomija ponowne pobieranie.

### 2. Współdzielenie artefaktów i wag modeli (`artifact_sync.py`)

Katalog `models/` w korzeniu projektu służy do wymiany wag modeli `.pth`, metadanych oraz wyekstrahowanych embeddingów `.parquet`:

```bash
# Pobranie najnowszych modeli i artefaktów do folderu models/:
python src/utils/artifact_sync.py --download

# Wypchnięcie nowej wersji wytrenowanych modeli na Kaggle:
python src/utils/artifact_sync.py --upload -m "Wytrenowane wagi DenseNet-121 fold 0 (MCC: 0.74)"
```

Funkcje te można również importować bezpośrednio w kodzie Python:
```python
from src.utils.artifact_sync import download_artifacts, upload_artifacts

# Pobranie artefaktów do katalogu models/ w korzeniu projektu
download_artifacts()

# Upload nowej wersji po treningu
upload_artifacts(folder_to_upload="models/", version_notes="Nowe wagi DenseNet")
```


---

## 🧪 Uruchamianie testów jednostkowych

Pakiet testów weryfikuje poprawność zrefaktoryzowanych modułów deterministycznie (bez konieczności pobierania pełnego zbioru ImageNet czy ISIC):
* Poprawność kształtu tensorów zwracanych przez `DataLoader` i `ISICDataset` (`torch.Size([B, 3, 300, 300])`).
* Propagację w przód (`forward pass`) na sztucznym wejściu (`dummy tensor`) dla `DenseNet-121` i ekstraktora cech.
* Matematyczną poprawność specyficzności (One-vs-Rest) oraz zestawu metryk diagnostycznych.
* Działanie procedury dwuetapowego uczenia (`Trainer`) oraz uruchomienia CLI (`train.py --dry-run`).

Aby uruchomić wszystkie testy:

```bash
pytest
```

Szczegółowy widok wykonania:

```bash
pytest -v
```

---

## 🚀 Uruchamianie treningu (`train.py`)

Punkt wejścia `train.py` pozwala na uruchomienie treningu dowolnego foldu walidacji krzyżowej z poziomu terminala.

### Przykłady wywołań:

#### 1. Standardowy trening foldu 0:
```bash
python train.py --fold 0 --metadata-path ISIC_metadata_processed.csv --images-dir images_processed
```

#### 2. Trening wybranego foldu (np. fold 2) z dostosowanymi hiperparametrami:
```bash
python train.py \
    --fold 2 \
    --metadata-path data/ISIC_metadata_processed.csv \
    --images-dir data/images_processed \
    --output-dir results \
    --batch-size 32 \
    --warmup-epochs 10 \
    --finetune-epochs 40 \
    --device cuda
```

#### 3. Szybka weryfikacja przepływu (Dry-Run / Smoke Test):
```bash
python train.py \
    --metadata-path ISIC_metadata_processed.csv \
    --images-dir images_processed \
    --dry-run \
    --no-pretrained \
    --batch-size 4
```

### Dostępne parametry CLI (`train.py --help`):

| Parametr | Typ | Wartość domyślna | Opis |
| :--- | :--- | :--- | :--- |
| `--fold` | `int` | `0` | Indeks foldu walidacji krzyżowej (0..4) |
| `--metadata-path` | `str` | `ISIC_metadata_processed.csv` | Ścieżka do przetworzonego pliku CSV z metadanymi |
| `--images-dir` | `str` | `images_processed` | Ścieżka do katalogu z przetworzonymi obrazami |
| `--output-dir` | `str` | `results` | Katalog do zapisu wag modeli `.pth` i raportów JSON |
| `--batch-size` | `int` | `64` | Wielkość partii danych |
| `--num-workers` | `int` | `2` | Liczba procesów ładujących DataLoader |
| `--warmup-epochs` | `int` | `10` | Liczba epok w etapie 1 (trening samej głowicy) |
| `--finetune-epochs` | `int` | `60` | Liczba epok w etapie 2 (dostrajanie głębokie) |
| `--warmup-lr` | `float` | `1e-3` | Współczynnik uczenia dla etapu 1 |
| `--finetune-lr` | `float` | `3e-5` | Współczynnik uczenia dla etapu 2 |
| `--warmup-patience` | `int` | `5` | Cierpliwość wczesnego zatrzymania (etap 1) |
| `--finetune-patience` | `int` | `12` | Cierpliwość wczesnego zatrzymania (etap 2) |
| `--weight-decay` | `float` | `1e-2` | Waga kary L2 optymalizatora AdamW |
| `--seed` | `int` | `42` | Ziarno losowości dla reprodukowalności |
| `--device` | `str` | `None` (auto) | Urządzenie obliczeniowe (`cuda`, `mps`, `cpu`) |
| `--no-pretrained` | `flag` | `False` | Inicjalizacja losowa zamiast wag ImageNet |
| `--dry-run` | `flag` | `False` | Uruchamia po 1 epoce testowej w celu weryfikacji pipeline'u |

---

## 🔬 Metodologia badawcza

1. **Przygotowanie danych (`src/data/dataset.py`, `old_notebooks/01_data_preparation.ipynb`):**
   * Podział **Stratified Group K-Fold** zapobiegający przeciekowi danych na poziomie pacjenta (`lesion_id`).
   * Augmentacja obrazów: `RandomResizedCrop`, odbicia lustrzane, `ColorJitter`, `RandomRotation`, normalizacja ImageNet.

2. **Dwuetapowe uczenie transferowe (`src/models/trainer.py`, `old_notebooks/02_densente_training.ipynb`):**
   * **Etap 1 (Rozgrzewka głowicy):** Ekstraktor cech zamrożony, trening głowicy `Dropout(0.5) + Linear(1024, 8)` z wyższym LR (`1e-3`).
   * **Etap 2 (Dostrajanie głębokie):** Odmrożenie ostatnich bloków konwolucyjnych (`denseblock3`, `transition3`, `denseblock4`, `norm5`) z niskim LR (`3e-5`) i schedulerem `ReduceLROnPlateau`.
   * Ważona funkcja straty `CrossEntropyLoss` z wagami $w_c = \frac{1}{\sqrt{N_c}}$ oraz Label Smoothing (0.1).
   * Zapis najlepszego stanu w oparciu o metrykę **MCC (Matthews Correlation Coefficient)**.

3. **Klasyfikatory hybrydowe (`old_notebooks/03_hybrid_models_training.ipynb`):**
   * Wyciągnięcie wektorów cech z warstwy przedklasyfikacyjnej DenseNet (`src/models/densenet.py:get_densenet_extractor`).
   * Trening klasyfikatorów **SVM** i **Random Forest** na wyekstrahowanych embeddingach.

4. **Porównanie statystyczne (`old_notebooks/04_models_comparison.ipynb`):**
   * Test Friedmana oraz test post-hoc Nemenyiego potwierdzające istotną statystycznie przewagę podejścia hybrydowego (DenseNet + SVM) nad klasycznym softmaxem.

---

## 👤 Autor
* **Rafał Wodziński** – Politechnika Krakowska
