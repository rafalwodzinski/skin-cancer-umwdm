import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Określenie korzenia projektu (katalog repozytorium)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Wczytanie zmiennych środowiskowych z pliku .env w korzeniu projektu
load_dotenv(PROJECT_ROOT / ".env")


def init_kaggle_api():
    """Weryfikuje zmienne środowiskowe i inicjalizuje klienta Kaggle API."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key or username == "twój_login_kaggle" or key.startswith("twój_"):
        raise EnvironmentError(
            "Brak poprawnych danych uwierzytelniających Kaggle (KAGGLE_USERNAME, KAGGLE_KEY) w pliku .env!\n"
            "Skopiuj .env.example do .env i uzupełnij swoje dane API z profilu Kaggle."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_raw_data(
    dataset: str | None = None,
    download_dir: str = "data/raw",
    force: bool = False,
) -> Path:
    """
    Pobiera surowe obrazy i metadane ISIC 2024 z prywatnego zbioru Kaggle.
    Zapisuje pobrane pliki w folderze korzenia projektu (domyślnie: data/raw/).

    UWAGA: Skrypt przeznaczony wyłącznie dla osób trenujących sieci wizyjne.
    """
    target_path = Path(download_dir)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / target_path).resolve()

    target_path.mkdir(parents=True, exist_ok=True)

    # Weryfikacja czy dane już istnieją (ignorując plik techniczny .gitkeep)
    existing_files = [f for f in target_path.iterdir() if f.name != ".gitkeep"]
    if existing_files and not force:
        print(f"[INFO] Dane już znajdują się w {target_path} (znaleziono {len(existing_files)} elementów). Pomijanie pobierania.")
        print("[INFO] Użyj argumentu force=True (lub flagi --force w CLI), aby wymusić ponowne pobranie.")
        return target_path

    if not dataset:
        dataset = os.getenv("KAGGLE_RAW_DATASET", "rafalwodzinski/isic2024")

    print(f"[INFO] Inicjalizacja pobierania surowych danych ze zbioru '{dataset}' do: {target_path}")
    api = init_kaggle_api()

    api.dataset_download_cli(
        dataset=dataset,
        path=str(target_path),
        unzip=True,
        force=force,
        quiet=False,
    )
    print(f"[SUCCESS] Pobieranie i rozpakowywanie danych do {target_path} zakończone pomyślnie.")
    return target_path


def main():
    parser = argparse.ArgumentParser(
        description="Pobieranie surowych danych ISIC 2024 z Kaggle API do katalogu data/raw/."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Identyfikator zbioru na Kaggle (domyślnie z KAGGLE_RAW_DATASET w .env)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Ścieżka docelowa pobierania (domyślnie: data/raw/ w korzeniu projektu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wymuszenie ponownego pobrania nawet jeśli katalog nie jest pusty",
    )
    args = parser.parse_args()

    try:
        download_raw_data(
            dataset=args.dataset,
            download_dir=args.target_dir,
            force=args.force,
        )
    except Exception as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()