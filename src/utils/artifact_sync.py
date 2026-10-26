import argparse
import json
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


def download_artifacts(
    download_dir: str = "models/",
    dataset: str | None = None,
) -> Path:
    """
    Pobiera współdzielone wagi modeli i wektory cech z prywatnego zbioru Kaggle.
    Zapisuje artefakty w folderze korzenia projektu (domyślnie: models/).
    """
    target_path = Path(download_dir)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / target_path).resolve()

    target_path.mkdir(parents=True, exist_ok=True)

    if not dataset:
        dataset = os.getenv(
            "KAGGLE_ARTIFACTS_DATASET",
            "twoj-team/skin-cancer-models-artifacts",
        )

    print(f"[INFO] Pobieranie artefaktów ze zbioru '{dataset}' do: {target_path}")
    api = init_kaggle_api()

    api.dataset_download_cli(
        dataset=dataset,
        path=str(target_path),
        unzip=True,
        force=True,
        quiet=False,
    )
    print(f"[SUCCESS] Artefakty pobrane pomyślnie do {target_path}.")
    return target_path


def upload_artifacts(
    folder_to_upload: str = "models/",
    version_notes: str = "Nowe wagi",
    dataset: str | None = None,
):
    """
    Tworzy nową wersję zbioru artefaktów na Kaggle na podstawie lokalnego folderu (domyślnie: models/).
    Używa kaggle datasets version (api.dataset_create_version).
    """
    source_path = Path(folder_to_upload)
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if not source_path.is_dir():
        raise FileNotFoundError(f"Katalog '{source_path}' nie istnieje!")

    # Sprawdzenie czy w folderze znajdują się jakiekolwiek pliki do wysłania (poza .gitkeep)
    files = [f for f in source_path.iterdir() if f.name != ".gitkeep"]
    if not files:
        raise ValueError(
            f"Folder '{source_path}' nie zawiera żadnych plików/artefaktów do wysłania!"
        )

    if not dataset:
        dataset = os.getenv(
            "KAGGLE_ARTIFACTS_DATASET",
            "twoj-team/skin-cancer-models-artifacts",
        )

    # Kaggle dataset_create_version wymaga obecności pliku dataset-metadata.json w folderze
    metadata_file = source_path / "dataset-metadata.json"
    if not metadata_file.is_file():
        dataset_slug = dataset.split("/")[-1]
        dataset_title = dataset_slug.replace("-", " ").replace("_", " ").title()
        metadata_content = {
            "title": dataset_title,
            "id": dataset,
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=2)
        print(f"[INFO] Wygenerowano plik metadanych datasetu: {metadata_file}")
    else:
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        if meta_data.get("id") != dataset:
            meta_data["id"] = dataset
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

    api = init_kaggle_api()
    print(
        f"[INFO] Wysyłanie nowej wersji artefaktów do '{dataset}' z opisem: '{version_notes}'..."
    )

    response = api.dataset_create_version(
        str(source_path),
        version_notes=version_notes,
        dir_mode="zip",
        quiet=False,
    )
    print(f"[SUCCESS] Nowa wersja artefaktów wysłana pomyślnie. Status: {response}")
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Synchronizacja wag modeli i artefaktów z prywatnym zbiorem Kaggle."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--download",
        action="store_true",
        help="Pobierz artefakty z Kaggle do folderu models/",
    )
    group.add_argument(
        "--upload",
        action="store_true",
        help="Wyślij nowe artefakty na Kaggle",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="models/",
        help="Ścieżka do katalogu artefaktów (domyślnie: models/ w korzeniu projektu)",
    )
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        default="Nowe wagi",
        help="Opis nowej wersji przy uploadzie (domyślnie: 'Nowe wagi')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Identyfikator datasetu Kaggle (domyślnie z KAGGLE_ARTIFACTS_DATASET w .env)",
    )

    args = parser.parse_args()

    try:
        if args.download:
            download_artifacts(download_dir=args.dir, dataset=args.dataset)
        elif args.upload:
            upload_artifacts(
                folder_to_upload=args.dir,
                version_notes=args.message,
                dataset=args.dataset,
            )
    except Exception as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
