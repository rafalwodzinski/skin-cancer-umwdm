import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import numpy as np
import pandas as pd
from PIL import Image


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """
    Tworzy tymczasowy katalog z syntetycznymi obrazami oraz ramkę metadanych.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    num_samples = 12

    for i in range(num_samples):
        filename = f"ISIC_{i:04d}.jpg"
        file_path = img_dir / filename

        # Tworzenie syntetycznego obrazka RGB 100x100
        arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(file_path)

        records.append({
            'nazwa_pliku': filename,
            'target': i % 8,
            'id_zmiany': f"lesion_{i // 2}",
            'typ_puli': 'trening',
            'nr_podzbioru': i % 3  # podział na 3 foldy (0, 1, 2)
        })

    df = pd.DataFrame(records)
    return img_dir, df
