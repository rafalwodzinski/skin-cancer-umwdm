import torch
from torch.utils.data import DataLoader
from src.data.dataset import (
    ISICDataset,
    get_train_transforms,
    get_val_transforms,
    denormalize,
    get_fold_data,
    IMG_SIZE,
    NORM_MEAN,
    NORM_STD,
)


def test_isic_dataset_and_dataloader_shapes(mock_dataset_dir):
    img_dir, df = mock_dataset_dir
    transform = get_val_transforms(img_size=IMG_SIZE)

    dataset = ISICDataset(dataframe=df, root_dir=str(img_dir), transform=transform)
    assert len(dataset) == len(df)

    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, IMG_SIZE, IMG_SIZE)
    assert isinstance(label, int)
    assert label == df.iloc[0]['target']

    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_images, batch_labels = next(iter(loader))

    assert batch_images.shape == (batch_size, 3, IMG_SIZE, IMG_SIZE)
    assert batch_labels.shape == (batch_size,)
    assert batch_labels.dtype == torch.int64


def test_train_transforms_shape(mock_dataset_dir):
    img_dir, df = mock_dataset_dir
    transform = get_train_transforms(img_size=IMG_SIZE)

    dataset = ISICDataset(dataframe=df, root_dir=str(img_dir), transform=transform)
    img, _ = dataset[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, IMG_SIZE, IMG_SIZE)


def test_denormalize_math():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # Stwórz tensor znormalizowany: (orig - mean) / std = (1.0 - 0.5) / 0.5 = 1.0
    normalized = torch.ones(3, 10, 10)
    denorm = denormalize(normalized, mean=mean, std=std)

    # Oczekiwany wynik: 1.0 * 0.5 + 0.5 = 1.0
    expected = torch.ones(3, 10, 10)
    assert torch.allclose(denorm, expected, atol=1e-6)

    # Inny test: tensor zerowy -> 0.0 * 0.5 + 0.5 = 0.5
    zero_tensor = torch.zeros(3, 10, 10)
    denorm_zero = denormalize(zero_tensor, mean=mean, std=std)
    assert torch.allclose(denorm_zero, torch.full((3, 10, 10), 0.5), atol=1e-6)


def test_get_fold_data_split(mock_dataset_dir):
    img_dir, df = mock_dataset_dir
    fold_idx = 1
    batch_size = 2

    train_loader, val_loader, train_df = get_fold_data(
        df=df,
        fold_idx=fold_idx,
        processed_dir=str(img_dir),
        batch_size=batch_size,
        num_workers=0
    )

    # Walidacja powinna zawierać tylko próbki z nr_podzbioru == fold_idx
    val_indices = df[df['nr_podzbioru'] == fold_idx]
    assert len(val_loader.dataset) == len(val_indices)

    # Trening powinien zawierać tylko próbki z nr_podzbioru != fold_idx
    train_indices = df[df['nr_podzbioru'] != fold_idx]
    assert len(train_df) == len(train_indices)
    assert len(train_loader.dataset) == len(train_indices)
    assert fold_idx not in train_df['nr_podzbioru'].values
