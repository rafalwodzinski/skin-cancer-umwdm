import torch
import torch.nn as nn
from src.models.densenet import get_model, get_densenet_extractor, NUM_CLASSES


def test_densenet_forward_pass_dummy_tensor():
    """
    Weryfikuje poprawność propagacji w przód (forward pass) modelu DenseNet-121
    na sztucznym wejściu (dummy tensor).
    """
    batch_size = 2
    num_classes = 8
    img_size = 300

    # Inicjalizacja modelu bez pobierania zewnętrznych wag (szybki test deterministyczny)
    model = get_model(num_classes=num_classes, weights=None, drop_rate=0.2)
    model.eval()

    # Sprawdzenie struktury warstwy klasyfikacyjnej
    assert isinstance(model.classifier, nn.Sequential)
    assert len(model.classifier) == 2
    assert isinstance(model.classifier[0], nn.Dropout)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == num_classes
    assert model.classifier[1].in_features == 1024

    # Dummy tensor o wymiarach (B, C, H, W)
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, num_classes)
    assert not torch.isnan(output).any(), "Wyjście modelu zawiera wartości NaN"
    assert not torch.isinf(output).any(), "Wyjście modelu zawiera wartości inf"


def test_densenet_extractor_forward_pass():
    """
    Weryfikuje, czy ekstraktor cech DenseNet poprawnie zwraca wektor cech (embeddings)
    o wymiarze 1024 na sztucznym wejściu.
    """
    batch_size = 2
    img_size = 300

    extractor = get_densenet_extractor(weights_path=None, num_classes=NUM_CLASSES)
    assert isinstance(extractor.classifier, nn.Identity)

    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    with torch.no_grad():
        features = extractor(dummy_input)

    assert features.shape == (batch_size, 1024)
    assert not torch.isnan(features).any()
