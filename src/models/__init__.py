"""
Moduł architektur modeli oraz pętli treningowej.
"""
from src.models.densenet import get_model, get_densenet_extractor
from src.models.trainer import Trainer, get_loss

__all__ = [
    "get_model",
    "get_densenet_extractor",
    "Trainer",
    "get_loss",
]
