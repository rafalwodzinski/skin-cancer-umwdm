import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Znaleziono CUDA. Karta: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Znaleziono MPS (Apple Silicon).")
    else:
        device = torch.device("cpu")
        print("[INFO] Nie znaleziono akceleratora GPU/MPS. Domyślne urządzenie: CPU.")

    return device

if __name__ == "__main__":
    device = get_device()
    print(f"Wersja PyTorch: {torch.__version__}")
    print(f"[INFO] Dostępny akcelerator {device}")
