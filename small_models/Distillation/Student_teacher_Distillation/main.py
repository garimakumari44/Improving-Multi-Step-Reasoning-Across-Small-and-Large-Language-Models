from training.train import train
from utils.device import get_device
from utils.seed import set_seed

if __name__ == "__main__":
    set_seed(42)
    device = get_device()

    student, teacher = train(
        epochs=5,
        lr=3e-4,
        device=device
    )