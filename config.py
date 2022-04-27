import albumentations as A
import torch
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
NUM_WORKERS = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5

LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.0  # 0.5 * LAMBDA_CYCLE

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_CYCLE_DIS = "cycle_dis.pth.tar"
CHECKPOINT_CYCLE_GEN = "cycle_gen.pth.tar"

CHECKPOINT_CLASS = "class.pth.tar"

COMPRESS_QUALITY = 1


transform_train = T.Compose(
    [
        T.ToTensor(),
        T.Resize(size=(256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_val = T.Compose(
    [
        T.Resize(size=(256, 256)),
        T.ToTensor(),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
