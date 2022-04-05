import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.0  # 0.5 * LAMBDA_CYCLE
NUM_WORKERS = 16
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_X = "gen_x.pth.tar"
CHECKPOINT_GEN_Y = "gen_y.pth.tar"
CHECKPOINT_DIS_X = "dis_x.pth.tar"
CHECKPOINT_DIS_Y = "dis_y.pth.tar"

CHECKPOINT_CLASS = "class.pth.tar"

COMPRESS_QUALITY = 1


transform_train = T.Compose(
    [
        T.ToTensor(),
        T.Resize(size=(256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

transform_val = T.Compose(
    [
        T.ToTensor(),
        T.Resize(size=(256, 256)),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
