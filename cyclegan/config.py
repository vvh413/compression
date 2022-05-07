import torch
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tag = "co1_1e5_wd_1024ch_2x2_6res_bce_c"

BATCH_SIZE = 1
NUM_WORKERS = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-5

LAMBDA_CYCLE = 100
LAMBDA_IDENTITY = 0.0  # 0.5 * LAMBDA_CYCLE

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_CYCLE_GEN = f"checkpoints/cycle_gen_{tag}.pth.tar"
CHECKPOINT_CYCLE_DIS = f"checkpoints/cycle_dis_{tag}.pth.tar"
RESULTS = f"results/{tag}"

START_EPOCH = 0

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
