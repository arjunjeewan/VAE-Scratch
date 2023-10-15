import torch

DEVICE = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 16
Z_DIM = 8
EPOCHS = 200
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
PROGRESS_BAR = True 