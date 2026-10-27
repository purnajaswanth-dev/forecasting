from pathlib import Path
import torch

# ==========================
# Paths
# ==========================

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
SEQUENCE_DATA = DATA_DIR / "sequences"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# ==========================
# Model Parameters
# ==========================

SEQUENCE_LENGTH = 24

INPUT_SIZE = 7

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.3

OUTPUT_SIZE = 1

# ==========================
# Training
# ==========================

BATCH_SIZE = 256

LEARNING_RATE = 0.0005

EPOCHS = 100

PATIENCE = 10

MIN_DELTA = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)