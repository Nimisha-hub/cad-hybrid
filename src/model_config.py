
from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV = PROCESSED_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Model settings
MODEL_NAME = "ResNet18"
PRETRAINED = True

# Image settings
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# Training settings
BATCH_SIZE = 8
RANDOM_STATE = 42

# Device configuration
DEVICE = "cpu"

# Class labels
CLASS_NAMES = [
    "cad_positive",
    "negative",
    "arrhythmia"
]

NUM_CLASSES = len(CLASS_NAMES)

# Model output directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("ECG model configuration loaded successfully.")
