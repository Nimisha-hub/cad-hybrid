
import pandas as pd
import torch
from pathlib import Path

from src.model_config import PROJECT_ROOT, TEST_CSV
from src.ecg_dataset import ECGDataset
from src.predict_ecg import preprocess_image


# Load test dataset
test_data = pd.read_csv(TEST_CSV)

# Select the first test image
row = test_data.iloc[0]

# Get original image path
processed_path = Path(row["path"])
parts = processed_path.parts

interim_index = parts.index("interim")
relative_path = Path(*parts[interim_index + 1:])

raw_image_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ecg_multan_pk"
    / relative_path
)

# Load the training-style preprocessed image
dataset = ECGDataset(TEST_CSV)

training_tensor, label, group = dataset[0]

# Load the same original image through prediction preprocessing
prediction_tensor = preprocess_image(str(raw_image_path)).squeeze(0)

# Compare both tensors
difference = torch.abs(training_tensor - prediction_tensor)

print("\nPREPROCESSING COMPARISON")
print("==============================")

print("Image:", raw_image_path.name)
print("Actual label:", row["label"])

print("Training tensor shape:", training_tensor.shape)
print("Prediction tensor shape:", prediction_tensor.shape)

print(
    "Mean absolute difference:",
    difference.mean().item()
)

print(
    "Maximum absolute difference:",
    difference.max().item()
)

print(
    "Are tensors identical:",
    torch.equal(training_tensor, prediction_tensor)
)
