
import pandas as pd
import torch
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.model_config import PROJECT_ROOT, TEST_CSV, CLASS_NAMES
from src.predict_ecg import load_models, preprocess_image


# Load test dataset
test_data = pd.read_csv(TEST_CSV)

# Load models once
feature_model, classifier = load_models()

y_true = []
y_pred = []

print("\nFULL PREDICTION INTERFACE EVALUATION")
print("====================================")

for index, row in test_data.iterrows():

    # Get original ECG image path
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

    # Preprocess original image
    image_tensor = preprocess_image(str(raw_image_path))

    # Extract CNN features
    with torch.no_grad():
        features = feature_model(image_tensor)
        features = features.reshape(features.size(0), -1)

    # Predict class
    prediction = classifier.predict(
        features.cpu().numpy()
    )[0]

    # Store actual and predicted labels
    label_to_id = {
        name: i for i, name in enumerate(CLASS_NAMES)
    }

    y_true.append(label_to_id[row["label"]])
    y_pred.append(int(prediction))

    print(
        f"Processed image {index + 1}/{len(test_data)}",
        end="\r"
    )


# Calculate final metrics
print("\n\nFULL PIPELINE RESULTS")
print("============================")

print(f"Total images: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(
    f"Macro-F1: "
    f"{f1_score(y_true, y_pred, average='macro'):.4f}"
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )
)

print("\nFull prediction interface evaluation completed.")
