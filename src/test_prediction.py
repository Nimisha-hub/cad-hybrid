
import pandas as pd
import torch
from pathlib import Path

from src.model_config import PROJECT_ROOT, TEST_CSV, CLASS_NAMES
from src.predict_ecg import load_models, preprocess_image


# Load test data
test_data = pd.read_csv(TEST_CSV)

# Load models
feature_model, classifier = load_models()

print("\nTHREE-CLASS PREDICTION ANALYSIS")
print("================================")

for index, row in test_data.head(5).iterrows():

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

    # Preprocess image
    image_tensor = preprocess_image(str(raw_image_path))

    # Extract features
    with torch.no_grad():
        features = feature_model(image_tensor)
        features = features.reshape(features.size(0), -1)

    # Get all three class probabilities
    probabilities = classifier.predict_proba(
        features.cpu().numpy()
    )[0]

    predicted_class = classifier.predict(
        features.cpu().numpy()
    )[0]

    print(f"\nImage {index + 1}")
    print("----------------------------")
    print("Actual class:", row["label"])
    print("Predicted class:", CLASS_NAMES[predicted_class])

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: "
            f"{probabilities[class_id] * 100:.2f}%"
        )

print("\nAnalysis completed.")
