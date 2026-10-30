
from pathlib import Path

import cv2
import torch
import joblib

from PIL import Image, ImageOps
from torchvision import transforms

from src.model_config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    DEVICE
)

from src.feature_extractor import create_feature_extractor

from preprocess_ecg import (
    crop_and_deskew,
    resize_ecg
)


# Model path
CLASSIFIER_PATH = (
    PROJECT_ROOT / "models" / "ecg_calibrated_classifier.pkl"
)


# Load models only once
_feature_model = None
_classifier = None


# ImageNet normalization
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_models():
    """Load the ResNet18 feature extractor and calibrated SVM."""

    global _feature_model, _classifier

    if _feature_model is None:
        _feature_model = create_feature_extractor()

    if _classifier is None:

        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                f"Calibrated classifier not found: {CLASSIFIER_PATH}"
            )

        _classifier = joblib.load(CLASSIFIER_PATH)

    return _feature_model, _classifier


def preprocess_image(image_path):
    """
    Preprocess an ECG image using the same steps
    used during batch preprocessing and training.
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"ECG image not found: {image_path}"
        )

    # 1. Read original image in BGR format
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read the ECG image.")

    # 2. Apply the original batch preprocessing
    image = crop_and_deskew(image)

    image = resize_ecg(
        image,
        target_size=(512, 512)
    )

    # 3. Reproduce JPEG saving
    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError(
            "Unable to encode processed ECG image."
        )

    # 4. Reproduce JPEG reloading
    image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to decode processed ECG image."
        )

    # 5. Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # 6. Convert to PIL image
    image = Image.fromarray(image)

    # 7. Resize while preserving aspect ratio
    image = ImageOps.contain(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.LANCZOS
    )

    # 8. Create white background
    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        "white"
    )

    x = (IMAGE_SIZE - image.width) // 2
    y = (IMAGE_SIZE - image.height) // 2

    canvas.paste(image, (x, y))

    # 9. Apply ImageNet normalization
    tensor = image_transform(canvas)

    return tensor.unsqueeze(0)


def predict_ecg(image_path: str) -> float:
    """
    Predict calibrated CAD-positive probability.

    Returns:
        float: CAD-positive probability between 0 and 1.
    """

    feature_model, classifier = load_models()

    # Preprocess ECG image
    image_tensor = preprocess_image(image_path).to(DEVICE)

    # Extract ResNet18 features
    with torch.no_grad():

        features = feature_model(image_tensor)

        features = features.reshape(
            features.size(0),
            -1
        )

    # Convert features to NumPy
    features = features.cpu().numpy()

    # Predict calibrated probabilities
    probabilities = classifier.predict_proba(features)

    # Find CAD-positive class (label 0)
    cad_index = list(classifier.classes_).index(0)

    cad_probability = float(
        probabilities[0, cad_index]
    )

    return cad_probability


# Test prediction function
if __name__ == "__main__":

    sample_image = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "ecg_multan_pk"
        / "ECG Images of Myocardial Infarction Patients (240x12=2880)"
        / "MI(1).jpg"
    )

    probability = predict_ecg(str(sample_image))

    print("\nECG PREDICTION")
    print("==========================")

    print(
        f"CAD-positive probability: {probability:.4f}"
    )

    print(
        f"CAD-positive percentage: {probability * 100:.2f}%"
    )
