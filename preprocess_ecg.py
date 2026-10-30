
from pathlib import Path
import cv2
import numpy as np


def crop_and_deskew(image):
    """
    Safely preprocess an ECG report.
    Preserve the complete report and waveform regions.
    """

    if image is None:
        raise ValueError("Unable to read ECG image.")

    # Keep the complete report; do not crop automatically.
    processed = image.copy()

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=300,
        maxLineGap=10
    )

    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]

            angle = np.degrees(
                np.arctan2(y2 - y1, x2 - x1)
            )

            if abs(angle) <= 3:
                angles.append(angle)

    # Correct only a small, consistently detected rotation.
    if angles:
        rotation_angle = float(np.median(angles))

        if 0.2 <= abs(rotation_angle) <= 3:
            height, width = processed.shape[:2]
            center = (width // 2, height // 2)

            matrix = cv2.getRotationMatrix2D(
                center, rotation_angle, 1.0
            )

            processed = cv2.warpAffine(
                processed,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    return processed


def resize_ecg(image, target_size=(512, 512)):
    """
    Resize while preserving aspect ratio.
    """

    if image is None:
        raise ValueError("Input image cannot be empty.")

    height, width = image.shape[:2]

    scale = min(
        target_size[0] / width,
        target_size[1] / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


if __name__ == "__main__":

    image_path = Path(
        "data/raw/ecg_multan_pk/ECG Images of Myocardial Infarction Patients (240x12=2880)/MI(1).jpg"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed = crop_and_deskew(image)
    processed = resize_ecg(processed)

    output_path = Path("test_preprocessed.jpg")

    if not cv2.imwrite(str(output_path), processed):
        raise RuntimeError("Unable to save processed image.")

    print("\nUPDATED PREPROCESSING TEST")
    print("--------------------------")
    print("Original dimensions:", image.shape)
    print("Processed dimensions:", processed.shape)
    print("Saved:", output_path)
    print("Completed successfully.")
