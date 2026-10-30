
import pandas as pd
import torch

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from src.model_config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    CLASS_NAMES
)


# Convert class names into numerical labels
LABEL_MAP = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ImageNet normalization
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ECGDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        required_columns = {
            "path",
            "label",
            "dup_group"
        }

        if not required_columns.issubset(self.data.columns):
            raise ValueError("CSV is missing required columns.")

        if self.data.empty:
            raise ValueError("Dataset CSV is empty.")

        unknown_labels = set(self.data["label"]) - set(LABEL_MAP)

        if unknown_labels:
            raise ValueError(f"Unknown labels: {unknown_labels}")

        if self.data["dup_group"].isnull().any():
            raise ValueError("Missing duplicate group identifiers.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # Load ECG image
        image_path = PROJECT_ROOT / row["path"]

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            # Preserve aspect ratio
            image = ImageOps.contain(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.LANCZOS
            )

            # Add white padding
            canvas = Image.new(
                "RGB",
                (IMAGE_SIZE, IMAGE_SIZE),
                color="white"
            )

            x_offset = (IMAGE_SIZE - image.width) // 2
            y_offset = (IMAGE_SIZE - image.height) // 2

            canvas.paste(image, (x_offset, y_offset))

        # Convert image into tensor
        image_tensor = IMAGE_TRANSFORM(canvas)

        # Numerical class label
        label = LABEL_MAP[row["label"]]

        # Duplicate group identifier
        duplicate_group = str(row["dup_group"])

        return (
            image_tensor,
            torch.tensor(label, dtype=torch.long),
            duplicate_group
        )


if __name__ == "__main__":

    from src.model_config import TRAIN_CSV

    dataset = ECGDataset(TRAIN_CSV)

    image, label, duplicate_group = dataset[0]

    print("ECG dataset loaded successfully.")
    print("Total training images:", len(dataset))
    print("Image tensor shape:", image.shape)
    print("Numerical label:", label.item())
    print("Class name:", CLASS_NAMES[label.item()])
    print("Duplicate group:", duplicate_group)
