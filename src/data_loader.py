
from torch.utils.data import DataLoader

from src.ecg_dataset import ECGDataset
from src.model_config import (
    TRAIN_CSV,
    VALIDATION_CSV,
    TEST_CSV,
    BATCH_SIZE
)


def create_data_loaders():

    # Load datasets
    train_dataset = ECGDataset(TRAIN_CSV)
    validation_dataset = ECGDataset(VALIDATION_CSV)
    test_dataset = ECGDataset(TEST_CSV)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":

    train_loader, validation_loader, test_loader = (
        create_data_loaders()
    )

    print("ECG DataLoaders created successfully.")

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    # Check one training batch
    images, labels = next(iter(train_loader))

    print("\nSample batch:")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels:", labels.tolist())
