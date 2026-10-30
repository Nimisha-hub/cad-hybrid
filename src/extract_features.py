
from pathlib import Path

import numpy as np
import torch

from src.feature_extractor import create_feature_extractor
from src.data_loader import create_data_loaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


def extract_features(model, data_loader, split_name):

    all_features = []
    all_labels = []
    all_groups = []

    model.eval()

    print(f"\nExtracting {split_name} features...")

    with torch.no_grad():

        for batch_index, (images, labels, groups) in enumerate(data_loader):

            images = images.to(DEVICE)

            features = model(images)
            features = torch.flatten(features, start_dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_groups.extend(list(groups))

            if (batch_index + 1) % 10 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(data_loader)}"
                )

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    group_ids = np.array(all_groups, dtype=str)

    print(f"{split_name} features: {X.shape}")
    print(f"{split_name} labels: {y.shape}")
    print(f"{split_name} groups: {group_ids.shape}")

    return X, y, group_ids


if __name__ == "__main__":

    print("Loading ResNet18 feature extractor...")
    model = create_feature_extractor()

    print("Loading ECG datasets...")
    train_loader, validation_loader, test_loader = (
        create_data_loaders()
    )

    X_train, y_train, groups_train = extract_features(
        model, train_loader, "Training"
    )

    X_validation, y_validation, groups_validation = extract_features(
        model, validation_loader, "Validation"
    )

    X_test, y_test, groups_test = extract_features(
        model, test_loader, "Testing"
    )

    np.savez_compressed(
        FEATURE_DIR / "train_features.npz",
        X=X_train,
        y=y_train,
        groups=groups_train
    )

    np.savez_compressed(
        FEATURE_DIR / "validation_features.npz",
        X=X_validation,
        y=y_validation,
        groups=groups_validation
    )

    np.savez_compressed(
        FEATURE_DIR / "test_features.npz",
        X=X_test,
        y=y_test,
        groups=groups_test
    )

    print("\n===================================")
    print("GROUP-AWARE FEATURE EXTRACTION COMPLETED")
    print("===================================")

    print("Training:", X_train.shape, groups_train.shape)
    print("Validation:", X_validation.shape, groups_validation.shape)
    print("Testing:", X_test.shape, groups_test.shape)

    print(f"\nFeatures saved in: {FEATURE_DIR}")
