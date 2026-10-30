
from src.data_loader import create_data_loaders

train_loader, validation_loader, test_loader = create_data_loaders()

images, labels, groups = next(iter(train_loader))

print("Group-aware DataLoader check passed.")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Number of group IDs:", len(groups))

print("Sample group IDs:", list(groups))
