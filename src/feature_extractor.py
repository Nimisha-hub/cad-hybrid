
import torch
import torch.nn as nn
from torchvision import models

from src.model_config import DEVICE, PRETRAINED


def create_feature_extractor():
    """
    Load ImageNet-pretrained ResNet18
    and remove its final classification layer.
    """

    # Load ResNet18
    weights = (
        models.ResNet18_Weights.DEFAULT
        if PRETRAINED
        else None
    )

    model = models.resnet18(weights=weights)

    # Remove the final classification layer
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    # Set model to evaluation mode
    feature_extractor.eval()

    # Freeze pretrained parameters
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor = feature_extractor.to(DEVICE)

    return feature_extractor


if __name__ == "__main__":

    model = create_feature_extractor()

    # Test using a sample image tensor
    sample_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = model(sample_input)

    print("ResNet18 feature extractor created successfully.")
    print("Input shape:", sample_input.shape)
    print("Feature shape:", features.shape)
