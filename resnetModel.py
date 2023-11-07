from torchvision import models
import torch.nn as nn

# Define the ResNet model
def create_resnet(num_classes, input_channels, pretrained=True):
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=pretrained)

    # Modify the first convolutional layer to accept single-channel input
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modify the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

    return model

def create_resnet_3d(num_classes, input_channels):
    # Define a simple 3D CNN model that mimics ResNet's initial layers
    class ResNet3D(nn.Module):
        def __init__(self, num_classes, input_channels):
            super(ResNet3D, self).__init__()
            self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
            self.bn1 = nn.BatchNorm3d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            # ... You would need to continue and define the rest of the ResNet blocks using 3D layers

            # This is just a placeholder for the adaptive pooling layer and the final fully connected layer
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(64, num_classes)  # Adjust the size according to your architecture

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # ... The rest of the forward pass through the ResNet blocks

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNet3D(num_classes, input_channels)

