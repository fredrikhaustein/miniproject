from torchvision import models
import torch.nn as nn

def create_custom_cnn(input_channels, num_classes):
    class CustomCNN(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(CustomCNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Assuming the input size is 24x24 after pooling
            self.drop = nn.Dropout(0.25)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.drop(x)
            x = self.fc2(x)
            return x

    return CustomCNN(input_channels, num_classes)
