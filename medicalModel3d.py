from medmnist import OrganMNIST3D
import os
from torchvision import transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from resnetModel import create_resnet_3d

def main():
    # Define the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    download = True
    info = OrganMNIST3D.INFO
    n_classes = len(info['label'])

    # Define a data augmentation pipeline for 3D data
    # Note: 3D transformations are not included in torchvision, and you'd need a custom or specialized library
    data_transforms = transforms.Compose([
        # Add any 3D-specific transforms here
    ])

    # Initialize the datasets with the transforms
    train_dataset = OrganMNIST3D(split='train', transform=data_transforms, download=download)
    val_dataset = OrganMNIST3D(split='val', download=download)  # Add transforms if needed
    test_dataset = OrganMNIST3D(split='test', download=download)  # Add transforms if needed

    # Data loaders for 3D data
    batch_size = 2  # 3D data is larger, so you might need to reduce the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the 3D model
    model_3d = create_resnet_3d(num_classes=n_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3d.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Model directory for saving the best model
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_path = os.path.join(model_dir, "best_model_3d.pth")

    # Training loop
    # ... (similar to the 2D version, but make sure to handle 3D data correctly)

    # Load the best model once training is complete
    model_3d.load_state_dict(torch.load(best_model_path))

# ... (rest of the code including the if __name__ == "__main__": block)