from medmnist import RetinaMNIST
from torchvision import transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from resnetModel import create_resnet
from unet import UNet
from medmnist import INFO, Evaluator

import os

# Hyperparameters
n_channels = 1  # Number of input channels in the image, e.g., 1 for grayscale
n_classes = 2  # Number of classes, including background
batch_size = 4
learning_rate = 1e-4
num_epochs = 25

def main():
    # Create the directory if it does not exist
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_model_path = os.path.join(model_dir, "best_model.pth")


    # Define the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    data_flag = 'retinamnist'
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])


    # Define a data augmentation pipeline
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale with mean and std dev of 0.5
    ])

    # Initialize the datasets with the transforms
    train_dataset = RetinaMNIST(split='train', transform=data_transforms, download=download)
    val_dataset = RetinaMNIST(split='val', download=download)  # Add transforms if needed
    test_dataset = RetinaMNIST(split='test', download=download)  # Add transforms if needed

    # Data loaders
    batch_size = 32  # You can modify this based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    # resnet_model = create_resnet(num_classes=n_classes, input_channels=n_channels, pretrained=False)
    # resnet_model.to(device)

    unetmodel = UNet(n_channels=n_channels, n_classes=n_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(unetmodel.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Training phase
        unetmodel.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = unetmodel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Scheduler step
        scheduler.step()

        # Validation phase
        unetmodel.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = unetmodel(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


            # Calculate validation accuracy
        val_accuracy = 100 * correct / total

        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {running_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

        # Here you can add code to save the model if it has improved on the validation set
        # Save the model if validation accuracy has improved
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:.2f}% to {val_accuracy:.2f}%. Saving model...')
            best_val_accuracy = val_accuracy
            torch.save(unetmodel.state_dict(), best_model_path)

        # Load the best model once training is complete
        unetmodel.load_state_dict(torch.load(best_model_path))

if __name__ == "__main__":
    main()