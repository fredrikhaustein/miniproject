from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torch.nn.functional as F
from resnetModel import ResNet, BasicBlock, Bottleneck
import medmnist
from medmnist import INFO, Evaluator

from sklearn.metrics import mean_absolute_error, cohen_kappa_score


data_flag = 'retinamnist'
download = True

NUM_EPOCHS = 100
BATCH_SIZE = 64
lr = 0.001

# MODEL NAME = MEDICAL_MODEL_2D_%EPOCH%_%BATCHSIZE%_%
MODEL_NAME = "medical_model_2D_100_64"

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

# load the data
train_dataset = DataClass(split='train', transform=augmentation_transforms, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# evaluation

def test(split, model):
    model.eval()
    y_true = torch.tensor([], device='cuda')  # Create a tensor on the GPU
    y_score = torch.tensor([], device='cuda')  # Create a tensor on the GPU
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to('cuda')  # Move inputs to GPU
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to('cuda')  # Move targets to GPU and convert to float
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long().to('cuda')  # Move targets to GPU and convert to long
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        # Move the tensors back to CPU for evaluation and numpy conversion
        y_true = y_true.to('cpu').numpy()
        y_score = y_score.to('cpu').detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


class OrdinalRegressionLoss(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()

    def forward(self, output, target):
        # Prepare target to be the same size as output
        target = target.view(-1, 1).float()
        # Create cumulative targets
        cum_target = (target > torch.arange(output.size(1)).float().to(target.device)).float()
        return F.binary_cross_entropy_with_logits(output, cum_target, reduction='mean')

def testVal(split, model, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    data_loader = val_loader if split == 'val' else test_loader  # Choose the correct data loader

    with torch.no_grad():  # No need to track gradients for validation
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    val_accuracy = 100 * correct / total
    print(f'{split.capitalize()} Accuracy: {val_accuracy:.2f}%')
    return val_accuracy

def testVal_mean_kappa(split, model, device):
    model.eval()
    y_true = []
    y_pred = []
    
    data_loader = val_loader if split == 'val' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Convert outputs to probabilities and then to class predictions
            probabilities = torch.sigmoid(outputs).data.cpu().numpy()
            predictions = np.sum(probabilities > 0.5, axis=1) - 1  # Adjusted for zero-indexing
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions)

    # Calculate MAE and Quadratic Weighted Kappa
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return mae, qwk

def main():

    patience = 10  # For early stopping
    best_val_accuracy = 0  # Initialize best validation accuracy
    epochs_no_improve = 0  # Counter for early stopping


     # Print dataset information
    print(train_dataset)
    print("===================")
    print(test_dataset)

    # Define the ResNet models
    def ResNet18(in_channels, num_classes):
        return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)

    def ResNet50(in_channels, num_classes):
        return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

    # Initialize the model
    model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move your model to the GPU

    # Define loss function and optimizer
    if task == "ordinal-regression":
        criterion = OrdinalRegressionLoss()
    elif task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Directory for saving models
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_path = os.path.join(model_dir, MODEL_NAME)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
            else:
                targets = targets.squeeze().long()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Calculate training accuracy
        train_accuracy = 100 * train_correct / train_total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')

        # Validation phase
        val_accuracy = testVal('val', model, device)  # Assuming test function returns accuracy

        # Checkpointing
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:.2f}% to {val_accuracy:.2f}%. Saving model...')
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Validation accuracy did not improve. Count: {epochs_no_improve}/{patience}')

        # Early stopping
        if epochs_no_improve == patience:
            print('Early stopping triggered.')
            break

    # Load the best model once training is complete
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Evaluate on the training and test set
    print('==> Evaluating ...')
    mae_train, qwk_train = testVal_mean_kappa('train', model, device)
    mae_test, qwk_test = testVal_mean_kappa('test', model, device)

    print("Train evaluation: ",mae_train, qwk_train )
    print("Test evaluation: ",mae_test, qwk_test )


if __name__ == "__main__":
    main()