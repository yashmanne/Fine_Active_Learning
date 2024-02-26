#Alex Netzley
#2/22/2024

import torch
import torchvision
#from torchinfo import summary
import scipy
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

def train_model(lr, num_epochs, num_samples, dataset_name, train_loader, val_loader):

    # Load EfficientNet-B1 model
    model = torchvision.models.efficientnet_b1(weights="DEFAULT")

    if dataset_name =="flowers":
        classes = 102
    elif dataset_name =="aircraft":
        classes = 100

  # update the number of classes in the last layers
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=classes, bias=True)

    # Freeze all layers except the last one
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        early_stopping_patience = 2
        best_val_loss = float('inf')
        patience_counter = 0
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / num_samples
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / num_samples
        epoch_val_acc = correct / total
        val_accs.append(epoch_val_acc)
        val_losses.append(epoch_val_loss)
        print(f"Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model if needed
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    training_stats = train_losses, val_losses, val_accs
    return model, training_stats

