#Yash Manne
#2/23/2024

import torch
import torchvision
#from torchinfo import summary
import scipy
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102, FGVCAircraft, Food101
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import numpy as np
import wandb


class ModelClass:

    def __init__(self,
                 dataset_name,
                 preprocess_transform=None,
                 AL_method=None,
                 num_samples=5,
                 seed=1):
        """
        :param dataset:
        :param preprocess_transform:
        :param AL_method:
        """
        self.seed = seed
        self.dataset = dataset_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        all_dataset_modules = {
            "Flowers102": Flowers102,
            "FGVCAircraft": FGVCAircraft,
            "Food101": Food101
        }
        self.dataset_module = all_dataset_modules[self.dataset]
        all_dataset_classes = {
            "Flowers102": 102,
            "FGVCAircraft": 102,
            "Food101": 101,
        }
        self.num_classes = all_dataset_classes[self.dataset]
        self.AL_method = AL_method
        self.num_samples = num_samples
        # Note that the train/test for Flowers102 is 10xNclasses so num_samples must be <10
        if self.dataset == 'Flowers102':
            assert self.num_classes <= 10, f"Expected num_samples <=10, got {self.num_classes}."

        # Define transforms for preprocessing
        if preprocess_transform:
            self.preprocess_transform = preprocess_transform
        else:
            self.preprocess_transform = transforms.Compose([
                transforms.Resize(256),         # Resize the image to 256x256
                transforms.CenterCrop(224),     # Crop the center 224x224 portion of the image
                transforms.ToTensor(),          # Convert the image to a PyTorch tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])  # Normalize the image (ImageNet RGB means)
            ])
        # Get data subsets
        self.train_subset = self.get_data_subsets(split='train')
        self.valid_subset = self.get_data_subsets(split='valid')
        self.test_subset = self.get_data_subsets(split='test')

        # Get Model & Training Parameters
        self.model = self._get_model()

    def get_data_subsets(self, split):
        "Get a subsets of the data for the train, val, test"
        full_dataset = self.dataset_module(root="./data/", split=split,
                                           transform=self.preprocess_transform, download=True)
        # self.full_dataset = full_dataset
        sample_method = self.AL_method if split=="train" else "StratifiedRandomSamples"
        if split != "test":
            subset_indices = self._data_subset_indices(full_dataset, sample_method=sample_method)
            sub_dataset = Subset(full_dataset, indices=subset_indices)
            return sub_dataset
        return full_dataset

    def _data_subset_indices(self, full_dataset, sample_method=None):
        if sample_method is None:
            sample_method = self.AL_method
        if sample_method == 'SimpleRandom':
            torch.seed(self.seed)
            subset_indices = torch.randperm(len(full_dataset))[:self.num_samples*self.num_classes]

        elif sample_method == 'StratifedRandomsSamples':
            subset_indices = self._get_stratified_random_sample(full_dataset)
        else:
            raise ValueError(f"AL method {sample_method} not implemented")
        return subset_indices

    def _get_stratified_random_sample(self, full_dataset):
        np.random.seed(self.seed)
        # Initialize a list to hold indices for each class
        class_indices = [[] for _ in range(self.num_classes)]
        # Iterate over the dataset and collect indices for each class
        for idx, (image, label) in enumerate(full_dataset):
            class_indices[label].append(idx)
        # Gather indices for shuffles
        subset_indices = []
        for indices in class_indices:
            np.random.shuffle(indices)
            subset_indices.extend(indices[:self.num_samples])
        return torch.tensor(subset_indices)

    def _get_model(self):
        "A simple model"
        # Load EfficientNet-B1 model
        model = torchvision.models.efficientnet_b1(weights="DEFAULT")
        # update the number of classes in the last layers
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features,
                                         out_features=self.num_classes, bias=True)
        # Freeze all layers except the last one
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
        return model

    def validate_model(self, model, valid_dl, loss_func, log_images=False, batch_idx=0):
        "Compute performance of the model on the validation dataset and log a wandb.Table"
        model.eval()
        val_loss = 0.
        with torch.inference_mode():
            correct = 0
            for i, (images, labels) in enumerate(valid_dl):
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = model(images)
                val_loss += loss_func(outputs, labels)*labels.size(0)
                # Compute accuracy and accumulate
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Log one batch of images to the dashboard, always same batch_idx.
                if (i==batch_idx) and log_images:
                    log_image_table(images, predicted, labels, outputs.softmax(dim=1))
        return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

    def train_model(self, lr, batch_size, num_epochs):
        torch.seed(self.seed)
        model = self.model
        # Initialize WandB
        wandb.init(
            project=f"{self.dataset}-{self.num_samples}-{self.AL_method}",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
            })

        # Get Train, Val, Test DataLoaders
        train_loader = DataLoader(dataset=self.train_subset, batch_size=batch_size,
                                  shuffle=True, pin_memory=True, num_workers=2)
        val_loader = DataLoader(dataset=self.valid_subset, batch_size=batch_size,
                                shuffle=False, pin_memory=True, num_workers=2)
        test_loader = DataLoader(dataset=self.test_subset, batch_size=batch_size,
                                 shuffle=False, pin_memory=True, num_workers=2)

        # Copy your config
        config = wandb.config
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Chekpointing variables
        early_stopping_patience = 2
        best_val_loss = np.inf
        patience_counter = 0

        # Start training
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('-' * 10)
            # Training Phase
            model.train()
            running_loss = 0.0
            for t_i, (images, labels) in enumerate(tqdm(train_loader)):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                # Log metrics in WandB
                metrics = {"train/train_loss": loss}
                # log metrics if not at the end of epoch (if at the end, log the val metrics together
                # if t_i < len(train_loader)
                wandb.log(metrics)

            # Validation phase
            model.eval()
            # store images if needed
            val_loss, val_accuracy = self.validate_model(model, valid_dl=val_loader, loss_func=criterion, log_images=False)
                                                     # log_images=(epoch == (num_epochs - 1)))
            epoch_loss = running_loss / len(train_loader.dataset)
            # Log epoch metrics
            epoch_metrics = {
                "train/epoch_loss": epoch_loss,
                "val/val_loss": val_loss,
                "val/val_accuracy": val_accuracy}
            wandb.log(epoch_metrics)
            print(f"Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
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
        # Log best model to Wandb (storing as dataset to avoid interfacing with model registry)
        artifact = wandb.Artifact(name="best-model", type="dataset")
        artifact.add_file(local_path='./best_model.pth')
        wandb.log_artifact(artifact)

        # Calculate best test performance
        model.load_state_dict(torch.load('./best_model.pth'))
        model.eval()
        test_loss, test_accuracy = self.validate_model(model, valid_dl=test_loader, loss_func=criterion, log_images=False)
        return model


# Auxillary Functions
def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)

