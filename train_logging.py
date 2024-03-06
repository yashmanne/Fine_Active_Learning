# Yash Manne
# 2/23/2024

import torch
import torchvision
# from torchinfo import summary
import scipy
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102, FGVCAircraft, DTD
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm, trange
import numpy as np
import wandb
import copy
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from scipy.spatial import distance

class ModelClass:
    """
    A class to manage EfficientNet Finetuning pipeline on Flowers102, FGVCAircraft, and Food101 datasets.

    Attributes:
        dataset (str): Name of the dataset. Options: Flowers102, FGVCAircraft, DTD
        preprocess_transform (torchvision.transforms.Compose): Preprocessing transformation for input data.
        AL_method (str): Active learning method used for subset selection
        num_samples (int): Number of samples per class to subset for fine tuning
        seed (int): Random seed for reproducibility.
        num_classes (int): Number of classes in the dataset.
        dataset_module (torchvision.datasets): PyTorch dataset module.
        train_subset (torch.utils.data.Subset): Subset of the training data.
        valid_subset (torch.utils.data.Subset): Subset of the validation data.
        test_subset (torch.utils.data.Dataset): Test dataset.
        model (torch.nn.Module): EfficientNet B1 with modified last affine layer.
    """

    def __init__(self,
                 dataset_name,
                 preprocess_transform=None,
                 AL_method=None,
                 num_samples=5,
                 seed=1):
        """
        Initializes ModelClass with the specified parameters.

        Parameters:
            dataset_name (str): Name of the dataset.
            preprocess_transform (torchvision.transforms.Compose, optional): Preprocessing transformation for input data.
            AL_method (str, optional): Active learning method. Default is SimpleRandom
                Options: 'SimpleRandom', 'StratifiedRandomSample', 'K-Medoids', 'LSS'
            num_samples (int): Number of samples per class to subset original dataset. Default is 5
            seed (int): Random seed for reproducibility. Default is 1
        """
        self.seed = seed
        self.dataset = dataset_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        all_dataset_modules = {
            "Flowers102": Flowers102,
            "FGVCAircraft": FGVCAircraft,
            "DTD": DTD
        }
        self.dataset_module = all_dataset_modules[self.dataset]
        all_dataset_classes = {
            "Flowers102": 102,
            "FGVCAircraft": 100,
            "DTD": 47,
        }
        self.num_classes = all_dataset_classes[self.dataset]
        self.AL_method = AL_method if AL_method else 'SimpleRandom'
        self.num_samples = num_samples
        # Note that the train/test for Flowers102 is 10xNclasses so num_samples must be <10
        if self.dataset == 'Flowers102':
            assert self.num_samples <= 10, f"Expected num_samples <=10, got {self.num_samples}."
        # Note that the train/test for DTD is 40xNclasses so num_samples must be <10
        if self.dataset == 'FGVCAircraft':
            assert self.num_samples <= 100, f"Expected num_samples <=100, got {self.num_samples}."
        # Note that the train/test for DTD is 40xNclasses so num_samples must be <10
        if self.dataset == 'DTD':
            assert self.num_samples <= 40, f"Expected num_samples <=40, got {self.num_samples}."

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
        self.valid_subset = self.get_data_subsets(split='val')
        self.test_subset = self.get_data_subsets(split='test')

        # Get Model & Training Parameters
        self.model = self._get_model()

    def get_data_subsets(self, split):
        """
        Gets subsets of the data for the specified split.

        Parameters:
            split (str): Split of the data ('train', 'val', or 'test').

        Returns:
            torch.utils.data.Subset or torch.utils.data.Dataset: Subset or dataset for the specified split.
        """
        full_dataset = self.dataset_module(root="./data/", split=split,
                                           transform=self.preprocess_transform, download=True)
        # self.full_dataset = full_dataset
        sample_method = self.AL_method if split == "train" else "StratifiedRandomSample"
        if split != "test":
            subset_indices = self._data_subset_indices(full_dataset, sample_method=sample_method)
            sub_dataset = Subset(full_dataset, indices=subset_indices)
            return sub_dataset
        return full_dataset

    def _data_subset_indices(self, full_dataset, sample_method=None):
        """
        Generates indices for creating a subset of the dataset
        using a specified sampling method. The default is the
        class's Active learning method.

        Args:
           full_dataset (torch.utils.data.Dataset): Full dataset.
           sample_method (str, optional): Sampling method.

        Returns:
           torch.Tensor: Indices for the subset.
        """
        print(sample_method)
        if sample_method is None:
            sample_method = self.AL_method
        if sample_method == 'SimpleRandom':
            torch.manual_seed(self.seed)
            subset_indices = torch.randperm(len(full_dataset))[:self.num_samples * self.num_classes]

        elif sample_method == 'StratifiedRandomSample':
            subset_indices = self._get_stratified_random_sample(full_dataset)
        elif sample_method == 'K-Medoids':
            subset_indices = self._get_K_medoids_sample(full_dataset)
        elif sample_method == 'LSS':
            subset_indices = self._get_LSS_sample(full_dataset)
        else:
            raise ValueError(f"AL method {sample_method} not implemented")
        return subset_indices

    def _get_stratified_random_sample(self, full_dataset):
        """
        Generates a stratified random sample of indices from the dataset.

        Parameters:
            full_dataset (torch.utils.data.Dataset): Full dataset.

        Returns:
            torch.Tensor: Indices for the stratified random sample.
        """
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

    def _get_K_medoids_sample(self, full_dataset):
        #Extract the features of all unlabelled datapoints
        feature_extractor = nn.Sequential(*list((self._get_model()).children())[:-1]) #Select all layers except the last]
        feature_extractor.eval()

        #print('Feature Extractor:', feature_extractor)
        train_features = []
        for image, _ in full_dataset:
            with torch.no_grad():
                features = feature_extractor(image.unsqueeze(0))
            train_features.append(features.squeeze().flatten().numpy())
        train_features = np.array(train_features)

        full_dataset = train_features
        print('Full Dataset Shape:', full_dataset.shape)

        #Fit K-Medoids on the full dataset
        num_clusters = self.num_classes
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=self.seed)
        kmedoids.fit(full_dataset)

        # Get the centers of the clusters
        cluster_centers = torch.tensor(full_dataset[kmedoids.medoid_indices_])

        # Initialize labeled and unlabeled datasets
        subset_indices = []
        #unlabeled_indices = list(range(len(full_dataset)))  # Initially, all indices are unlabeled

        subset_indices = []
        # shuffle distance matrix
        print('Kmed ind shape: ', kmedoids.medoid_indices_.shape)
        kmedoids.medoid_indices_ = kmedoids.medoid_indices_[np.random.permutation(num_clusters)]

        while (len(subset_indices)) < num_clusters*self.num_samples:
            this_loop_list = []
            for i, medoid_idx in enumerate(kmedoids.medoid_indices_):
                cluster_indices = (kmedoids.labels_ == i)
                cluster_data_indices = np.where(cluster_indices)[0]  # Get indices directly using np.where

                #MASK
                cluster_data_indices = np.setdiff1d(cluster_data_indices, subset_indices)

                #changed this from cdi to cd
                cluster_data = full_dataset[cluster_data_indices]

                # Calculate distances to the medoid
                distances_to_medoid = np.linalg.norm(cluster_data - cluster_centers[i].numpy(), axis=1)

                # Mask Previous
                print('dtm shape:', distances_to_medoid.shape)
                # og_indices = [cluster_data_indices[idx] for idx in np.argsort(distances_to_medoid)]
                # filtered_og_indices = np.setdiff1d(og_indices, subset_indices)

                # Sort distances and extract the top 10 indices
                closest_to_medoid_indices = np.argsort(distances_to_medoid)[:self.num_samples]
                print('ctm shpae:', closest_to_medoid_indices.shape)
                closest_to_medoid_indices = [cluster_data_indices[idx] for idx in closest_to_medoid_indices]  # Map back to original indices

                #Mask 
                filtered_og_indices = np.setdiff1d(closest_to_medoid_indices, subset_indices)
                this_loop_list.extend(closest_to_medoid_indices)
                # add mask to set values to np.inf
                # reloop to ensure len(subset_indices.unqiue() is the expected number of samples)
            subset_list = np.unique(this_loop_list)
            remain_add = (num_clusters*self.num_samples) - len(subset_list)
            subset_indices.extend(subset_list[:remain_add])


        #subset_indices = np.unique(subset_indices)  # Take unique indices
        print('indices:', subset_indices)
        print('Compairson of unique: ', len(subset_indices), len(np.unique(subset_indices)))
        return torch.tensor(subset_indices)

    def _get_LSS_sample(self, full_dataset):
        ### LSS STEP 1: FEATURE EXTRACTOR ###

        #Extract the features of all unlabelled datapoints
        feature_extractor = nn.Sequential(*list((self._get_model()).children())[:-1]) #Select all layers except the last]
        feature_extractor.eval()

        #print('Feature Extractor:', feature_extractor)
        train_features = []
        for image, _ in full_dataset:
            with torch.no_grad():
                features = feature_extractor(image.unsqueeze(0))
            train_features.append(features.squeeze().flatten().numpy())
        train_features = np.array(train_features)

        full_dataset = train_features
        print('Full Dataset Shape:', full_dataset.shape)
        
        ### Normalize the feature vectors ###
        full_dataset = (full_dataset - np.mean(full_dataset, axis = 1).reshape(-1, 1))/np.std(full_dataset, axis = 1).reshape(-1, 1)

        ###  "Finally we smooth them by representing the data as a graph and propagating the features of the data points to their closest neighbors... " ###

        np.random.seed(self.seed)
        ### Apply soft k-means clustering ###
        def soft_kmeans(X, k, beta=1.0, max_iters=100, tol=1e-4):
            best_centroids_sum = np.inf
            best_centroids = None
            best_soft_assignments = None

            #test out multiple orientations
            for i in range(3):
                n_samples, n_features = X.shape
                # TODO multiple intializations and track one with least overall distances
                # Initialize centroids randomly
    
                centroids = X[np.random.choice(n_samples, k, replace=False)]
                
                for _ in range(max_iters):
                    # Compute distances and soft assignments
                    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                    print(distances.shape)
                    soft_assignments = np.exp(-distances ** 2 / beta)
                    print('SA', soft_assignments.shape)
                    soft_assignments /= np.sum(soft_assignments, axis=0, keepdims=True)
                    
                    # Update centroids
                    new_centroids = np.dot(soft_assignments.T, X) / np.sum(soft_assignments, axis=0, keepdims=True).T
                    
                    # Check for convergence
                    if np.linalg.norm(new_centroids - centroids) < tol:
                        break
                    
                    centroids = new_centroids

                if np.sum(centroids) < best_centroids_sum:
                    best_centroids = centroids
                    best_soft_assignments = soft_assignments
                    best_centroids_sum = np.sum(centroids)
            
            return best_centroids, best_soft_assignments
        
        k = self.num_classes

        # Run Soft K-means algorithm
        centroids, soft_assignments = soft_kmeans(full_dataset, k)
        
        ### Compute LPR ###

        #Compute Log Density
        #What should the k be here since it's a soft assignment?
        #log_density = np.log(1/(np.sqrt(2*np.pi)*np.std(Each_k))) - (1/(2*np.std(Each_k)))*(full_dataset - np.mean(Each_k)**2) #shape: (num_samples, 47)
        #lprk = log_density/np.sum(log_density, axis=1) #Shape: (num_samples, 47)
        #I think we jsut need the last one?
        soft_assignments = soft_assignments #Shape: (num_samples, 47)
        full_dataset = full_dataset #Shape: (num_samples, feature_dim)
        print('Soft Assignments Shape:', soft_assignments.shape)
        sum_zs_in_cf = np.matmul(soft_assignments.T, full_dataset) #Shape: (47, feature_dim), I'm thinking this is the weighted probability per class
        abs_cf = np.sum(soft_assignments, axis = 0)[:, np.newaxis] #Shape: (47), I'm thinking this should be the total sum of the probabilities for each class
        mean_cf = sum_zs_in_cf/abs_cf #shape: (47, feature_dim)
        print('Mean_cf Shape:', mean_cf.shape)
        
        lpr = np.zeros_like(soft_assignments)#Shape: (num_samples, 47)
        for ki in range(mean_cf.shape[0]):
            mean_ki = mean_cf[ki].reshape(1, -1)
            this_lpr = np.sum(((full_dataset - mean_ki)**2), axis = 1) #(num_samples,)
            lpr[:, ki] = this_lpr
        lpr = lpr/lpr.sum(axis=1).reshape(-1,1)
        print('lpr shape: ', lpr.shape)

        ### Iterate through the sampling process ###
        subset_indices = []
        #On the first iteration, select the sample with the lowest LPR (The most confident one)
        min_row_indices = np.argmin(lpr, axis=0)
        subset_indices.extend(min_row_indices)

        #Masked out the indices that we already chose
        lpr[subset_indices, :] = 0
        #For the next N-1 samples, select the N-1 samples with the highest LPR (The least confident ones)
        # Initialize an empty list to store the indices

        # Loop through each column
        #Shuffle columns
        lpr = lpr[:, np.random.permutation(k)]
        while len(subset_indices) < k*self.num_samples:
            subset_indices_list = []
            for col in range(k):
                # Find the row indices where the current column is the highest
                max_column_indices = np.argmax(lpr[:, col])
                # Filter the rows where the current column is the highest
                rows_with_max_column = lpr[lpr[:, col] == lpr[max_column_indices, col]]
                # Find the row indices where the current column is the lowest among the filtered rows
                num_extra_samples = self.num_samples - 1
                min_row_indices = np.argsort(rows_with_max_column[:, col])[:num_extra_samples]
                # Get the original indices of the filtered rows
                original_indices = np.where(np.isin(lpr, rows_with_max_column[min_row_indices]))
                # Append the original indices to the list
                subset_indices_list.extend(np.unique(original_indices[0]))
                # mask out values we already append it
                lpr[np.unique(original_indices[0]), :] = 0
            subset_indices_list = np.unique(subset_indices_list)
            remain_add = (k*self.num_samples) - len(subset_indices)
            subset_indices.extend(subset_indices_list[:remain_add])



        # Convert the list to a numpy array
        # subset_indices_array = np.array(subset_indices_list)
        # print('si array shape: ', subset_indices_array.shape)
        # subset_indices.extend(subset_indices_array)

        print(subset_indices)
        print(len(subset_indices), len(np.unique(subset_indices)))
        return torch.tensor(subset_indices)
    
    
    def _get_model(self):
        """
        Utility function that initializes and configures the machine learning model
        to have the right # of output for the final layer.

        Returns:
            torch.nn.Module: Machine learning model.
        """
        torch.manual_seed(self.seed)
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
        """
        Computes the performance of the model on the validation dataset.

        Parameters:
            model (torch.nn.Module): Machine learning model.
            valid_dl (torch.utils.data.DataLoader): Validation data loader.
            loss_func (torch.nn.Module): Loss function.
            log_images (bool, optional): Whether to log images to dashboard.
            batch_idx (int, optional): Index of the batch to log images.

        Returns:
            tuple: Validation loss and accuracy.
        """
        model.eval()
        val_loss = 0.
        with torch.inference_mode():
            correct = 0
            for i, (images, labels) in enumerate(valid_dl):
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = model(images)
                val_loss += loss_func(outputs, labels) * labels.size(0)
                # Compute accuracy and accumulate
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Log one batch of images to the dashboard, always same batch_idx.
                if (i == batch_idx) and log_images:
                    self.log_image_table(images, predicted, labels, outputs.softmax(dim=1))
        return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

    def train_model(self, wandb_config=None, lr=None,
                    batch_size=None, num_epochs=None, return_model=True):
        """
        Trains the machine learning model & logs the runs to WandB.

        Parameters:
            wandb_config (wandb.config): the configuration for a given model run
            lr (float): Learning rate.
            batch_size (int): Batch size.
            num_epochs (int): Number of epochs.
            return_model (bool): Whether or not to return anything
        Returns:
            Tuple:
                torch.nn.Module: Trained machine learning model.
                Tuple: Model WandB RunID and best validation loss
        """
        torch.manual_seed(self.seed)
        model = copy.deepcopy(self.model)
        # Initialize WandB
        if wandb_config:
            wandb.init(config=wandb_config)
        else:
            wandb.init(
                project=f"{self.dataset}-{self.num_samples}-{self.AL_method}-{self.seed}",
                config={
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                })
        config = wandb.config
        # Get Train, Val, Test DataLoaders
        train_loader = DataLoader(dataset=self.train_subset, batch_size=config.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=0)
        val_loader = DataLoader(dataset=self.valid_subset, batch_size=config.batch_size,
                                shuffle=False, pin_memory=True, num_workers=0)
        test_loader = DataLoader(dataset=self.test_subset, batch_size=config.batch_size,
                                 shuffle=False, pin_memory=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Checkpointing variables
        early_stopping_patience = 5
        best_val_loss = np.inf
        acc_at_best_val_loss = 0
        patience_counter = 0

        # Start training
        for epoch in trange(config.epochs):
            print(f"Epoch {epoch + 1}/{config.epochs}")
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

            # Validation phase
            model.eval()
            # store images if needed
            val_loss, val_accuracy = self.validate_model(model, valid_dl=val_loader, loss_func=criterion,
                                                         log_images=False)
                                                     # log_images=(epoch == (config.epochs - 1)))
            epoch_loss = running_loss / len(train_loader.dataset)
            # Log epoch metrics in wandb
            epoch_metrics = {
                "train/epoch_loss": epoch_loss,
                "val/val_loss": val_loss,
                "val/val_accuracy": val_accuracy}
            wandb.log(epoch_metrics)
            print(
                f"Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                acc_at_best_val_loss = val_accuracy
                patience_counter = 0
                # Save the best model if needed
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs.')
                    break
        # Log the best model to Wandb
        # (dataset so that we log a model for each run rather than 1 per project)
        artifact = wandb.Artifact(name="best-model", type="dataset")
        artifact.add_file(local_path='./best_model.pth')
        wandb.log_artifact(artifact)

        # Calculate best test performance
        model.load_state_dict(torch.load('./best_model.pth'))
        # self.model = model
        model.eval()
        test_loss, test_accuracy = self.validate_model(model, valid_dl=test_loader, loss_func=criterion,
                                                       log_images=False)
        # Log best val_loss at the end again
        wandb.log({
            "val/val_loss": best_val_loss,
            "val/val_accuracy": acc_at_best_val_loss
        })
        wandb.summary['test_accuracy'] = test_accuracy
        run_id = wandb.run.id
        wandb.finish()
        if return_model:
            return model, (run_id, best_val_loss)

    def hyperparameter_sweep(self, method='random', parameters_dict=None, max_runs=10,
                             early_terminate_args=None, num_epochs=50):
        """
        Perform a Wandb hyperparameter sweep on existing model, dataset, num-samples, and seed.
        Note: this might track the final ValAcc instead of the best ValAcc.
        Parameters:
            method (str): the method to choose hyperparameters.
                Options: 'grid', 'random', 'bayes'
            max_runs (int): the number of runs to consider for hyperparameter tuning. Default=10
            parameters_dict (dict): a dictionary of hyperparameters to consider for model training.
            early_terminate_args (dict): dict of args for Hyperband early termination. Default None
            num_epochs (int): number of epochs to run each sweep. Default is 50
        Returns:
            None
        """
        # check method validity
        valid_methods = ['random', 'grid', 'bayes']
        assert method in valid_methods, f"Invalid method {method}, expected one of {valid_methods}."

        # get default parameter_dict with distribution of min/max
        if parameters_dict is None:
            if method == 'grid':
                lr_dict = {'values': [1e-3, 5e-4]}
                bs_dict = {'value': 128}
            else:
                lr_dict = {
                    'distribution': 'q_log_uniform_values',
                    'max': 1e-1,
                    'min': 1e-6,
                    'q': 1e-8
                }
                bs_dict = {
                    'distribution': 'q_log_uniform_values',
                    'max': 128,
                    'min': 8,
                    'q': 8
                }
            parameters_dict = {
                'epochs': {'value': num_epochs},
                'lr': lr_dict,
                'batch_size': bs_dict
            }

        # early_terminate?
        # https://arxiv.org/pdf/1603.06560.pdf
        # https://open.gitcode.host/wandb-docs/sweeps/configuration.html#stopping-criteria
        # https://2020blogfor.github.io/posts/2020/04/hyperband/
        # essentially checks at various iterations that our metric is logged (epochs for us)
        # whether or not to keep training.
        # ex. training method of s=3 and eta=2, max_iter checks at epoch: 6, 12 & 25
        #     training method of eta=2, min_iter=5, checks at epoch 5, 10, 20, 40
        if early_terminate_args is None:
            early_terminate_args = {
                'type': 'hyperband',
                'min_iter': 5,  # minimum number of epochs to train before considering stopping
                'eta': 2,       # the bracket multiplier
                # 's': 2,         # total number of brackets
                # 'max_iter': parameters_dict['epochs']['value'],
            }
        # set up sweep config dictionary
        sweep_config = {
            'method': method,
            'metric': {'name': 'val/val_loss', 'goal': 'minimize'},
            'parameters': parameters_dict,
            'early_terminate': early_terminate_args
        }

        # delete early terminate if early_terminate_args doesn't have the right type
        if early_terminate_args.get('type', None) is None:
            del sweep_config['early_terminate']

        # Get sweep_id that generates the next set of hyperparameters to try
        sweep_id = wandb.sweep(sweep=sweep_config,
                               project=f"{self.dataset}-{self.num_samples}-{self.AL_method}-{self.seed}")

        # get original state_dict
        # og_state_dict = copy.deepcopy(self.model.state_dict())
        # define wrapper of objective function with 1 argument.
        def objective_func(config=None):
            self.model = self._get_model()
            self.train_model(wandb_config=config, return_model=False)

        # Get Agent to run sweeps
        wandb.agent(sweep_id, function=objective_func, count=max_runs)

        return

    # Auxillary Functions
    def log_image_table(self, images, predicted, labels, probs):
        "Log a wandb.Table with (img, pred, target, scores)"
        # 🐝 Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)])
        for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
        wandb.log({"predictions_table": table}, commit=False)
