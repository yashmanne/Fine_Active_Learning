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

from train_logging import ModelClass

class ModelAL(ModelClass):
    def __init__(self, dataset_name, preprocess_transform=None, AL_method=None, num_samples=10, seed=1,
                 cold_start_n=5, cold_start_epochs=5, cold_start_method='SimpleRandom'):
        """
        Initializes ModelClass with the specified parameters.

        Parameters:
            dataset_name (str): Name of the dataset.
            preprocess_transform (torchvision.transforms.Compose, optional): Preprocessing transformation for input data.
            AL_method (str, optional): Active learning method. Default is 'Entropy'
                Options: 'Entropy', 'LeastMargin', 'LeastConfidence',
            num_samples (int): Number of samples per class to subset original dataset. Default is 10
            seed (int): Random seed for reproducibility. Default is 1
            cold_start_n (float): number of samples to use for cold-start training. Default is 5
            cold_start_epochs (float): number of epochs to train using cold-start-method.
            cold_start_method (str): Sampling method to initialize model training. Options: 'SimpleRandom', 'StratifiedRandomSample', 'K-Medoids'
        """
        super().__init__(dataset_name=dataset_name, preprocess_transform=preprocess_transform,
                         AL_method=cold_start_method, num_samples=num_samples, seed=seed)
        self.cold_start_num_samples = cold_start_n
        self.cold_start_epochs = cold_start_epochs
        self.cold_start_method = cold_start_method
        # get cold_start_train_subset with correct n_samples
        self.num_samples = self.cold_start_num_samples
        self.train_subset = self.get_data_subsets(split='train')
        self.num_samples = num_samples - self.cold_start_num_samples
        self.cold_start_train_subset = self.train_subset
        # set actual method
        self.AL_method = AL_method if AL_method else 'Entropy'

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
        OG_epochs = config.get("epochs")
        # Call train_model_inner function
        model, best_val_loss = self.train_model_inner(
            model=model, config=config, train_subset=self.cold_start_train_subset,
            epochs=self.cold_start_epochs)
        # Now, get new subset of train_data based on the current model.
        self.model = model
        self.train_subset = self.get_train_subset()

        # Call train_model_inner function
        new_epochs = OG_epochs - self.cold_start_epochs
        model, best_val_loss = self.train_model_inner(
            model=model, config=config, train_subset=self.train_subset,
            epochs=new_epochs, log_test=True)

        # output model
        run_id = wandb.run.id
        wandb.finish()
        if return_model:
            return model, (run_id, best_val_loss)

    def get_train_subset(self):
        full_dataset = self.dataset_module(root="./data/", split='train',
                                           transform=self.preprocess_transform, download=True)
        if self.AL_method == "Entropy":
            subset_indices = self._prob_based_indices(full_dataset, method='Entropy')
        elif self.AL_method == "LeastConfidence":
            subset_indices = self._prob_based_indices(full_dataset, method='LeastConfidence')
        elif self.AL_method == "LeastMargin":
            subset_indices = self._prob_based_indices(full_dataset, method='LeastMargin')
        else:
            raise ValueError(f"AL method {self.AL_method} not implemented")

        sub_dataset = Subset(full_dataset, indices=subset_indices)

        return sub_dataset

    def _prob_based_indices(self, full_dataset, method='Entropy'):
        torch.manual_seed(self.seed)
        total_samples = self.num_samples * self.num_classes
        full_dl = DataLoader(dataset=full_dataset, batch_size=128,
                             shuffle=False, pin_memory=True, num_workers=0)
        # Initialize a list to hold indices for each class
        ind_indices = None
        all_scores = None
        # Iterate over the dataset and collect indices for each class
        model = self.model
        model.eval()
        for batch_id, (images, labels) in enumerate(tqdm(full_dl, desc="Gathering Scores")):
            images, labels = images.to(self.device), labels.to(self.device)
            # Forward pass
            outputs = model(images)
            probs = outputs.softmax(dim=1)  # N x C
            tmp_scores = None
            if method == 'Entropy':
                tmp_scores = torch.sum(-probs * torch.log(probs + 1e-8), dim=1)    # NxC
            elif method == 'LeastConfidence':
                tmp_scores = 1 - torch.max(probs, dim=1).values    # N,
            elif method == 'LeastMargin':
                top_k = torch.topk(probs, 2, dim=1).values
                tmp_scores = 1 - (top_k[:, 0] - top_k[:, 1])

            inds = torch.arange(128) + batch_id*128

            # Update set of final_indices & scores
            # concat if they exist
            if ind_indices is None:
                ind_indices = inds.to(self.device)
                all_scores = tmp_scores.to(self.device)
            else:
                inds = inds.to(self.device)
                tmp_scores = tmp_scores.to(self.device)
                ind_indices = torch.concat([ind_indices, inds])
                all_scores = torch.concat([all_scores, tmp_scores])
            # if the size is too large, keep only the top total_samples
            if len(ind_indices) > total_samples:
                scores, indices = torch.topk(all_scores, total_samples)
                all_scores = scores
                ind_indices = ind_indices[indices]

        return ind_indices

    def train_model_inner(self, model, config, train_subset, epochs=None, log_test=False):
        """
        Trains the machine learning model & logs the runs to WandB.

        Parameters:
            model (model): the EfficientNet-based model
            config (wandb.config): the configuration for a given model run
            train_subset (torch.utils.data.Subset): the subset for the training data
            lr (float): Learning rate.
            batch_size (int): Batch size.
        Returns:
            torch.nn.Module: Trained machine learning model.
        """
        if epochs is None:
            epochs = config.epochs
        torch.manual_seed(self.seed)
        # Get Train, Val, Test DataLoaders
        train_loader = DataLoader(dataset=train_subset, batch_size=config.batch_size,
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
        for epoch in trange(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
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
        # Log best val_loss at the end again
        wandb.log({
            "val/val_loss": best_val_loss,
            "val/val_accuracy": acc_at_best_val_loss
        })
        if log_test:
            test_loss, test_accuracy = self.validate_model(model, valid_dl=test_loader, loss_func=criterion,
                                                           log_images=False)
            wandb.summary['test_accuracy'] = test_accuracy
        return model, best_val_loss

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