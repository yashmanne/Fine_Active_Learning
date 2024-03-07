import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from torchvision.datasets import DTD
from torch.utils.data import DataLoader, Subset

from train_logging import ModelClass
from model_based_RandomStart import ModelAL

def get_features_from_images(full_dataset, ):
    pass

def get_model_feature_vectors(AL_methods=None, seeds=None, num_samples=None,
                              lrs=None):
    if AL_methods is None:
        AL_methods = ['SimpleRandom', 'StratifiedRandomSample', 'K-Medoids', "LSS",
                       'Entropy', 'LeastConfidence', 'LeastMargin']
    if seeds is None:
        seeds = [0, 1, 2]
    if num_samples is None:
        num_samples = [10, 20]
    if lrs is None:
        lrs = [1e-3, 5e-4]
    result_dict = {}
    for rs in tqdm(seeds):
        rs_dict = {}
        for al_m in AL_methods:
            al_dict = {}
            for num_s in tqdm(num_samples):
                num_sample = {}
                if al_m in ['SimpleRandom', 'StratifiedRandomSample', 'K-Medoids', 'LSS']:
                    MC = ModelClass(dataset_name='DTD', AL_method=al_m, num_samples=num_s, seed=rs)
                    # Gather TrainSet & Indices:
                    train_set = MC.train_subset
                    train_indices = train_set.indices
                    # Gather Validation Images & Indices:
                    valid_set = MC.valid_subset
                    valid_indices = valid_set.indices
                    num_sample['train_ind'] = train_indices
                else:
                    MC = ModelAL(dataset_name='DTD', AL_method=al_m, num_samples=num_s, seed=rs)
                    valid_set = MC.valid_subset
                    # Train for coldstart time.
                    train_list = []
                    for lr in lrs:
                        config = {
                         "lr": lr,
                         "batch_size": 128
                        }
                        MC.model = MC._get_model()
                        model, best_val_loss = MC.train_model_inner(
                            model=MC.model, config=config, train_subset=MC.cold_start_train_subset,
                            epochs=MC.cold_start_epochs, log_wandb=False)
                        MC.model = model
                        train_set = MC.get_train_subset()
                        train_indices = train_set.indices
                        train_list.append((lr, train_indices))
                    num_sample['train_ind'] = train_list


def get_features_from_indices(seed):
    MC = ModelClass(dataset_name='DTD', AL_method='SimpleRandom', num_samples=40, seed=0)
    feature_extractor = nn.Sequential(*list((MC.model).children())[:-1])  # Select all layers except the last]
    feature_extractor.eval()

    full_dl = DataLoader(dataset=MC.train_subset, batch_size=128,
                         shuffle=False, pin_memory=True, num_workers=0)
    # Initialize a list to hold indices for each class
    ind_indices = None
    all_scores = None
    # Iterate over the dataset and collect feature_vectors
    for batch_id, (images, labels) in enumerate(tqdm(full_dl, desc="Gathering values")):
        images, labels = images.to(MC.device), labels.to(MC.device)
        # Forward pass
        outputs = feature_extractor(images)

