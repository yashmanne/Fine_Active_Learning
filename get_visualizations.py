import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from torchvision.datasets import DTD
from torch.utils.data import DataLoader, Subset

from train_logging import ModelClass
from model_based_RandomStart import ModelAL
import pickle


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
    valid_dict = {}
    for rs in tqdm(seeds):
        rs_dict = {}
        val_inds = {}
        for al_m in tqdm(AL_methods):
            print(f"Evaluating {al_m}")
            al_dict = {}
            for num_s in tqdm(num_samples):
                print(f"Evaluating num_samples {num_s}")
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
                    if val_inds.get(f'n_{num_s}', None) is None:
                        val_inds[f'n_{num_s}'] = valid_indices.tolist()
                else:
                    MC = ModelAL(dataset_name='DTD', AL_method=al_m, num_samples=num_s, seed=rs)
                    valid_set = MC.valid_subset
                    # Train for coldstart time.
                    train_list = []
                    for lr in tqdm(lrs, desc='lr'):
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
                        train_list.append((lr, train_indices.tolist()))
                    num_sample['train_ind'] = train_list

                al_dict[num_s] = num_sample
            rs_dict[al_m] = al_dict
        valid_dict[rs] = val_inds
        result_dict[rs] = rs_dict
    return result_dict, valid_dict


def get_features_from_indices():
    MC = ModelClass(dataset_name='DTD', AL_method='SimpleRandom', num_samples=40, seed=0)
    feature_extractor = nn.Sequential(*list((MC.model).children())[:-1])  # Select all layers except the last]
    feature_extractor.eval()

    full_dl = DataLoader(dataset=MC.train_subset, batch_size=128,
                         shuffle=False, pin_memory=True, num_workers=0)
    # Initialize a list to hold indices for each class
    all_outputs = None
    # Iterate over the dataset and collect feature_vectors
    for batch_id, (images, labels) in enumerate(tqdm(full_dl, desc="Gathering values")):
        images, labels = images.to(MC.device), labels.to(MC.device)
        # Forward pass
        outputs = feature_extractor(images)
        if all_outputs is None:
            all_outputs = outputs
        else:
            all_outputs = torch.concat([all_outputs, outputs])

    return all_outputs


if __name__ == '__main__':
    train_dict, valid_dict = get_model_feature_vectors()
    with open('train_dict.pickle', 'wb') as handle:
        pickle.dump(train_dict, handle)

    with open('valid_dict.pickle', 'wb') as handle:
        pickle.dump(valid_dict, handle)
