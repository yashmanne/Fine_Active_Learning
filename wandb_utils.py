import wandb
from model_based_RandomStart import ModelAL
from tqdm.auto import tqdm
from train_logging import ModelClass

# Run All Combinations
def run_all_combinations(datasets, num_samples, al_methods, random_seeds, hyperparameter_kwargs):
    """
    Instantiate a ModelClass for each dataset-num_samples-AL-method-random_seed and
    use the ModelClass.hyperparameter_sweep()

    Parameters:
        datasets (list): strings of datasets to use (must be defined within ModelClass)
        num_samples (list): ints denoting the number of samples per class.
        al_methods (list): strs denoting which AL_method to use to subset the train set.
        random_seeds (list): ints denoting the random seed for random generators in the selection/training process
        hyperparameter_kwargs (dict): relevant arguments for ModelClass.hyperparameter_sweep().
    Returns:
        None
    """
    for dataset in tqdm(datasets):
        for num_s in tqdm(num_samples):
            for al_m in tqdm(al_methods):
                for rs in tqdm(random_seeds):
                    try:
                        if al_m in ['SimpleRandom', 'StratifiedRandomSample', 'K-Medoids', 'LSS']:
                            MC = ModelClass(dataset_name=dataset, AL_method=al_m, num_samples=num_s, seed=rs)
                        else:
                            MC = ModelAL(dataset_name=dataset, AL_method=al_m, num_samples=num_s, seed=rs)
                    except Exception as E:
                        print(f"ModelClass can't be instantiated with this combination {dataset}, {num_s}, {al_m}, {rs}.")
                        print(E)
                        continue

                    try:
                        MC.hyperparameter_sweep(**hyperparameter_kwargs)
                    except Exception as E:
                        print(f"An error occurred during the hyperparameter sweep.")
                        print(E)
    return


def get_results(projects):
    """
    Set of Projects
    :param projects:
    :return:
    """
    pass