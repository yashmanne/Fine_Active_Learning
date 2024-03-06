import wandb
import train_logging
import wandb_utils as wbu

import importlib
importlib.reload(wbu)
datasets = ['DTD']
num_samples = [10]
al_methods = ['K-Mediods']
random_seeds = [1]
hyperparameter_kwargs = {'max_runs':1, 'method':'grid', 'num_epochs':1}
wbu.run_all_combinations(datasets, num_samples, al_methods, random_seeds, hyperparameter_kwargs)