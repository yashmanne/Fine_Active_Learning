import wandb
import wandb_utils as wbu

wandb.login(key='67f26507b9d2f6c1bec3b0101751fa761b3a30d2')
datasets = ['DTD']
num_samples = [10, 20]
al_methods = ['Entropy', 'LeastConfidence', 'LeastMargin']
random_seeds = [0, 1, 2]
hyperparameter_kwargs = {'max_runs':2, 'method':'grid', 'num_epochs':75}
wbu.run_all_combinations(datasets, num_samples, al_methods, random_seeds, hyperparameter_kwargs)