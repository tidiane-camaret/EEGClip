# Import the W&B Python Library and log into W&B
import wandb
from scripts.clip.clip_tuh_eeg import run_training
wandb.login()


def main():
    wandb.init(project='EEGClip_HPO')

    model_name = 'EEGClip_' + \
        'lr_' + str(wandb.config.lr) + \
        'weight_decay_' + str(wandb.config.weight_decay) + \
        'string_sampling_' + str(wandb.config.string_sampling) + \
        'projected_emb_dim_' + str(wandb.config.projected_emb_dim) + \
        'num_fc_layers_' + str(wandb.config.num_fc_layers) + \
        'batch_size_' + str(wandb.config.batch_size)
     
    
    run_training(
        lr = wandb.config.lr,
        weight_decay = wandb.config.weight_decay,
        string_sampling = wandb.config.string_sampling,
        projected_emb_dim = wandb.config.projected_emb_dim,
        num_fc_layers = wandb.config.num_fc_layers,
        model_name = model_name,
        batch_size = wandb.config.batch_size
    )

# 2: Define the search space
sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'maximize', 
        'name': 'logreg'
        },
    'early_terminate':
        {'type': 'hyperband',
        'min_iter': 2,
        'eta':2
        },
    'parameters': 
    {
        'lr': {'distribution':'log_uniform_values','min': 1e-5, 'max': 1e-1,},
        'weight_decay': {'distribution':'log_uniform_values','min': 1e-5, 'max': 1e-1,},
        'string_sampling': {'values': [True, False]},
        'projected_emb_dim': {'values': [8, 16, 32, 64, 128]},
        'num_fc_layers': {'values': [1, 2, 3]},
        'batch_size': {'values': [8, 16, 32, 64, 128]}
        
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='EEGClip_HPO'
    )

wandb.agent(sweep_id, function=main, count=10)