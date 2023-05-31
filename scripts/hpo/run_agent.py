import wandb
from sweep import main

sweep_id = "ayousyku"
wandb.agent(sweep_id, function=main)