"""
paths to the different datasets/models. Feel free to modify
"""

# if using the nail cluster : 
results_dir = "/home/jovyan/EEGClip/results/"
tuh_data_dir = "/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/"

# if using KISlurm :
"""
results_dir = "/home/ndirt/dev/neuro_ai/EEGClip/results/"
tuh_data_dir = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/"
"""

# path to the dataframe containing the precomputed embeddings 
embs_df_path = '/home/jovyan/EEGClip/scripts/text_analysis/report_df_embs.csv'

# path to models trained on various tasks. Handy for baselines comparisons
model_paths = {"eegclip128":"/home/jovyan/EEGClip/results/wandb/EEGClip/1lgwz214/checkpoints/epoch=6-step=42609.ckpt",
              "eegclip":"/home/jovyan/EEGClip/results/wandb/EEGClip/3lh2536v/checkpoints/epoch=19-step=1760.ckpt",
              "pathological_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_few_shot/1vljui8s/checkpoints/epoch=9-step=7100.ckpt",
              "under_50_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_few_shot/akl12j6m/checkpoints/epoch=9-step=7100.ckpt"
            }