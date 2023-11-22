import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
"""
paths to the different datasets/models. Feel free to modify
"""

# if using the nail cluster : 
"""
results_dir = "/home/jovyan/EEGClip/results/"
tuh_data_dir = "/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/"
embs_df_path = '/home/jovyan/EEGClip/scripts/text_preprocessing/embs_df.csv'

"""
# if using KISlurm :

results_dir = "/home/ndirt/dev/neuro_ai/EEGClip/results/"
tuh_data_dir = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/"
embs_df_path = '/home/ndirt/dev/neuro_ai/EEGClip/scripts/text_preprocessing/embs_df.csv'


# path to models trained on various tasks. Handy for baselines comparisons
model_paths = {"eegclip128":"/home/jovyan/EEGClip/results/wandb/EEGClip/1lgwz214/checkpoints/epoch=6-step=42609.ckpt",
              "eegclip":"/home/jovyan/EEGClip/results/models/EEGClip_100_medicalai/ClinicalBERT_64.ckpt",
              "pathological_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_few_shot/1vljui8s/checkpoints/epoch=9-step=7100.ckpt",
              "under_50_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_few_shot/akl12j6m/checkpoints/epoch=9-step=7100.ckpt"
            }  