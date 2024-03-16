import os
import socket

import numpy as np
from braindecode.preprocessing import Preprocessor

nailcluster = socket.gethostname() == "vs3-0"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

"""
paths to the different datasets/models. Feel free to modify
"""

# if using the nail cluster :
if nailcluster:
    results_dir = "/home/jovyan/EEGClip/results/"
    tuh_data_dir = "/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/"
    embs_df_path = "/home/jovyan/EEGClip/scripts/text_preprocessing/embs_df.csv"

# if using KISlurm :
else:
    results_dir = "/home/ndirt/dev/neuro_ai/EEGClip/results/"
    tuh_data_dir = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/"
    embs_df_path = (
        "/home/ndirt/dev/neuro_ai/EEGClip/scripts/text_preprocessing/embs_df.csv"
    )
    zc_sentences_emb_dict_path = "/home/ndirt/dev/neuro_ai/EEGClip/scripts/text_embedding/zc_sentences_emb_dict.json"


# path to models trained on various tasks. Handy for baselines comparisons
model_paths = {
    "eegclip128": results_dir
    + "wandb/EEGClip/1lgwz214/checkpoints/epoch=6-step=42609.ckpt",
    # "eegclip":results_dir + "models/EEGClip_100_medicalai/ClinicalBERT_64.ckpt",
    "eegclip_bert": results_dir + "wandb/EEGClip/kg9zhzgx/checkpoints/epoch=11-step=10692.ckpt",
    "eegclip_instructor": results_dir + "wandb/EEGClip/xv65fc7j/checkpoints/epoch=15-step=14256.ckpt",
    "eegclip": results_dir
    + "wandb/EEGClip/v90rgytb/checkpoints/epoch=19-step=17820.ckpt",
    "pathological_task": results_dir
    + "wandb/EEGClip_few_shot/1vljui8s/checkpoints/epoch=9-step=7100.ckpt",
    "under_50_task": results_dir
    + "wandb/EEGClip_few_shot/akl12j6m/checkpoints/epoch=9-step=7100.ckpt",
}


n_max_minutes = 3
sfreq = 100
ar_ch_names = sorted(
    [
        "EEG A1-REF",
        "EEG A2-REF",
        "EEG FP1-REF",
        "EEG FP2-REF",
        "EEG F3-REF",
        "EEG F4-REF",
        "EEG C3-REF",
        "EEG C4-REF",
        "EEG P3-REF",
        "EEG P4-REF",
        "EEG O1-REF",
        "EEG O2-REF",
        "EEG F7-REF",
        "EEG F8-REF",
        "EEG T3-REF",
        "EEG T4-REF",
        "EEG T5-REF",
        "EEG T6-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]
)

preprocessors = [
    # DONE : correct the order of preprocessing steps ?
    Preprocessor(fn="pick_channels", ch_names=ar_ch_names, ordered=True),
    Preprocessor("crop", tmin=0, tmax=n_max_minutes * 60, include_tmax=True),
    Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
    Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True),
    # convert from volt to microvolt, directly modifying the numpy array
    Preprocessor("set_eeg_reference", ref_channels="average"),
    Preprocessor(fn=lambda x: x / 30, apply_on_array=True),  # this seemed best
    Preprocessor(fn="resample", sfreq=sfreq),
]
