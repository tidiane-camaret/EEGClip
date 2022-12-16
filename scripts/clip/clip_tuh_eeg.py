
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from braindecode.datasets import TUHAbnormal, BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, create_windows_from_events, scale as multiply)
import torch
from braindecode.util import set_random_seeds

from braindecode.models import ShallowFBCSPNet, deep4
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from braindecode import EEGClassifier

from EEGClip.clip_models import EEGClipModule

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

TUHAbnormal_PATH = '/home/jovyan/mne_data/TUH/tuh_eeg_abnormal/v2.0.0'
N_JOBS = 8  # specify the number of jobs for loading and windowing
N_SAMPLES = 100

tuh = TUHAbnormal(
    path=TUHAbnormal_PATH,
    recording_ids=list(range(N_SAMPLES)),
    target_name=('report'),#'pathological'),
    preload=False,
    add_physician_reports=True,
    n_jobs=N_JOBS, 
)

print("length of dataset : ", len(tuh))


print(tuh.description)

# create windows

window_size_samples = 1000
window_stride_samples = 1000
tuh_windows = create_fixed_length_windows(
    tuh,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=N_JOBS,

)

print("length of windowed dataset : ", len(tuh_windows))

# split the dataset in train and test 

subject_datasets = tuh_windows.split('subject')
n_subjects = len(subject_datasets)

n_split = int(np.round(n_subjects * 0.75))
keys = list(subject_datasets.keys())
train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
train_set = BaseConcatDataset(train_sets)
valid_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[keys[i]].datasets]
valid_set = BaseConcatDataset(valid_sets)


n_classes = 128
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

eeg_classifier_model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Send model to GPU


# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 32#64
n_epochs = 50
num_workers = 32

train_loader = torch.utils.data.DataLoader(train_set, 
                                          batch_size = batch_size, 
                                          num_workers = num_workers,
                                          shuffle=True,
                                          drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_set, 
                                           batch_size = batch_size, 
                                           num_workers = num_workers,
                                           shuffle=False,
                                           drop_last=False)

logger = TensorBoardLogger("results/tb_logs", name="EEG_Clip")

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    #devices=None, #1 if torch.cuda.is_available() else None,  
    max_epochs=n_epochs,
    #callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=logger,
    profiler="advanced"
)

trainer.fit(EEGClipModule(eeg_classifier_model=eeg_classifier_model, lr = lr), train_loader, valid_loader)
