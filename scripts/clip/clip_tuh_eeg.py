
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from braindecode.datasets import TUHAbnormal
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
N_SAMPLES = 5

tuh = TUHAbnormal(
    path=TUHAbnormal_PATH,
    recording_ids=list(range(N_SAMPLES)),
    target_name=('pathological'),#'report'),
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

print(len(tuh_windows))

# split the dataset in train and test (label is included in the data)
splitted = tuh_windows.split("train")
train_set = splitted['True']
valid_set = splitted['False']


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 2
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

classifier_model = ShallowFBCSPNet(
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

batch_size = 64
n_epochs = 50


train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size)

logger = TensorBoardLogger("results/tb_logs", name="EEG_Classifier")

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  
    max_epochs=n_epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=logger,
)

trainer.fit(EEGClipModule(classifier_model=classifier_model, lr = lr), train_loader, valid_loader)
