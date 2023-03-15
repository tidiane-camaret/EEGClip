import pandas as pd
import numpy as np
import torch 

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets import TUHAbnormal
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet, deep4
from braindecode.preprocessing import create_fixed_length_windows

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

from EEGClip.clip_models import EEGClipModule

import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

data_path = '/home/jovyan/mne_data/TUH/tuh_eeg_abnormal/v2.0.0/edf/'
#data_path = '/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/'

n_jobs = 4
n_epochs = 100
batch_size = 32
num_workers = 0


tuabn = TUHAbnormal(
        path=data_path,
        preload=False,  # True
        add_physician_reports=True, 
        n_jobs=n_jobs,
        target_name = 'report',
        recording_ids=range(300),
    )

ar_ch_names = sorted([
    'EEG A1-REF', 'EEG A2-REF',
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
from braindecode.preprocessing import preprocess, Preprocessor

n_max_minutes = 3
sfreq = 100

preprocessors = [
    Preprocessor(fn='pick_channels', ch_names=ar_ch_names, ordered=True),
    Preprocessor('crop', tmin=0, tmax=n_max_minutes*60, include_tmax=True),
    Preprocessor(fn=lambda x: np.clip(x, -800,800), apply_on_array=True),
    Preprocessor('set_eeg_reference', ref_channels='average'),
    # convert from volt to microvolt, directly modifying the numpy array
    Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
    Preprocessor(fn=lambda x: x / 30, apply_on_array=True), # this seemed best
    Preprocessor(fn='resample', sfreq=sfreq),
]
# Preprocess the data
preprocess(tuabn, preprocessors)


subject_datasets = tuabn.split('subject')
n_subjects = len(subject_datasets)

n_split = int(np.round(n_subjects * 0.75))
keys = list(subject_datasets.keys())
train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
train_set = BaseConcatDataset(train_sets)
valid_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[keys[i]].datasets]
valid_set = BaseConcatDataset(valid_sets)




#input_window_samples = 2000
#window_stride_samples = 1000
#n_preds_per_input = 1000


window_size_samples = 1000
window_stride_samples = 1000

window_train_set = create_fixed_length_windows(
    train_set,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=True,
    n_jobs=n_jobs,

)
window_valid_set = create_fixed_length_windows(
    valid_set,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=n_jobs,

)
"""

tuh_windows = create_fixed_length_windows(
    tuabn,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=n_jobs,

)

print("length of windowed dataset : ", len(tuh_windows))
window_train_set, window_valid_set = torch.utils.data.random_split(tuh_windows,[0.8, 0.2]) #splitted['True'], splitted['False'] 
"""



train_loader = torch.utils.data.DataLoader(
    window_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

valid_loader = torch.utils.data.DataLoader(
    window_valid_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False)



n_classes = 128
# Extract number of chans and time steps from dataset
n_chans = window_train_set[0][0].shape[0]
input_window_samples = window_train_set[0][0].shape[1]

eeg_classifier_model = deep4.Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
    stride_before_pool=True
)


# These values we found good for shallow network:
#lr = 0.0625 * 0.01
#weight_decay = 0

# For deep4 they should be:
lr = 1 * 0.001
weight_decay = 0.5 * 0.001

wandb_logger = WandbLogger(project="EEGClip",save_dir = "results/wandb")
#logger = TensorBoardLogger("results/tb_logs", name="EEG_Clip")

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=n_epochs,
    #callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=wandb_logger,
    #profiler="advanced"
)


"""
trainer.validate(                
                EEGClipModule(
                         eeg_classifier_model=eeg_classifier_model,
                         lr = lr, 
                         ), 
                         dataloaders=valid_loader
                )

"""
trainer.fit(
                EEGClipModule(
                         eeg_classifier_model=eeg_classifier_model,
                         lr = lr, 
                         ),
                train_loader, 
                valid_loader
            )

