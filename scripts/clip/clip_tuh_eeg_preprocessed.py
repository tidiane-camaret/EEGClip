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

n_jobs = 4
data_path = '/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal/v2.0.0/edf/'
recording_ids=range(150)
N_JOBS = 8 


sfreq  = 100
n_minutes = 20


tuabn = TUHAbnormal(
        path=data_path,
        preload=False,  # True
        add_physician_reports=True, 
        n_jobs=n_jobs,
        target_name = 'subject',
        recording_ids=recording_ids,
    )




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
    n_jobs=N_JOBS,

)
window_valid_set = create_fixed_length_windows(
    valid_set,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=N_JOBS,

)
"""

tuh_windows = create_fixed_length_windows(
    tuabn,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=N_JOBS,

)

print("length of windowed dataset : ", len(tuh_windows))
window_train_set, window_valid_set = torch.utils.data.random_split(tuh_windows,[0.8, 0.2]) #splitted['True'], splitted['False'] 

"""
batch_size = 32
num_workers = 32
n_epochs = 100

train_loader = torch.utils.data.DataLoader(
    window_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

valid_loader = torch.utils.data.DataLoader(
    window_valid_set,
    batch_size=batch_size,
    shuffle=False,
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
)


# These values we found good for shallow network:
#lr = 0.0625 * 0.01
#weight_decay = 0

# For deep4 they should be:
lr = 1 * 0.01
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
                         recordings_df = pd.read_csv('/home/jovyan/EEGClip/data/TUH_Abnormal_EEG_rep.csv')
                         ), 
                         dataloaders=valid_loader
                )

"""
trainer.fit(
                EEGClipModule(
                         eeg_classifier_model=eeg_classifier_model,
                         lr = lr, 
                         recordings_df = pd.read_csv('/home/jovyan/EEGClip/data/TUH_Abnormal_EEG_rep.csv')
                         ),
                train_loader, 
                valid_loader
            )


