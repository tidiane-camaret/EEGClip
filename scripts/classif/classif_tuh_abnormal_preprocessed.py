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

from EEGClip.classifier_models import EEGClassifierModule

import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

data_path = '/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/'

n_jobs = 4
n_epochs = 100
batch_size = 64
num_workers = 16


tuabn = TUHAbnormal(
        path=data_path,
        preload=False,  # True
        add_physician_reports=True, 
        n_jobs=n_jobs,
        target_name = ('pathological'),#'report'),
        #recording_ids=range(150),
    )


subject_datasets = tuabn.split('subject')
n_subjects = len(subject_datasets)

n_split = int(np.round(n_subjects * 0.75))
keys = list(subject_datasets.keys())
train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
train_set = BaseConcatDataset(train_sets)
valid_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[keys[i]].datasets]
valid_set = BaseConcatDataset(valid_sets)




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


lr = 1 * 0.001
weight_decay = 0.5 * 0.001

n_classes = 2
# Extract number of chans and time steps from dataset
n_chans = window_train_set[0][0].shape[0]
input_window_samples = window_train_set[0][0].shape[1]

eeg_classifier_model = deep4.Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

## USING BRAINDECODE EEGCLASSIFIER METHOD
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    eeg_classifier_model.cuda()

clf = EEGClassifier(
    eeg_classifier_model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(window_valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(window_train_set, y=None, epochs=n_epochs)
"""
## USING CUSTOM CLASSIFIER


wandb_logger = WandbLogger(project="EEGClip_classif",save_dir = "results/wandb")
#logger = TensorBoardLogger("results/tb_logs", name="EEG_Clip")

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=n_epochs,
    #callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=wandb_logger,
    #profiler="advanced"
)


trainer.fit(
    EEGClassifierModule(
        eeg_classifier_model=eeg_classifier_model, 
            lr = lr), 
        train_loader, 
        valid_loader)

"""