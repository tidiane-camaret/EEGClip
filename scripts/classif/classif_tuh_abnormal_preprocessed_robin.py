# %% [markdown]
# ## notebook setup, not necessary for code to run

# %%

# %% [markdown]
# ## Hyperparameters

# %%
n_recordings_to_load = 300
target_name = 'pathological'
n_max_minutes = 3
sfreq = 100
n_minutes = 2
input_window_samples = 1200
n_epochs = 50
batch_size = 64
# This was from High-Gamma dataset optimization:
#lr = 1 * 0.01
#weight_decay = 0.5 * 0.001
lr = 5e-3
weight_decay = 5e-4

seed = 20210325  # random seed to make results reproducible

# atm window stride determined automatically as n_preds_per_input, could also parametrize it 

# %% [markdown]
# ## Set random seeds for reproducibility

# %%
from braindecode.util import set_random_seeds
cuda = True
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed, cuda=cuda)

# %% [markdown]
# ## Model definition

# %%
import torch as th

th.backends.cudnn.benchmark = True

# %%
import torch
from braindecode.models import Deep4Net


n_classes = 2
# Extract number of chans from dataset
n_chans = 21

model = Deep4Net(
    in_chans=n_chans,
    n_classes=n_classes, 
    input_window_samples=None,
    final_conv_length=2,
    stride_before_pool=True,
)

# Send model to GPU
if cuda:
    model.cuda()
from braindecode.models.util import to_dense_prediction_model, get_output_shape
to_dense_prediction_model(model)


# %% [markdown]
# ## Data Loading

# %%

from braindecode.datasets.tuh import TUHAbnormal
data_path = '/home/jovyan/mne_data/TUH/tuh_eeg_abnormal/v2.0.0/edf/'
dataset = TUHAbnormal(
    path=data_path,
    recording_ids=range(n_recordings_to_load),  # loads the n chronologically first recordings
    target_name=target_name,  # age, gender, pathology
    preload=False,
    add_physician_reports=False,
)

# %%
from braindecode.datasets import BaseConcatDataset
#dataset = BaseConcatDataset(dataset.datasets[:n_recordings_to_load])

# %% [markdown]
# ## Data Preprocessing

# %%

from braindecode.preprocessing import preprocess, Preprocessor
import numpy as np
from copy import deepcopy


whole_train_set = dataset.split('train')['True']

ar_ch_names = sorted([
    'EEG A1-REF', 'EEG A2-REF',
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])

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
preprocess(whole_train_set, preprocessors)

# %% [markdown]
# ## Data Splitting

# %%
import numpy as np

from braindecode.datasets.base import BaseConcatDataset


subject_datasets = whole_train_set.split('subject')
n_subjects = len(subject_datasets)

n_split = int(np.round(n_subjects * 0.75))
keys = list(subject_datasets.keys())
train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
train_set = BaseConcatDataset(train_sets)
valid_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[keys[i]].datasets]
valid_set = BaseConcatDataset(valid_sets)

# %% [markdown]
# ## Data Compute Window Creation

# %%
import pandas as pd

from braindecode.models.util import to_dense_prediction_model, get_output_shape

n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

# %%
from braindecode.datautil.windowers import create_fixed_length_windows


window_train_set = create_fixed_length_windows(
    train_set,
    start_offset_samples=60*sfreq,
    stop_offset_samples=60*sfreq+n_minutes*60*sfreq,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=True,
)

window_valid_set = create_fixed_length_windows(
    valid_set,
    start_offset_samples=60*sfreq,
    stop_offset_samples=60*sfreq+n_minutes*60*sfreq,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
)

# %% [markdown]
# ## Initialize Data Loaders

# %%
num_workers = 0

# %%
train_loader = th.utils.data.DataLoader(
    window_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)
train_det_loader = th.utils.data.DataLoader(
    window_train_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)
valid_loader = th.utils.data.DataLoader(
    window_valid_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)

# %% [markdown]
# ## Initialize Optimizer and Scheduler

# %%
optim = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs,)

# %% [markdown]
# ## Run Training

# %%
from tqdm.autonotebook import tqdm, trange
from braindecode.training.scoring import trial_preds_from_window_preds
import pandas as pd


epochs_df = pd.DataFrame()

for i_epoch in trange(n_epochs):
    if i_epoch > 0:
        for X,y,i in tqdm(train_loader):
            #print(X.shape)
            #print(i)
            X = X.cuda()
            y = y.type(th.LongTensor).cuda()
            out = model(X)
            #print(out.shape)
            cross_ent = th.nn.functional.cross_entropy(th.mean(out, dim=(2)), y)
            optim.zero_grad()
            cross_ent.backward()
            optim.step()
            optim.zero_grad()
        scheduler.step()

    epoch_results = {}
    epoch_results['epoch'] = i_epoch
    for name, loader in {'train': train_det_loader, 'valid': valid_loader}.items():
        print(f"Epoch {i_epoch:d}")
        all_preds = []
        all_is = []
        all_ys = []
        with th.no_grad():
            for X,y,i in tqdm(loader):
                preds = model(X.cuda())
                all_preds.extend(preds.cpu().numpy())
                all_is.extend(i)
                all_ys.extend(y.cpu().numpy())
        all_preds = np.array(all_preds)
        #print(all_preds.shape) # (nb_crops, 2, nb_pred_per_crop)
        all_ys = np.array(all_ys)
        crop_preds = np.mean(all_preds, axis=(2)).argmax(axis=1)
        acc_cropwise = np.mean((crop_preds == all_ys))
        trial_ys = all_ys[np.diff(th.cat(all_is[0::3]), prepend=[np.inf]) != 1]
        #print(all_preds.shape, th.cat(all_is[0::3]).shape)
        preds_per_trial = trial_preds_from_window_preds(all_preds, th.cat(all_is[0::3]), 
                                                        th.cat(all_is[2::3]),)
        trial_preds = np.array([p.mean(axis=1).argmax(axis=0) for p in preds_per_trial])
        acc_trial = np.mean(trial_preds  == trial_ys)
        print(f"{name.capitalize()} Crop Acc:                {acc_cropwise:.1%}")
        print(f"{name.capitalize()} Recording Acc:           {acc_trial:.1%}")

        results = dict(acc_cropwise=acc_cropwise,
                       acc_trial=acc_trial,)
        #add 'train', 'valid' prefix
        results = {name + '_' + key: value for key, value in results.items()}
        epoch_results = {**epoch_results, **results}
    epochs_df = epochs_df.append(
        epoch_results,
        ignore_index=True)

# %%
epochs_df

# %%



