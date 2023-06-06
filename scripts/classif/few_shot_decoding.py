import argparse
import socket
import random
import pandas as pd
import numpy as np
import torch 

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.models import Deep4Net

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from EEGClip.classifier_models import EEGClassifierModel
from EEGClip.clip_models import EEGClipModel

import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

import os

"""
This script is used to train a classifier model
on the TUH EEG dataset for different classification tasks :
pathological
age
gender
report-based (medication, diagnosis ...)

"""
#MODEL_PATH = '/home/jovyan/EEGClip/results/wandb/EEGClip/df7e5wqd/checkpoints/epoch=7-step=48696.ckpt'
MODEL_PATH = {"eegclip":"/home/jovyan/EEGClip/results/wandb/EEGClip/1lgwz214/checkpoints/epoch=6-step=42609.ckpt",
              "pathological_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_classif/1oqtbdtr/checkpoints/epoch=9-step=7600.ckpt",
              "under_50_task" : "/home/jovyan/EEGClip/results/wandb/EEGClip_classif/3d1yl4md/checkpoints/epoch=9-step=7600.ckpt"
            }
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an EEG classifier on the TUH EEG dataset.')
    parser.add_argument('--task_name', type=str, default="pathological",
                        help='classification task name (pathological, age, gender, report-related tasks ....')    
    parser.add_argument('--n_rec', type=int, default=2993,
                        help='Number of recordings to load from TUH EEG dataset.')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train EEGClip model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to train EEGClip model.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate to train EEGClip model.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay to train EEGClip model.')
    #parser.add_argument('--nailcluster', action='store_true',
    #                    help='Whether to run on the Nail cluster(paths differ)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers to use for data loading.')
    parser.add_argument('--weights', type=str, default="eegclip",
                        help='weights from pretrained model, or random')    
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Whether to freeze encoder during training')
    parser.add_argument('--train_frac', type=int, default=1,
                        help='factor of division for the training set (few shot learning)')

    args = parser.parse_args()

    num_workers = args.num_workers
    train_frac = args.train_frac

    n_recordings_to_load = args.n_rec
    task_name = args.task_name #('pathological', 'age', 'gender', "epilep", "keppra", "dilantin", "seizure")
    if task_name in ['pathological', 'age', 'gender']: 
        target_name = task_name 
    elif task_name == "under_50":
        target_name = "age"
    else:
        target_name = "report"
    mapping = {'M': 0, 'F': 1} if target_name =="gender" else None
    # TODO : find a way to use several targets
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    nailcluster = (socket.gethostname() == "vs3-0")
    weights = args.weights
    freeze_encoder = args.freeze_encoder
    num_workers = args.num_workers
    

    if nailcluster:
        results_dir = "/home/jovyan/EEGClip/results/"
        tuh_data_dir = "/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/"
    else:
        results_dir = "/home/ndirt/dev/neuro_ai/EEGClip/results/"
        tuh_data_dir = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/edf/"

  # TODO : use get_output_shape (requires to load the model first)
    n_preds_per_input = 519 #get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]

  
    seed = 20210325  # random seed to make results reproducible

    cuda = torch.cuda.is_available()
    set_random_seeds(seed=seed, cuda=cuda)
    torch.backends.cudnn.benchmark = True

    # ## Load data
    dataset = TUHAbnormal(
        path=tuh_data_dir,
        recording_ids=range(n_recordings_to_load),  # loads the n chronologically first recordings
        target_name=target_name,  # age, gender, pathology
        preload=False,
        add_physician_reports=True,
        n_jobs=1)
    
    # ## Preprocessing

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
    if not nailcluster:
        preprocess(dataset, preprocessors)

   # ## Data Splitting
    # TODO : split using train and test splits instead
    # TODO : maybe load TUH now on top of TUH Abnormal ?

    
    #dataset = dataset.split('train')['True']

    subject_datasets = dataset.split('subject')
    n_subjects = len(subject_datasets)

    print("Nb subjects loaded : ", n_subjects)
    
    
    n_split = int(np.round(n_subjects * 0.75))
    keys = list(subject_datasets.keys())
    train_keys = keys[:n_split]
    train_keys = random.sample(train_keys, len(train_keys) // train_frac)
    train_sets = [d for k in train_keys for d in subject_datasets[k].datasets]
    print("Final nb train subjects loaded : ",len(train_sets))
    train_set = BaseConcatDataset(train_sets)

    valid_keys = keys[n_split:]
    valid_sets = [d for k in valid_keys for d in subject_datasets[k].datasets]
    print("Nb valid subjects loaded : ",len(valid_sets))
    valid_set = BaseConcatDataset(valid_sets)
    """
    train_set = dataset.split('train')['True']
    train_subject_datasets = train_set.split('subject')
    print("Nb train subjects loaded : ",len(train_subject_datasets))
    # Subsampling for few-shot learning TODO : balanced subsampling
    keys = list(train_subject_datasets.keys())
    keys = random.sample(keys, len(keys) // train_frac)
    train_sets = [d for k in keys for d in train_subject_datasets[k].datasets]
    print("Final nb train subjects loaded : ",len(train_sets))
    train_set = BaseConcatDataset(train_sets)

    valid_set = dataset.split('train')['False']
    valid_subject_datasets = valid_set.split('subject')
    print("Nb valid subjects loaded : ",len(valid_subject_datasets))
    """
    window_train_set = create_fixed_length_windows(
        train_set,
        start_offset_samples=60*sfreq,
        stop_offset_samples=60*sfreq+n_minutes*60*sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=True,
        mapping=mapping
    )

    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60*sfreq,
        stop_offset_samples=60*sfreq+n_minutes*60*sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        mapping=mapping
    )

    ### PREPROCESSING NECESSARY IF USING TUH_PRE
    if nailcluster:
        window_train_set.transform = lambda x: x*1e6
        window_valid_set.transform = lambda x: x*1e6

    train_loader = torch.utils.data.DataLoader(
        window_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)
    train_det_loader = torch.utils.data.DataLoader(
        window_train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    print(len(valid_loader.dataset))
    

    encoder_output_dim = 64 # size of the last layer of the EEG decoder
    n_chans = 21 # number of channels in the EEG data

    # ## Create model
    if weights == "eegclip":
            eegclipmodel = EEGClipModel.load_from_checkpoint(MODEL_PATH["eegclip"])
            EEGEncoder = torch.nn.Sequential(eegclipmodel.eeg_encoder,eegclipmodel.eeg_projection)
            
            """# classifier should have the same shape everywhere for fair comparison
            projectionhead = list(EEGEncoder.children())[-1]
            layer_sizes = []
            for layer in projectionhead.children():
                if hasattr(layer, 'out_features'):
                    layer_sizes.append(layer.out_features)
            encoder_output_dim = layer_sizes[-1]
            """
    elif weights == "random":
        EEGEncoder = Deep4Net(
            in_chans=n_chans,
            n_classes=encoder_output_dim, 
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
            )

        to_dense_prediction_model(EEGEncoder)

    else:
        EEGEncoder = Deep4Net(
            in_chans=n_chans,
            n_classes=encoder_output_dim, 
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
            )

        to_dense_prediction_model(EEGEncoder)
        eegclassifiermodel = EEGClassifierModel.load_from_checkpoint(MODEL_PATH[weights],EEGEncoder=EEGEncoder)

        EEGEncoder = eegclassifiermodel.eeg_encoder
        # get size of the last layer
        projectionhead = list(EEGEncoder.children())[-1]
        layer_sizes = []
        for layer in projectionhead.children():
            if hasattr(layer, 'out_features'):
                layer_sizes.append(layer.out_features)
        encoder_output_dim = layer_sizes[-1]

    print('encoder_output_dim', encoder_output_dim)
            # ## Run Training
    wandb_logger = WandbLogger(project="EEGClip_classif",
                        save_dir = results_dir + '/wandb',
                        log_model=True,
                        )

    wandb_logger.experiment.config.update({"freeze_encoder": freeze_encoder,
                                            "weights": weights,
                                            "task_name": task_name,
                                            "target_name": target_name},
                                            #allow_val_change=True
                                            )
    trainer = Trainer(
                default_root_dir=results_dir + '/models',
                devices=1,
                accelerator="gpu",
                max_epochs=n_epochs,
                logger=wandb_logger,
                #checkpoint_callback=False # do not save model
            )
    trainer.fit(
        EEGClassifierModel(
                  EEGEncoder, 
                  task_name = task_name,
                  freeze_encoder=freeze_encoder,
                  lr = lr,
                  weight_decay= weight_decay,
                  encoder_output_dim = encoder_output_dim,
        ),
        train_loader,
        valid_loader,
    )
