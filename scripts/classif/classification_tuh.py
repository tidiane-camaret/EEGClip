import argparse
import random
import socket

import mne
import numpy as np
import torch

mne.set_log_level("ERROR")  # avoid messages everytime a window is extracted

from braindecode.datasets import TUHAbnormal
from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
)
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import configs.EEGClip_config as EEGClip_config
from EEGClip.classifier_models import EEGClassifierModel
from EEGClip.clip_models import EEGClipModel

"""
This script is used to train a classifier model
on the TUH EEG dataset for different classification tasks :
pathological
age
gender
report-based (medication, diagnosis ...)

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EEG classifier on the TUH EEG dataset."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="pathological",
        help="classification task name (pathological, age, gender, report-related tasks ....",
    )
    parser.add_argument(
        "--n_rec",
        type=int,
        default=2993,
        help="Number of recordings to load from TUH EEG dataset.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to train EEGClip model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size to train EEGClip model."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate to train EEGClip model."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay to train EEGClip model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for data loading.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="eegclip",
        help="weights from pretrained model, or random",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Whether to freeze encoder during training",
    )
    parser.add_argument("--seed", type=int, default=20210325, help="random seed")
    parser.add_argument(
        "--crossval", action="store_true", help="Whether to do crossvalidation"
    )
    parser.add_argument(
        "--folds_nb", type=int, default=5, help="nb of folds for cross-validation"
    )
    parser.add_argument(
        "--fold_idx", type=int, default=0, help="(0 to folds_nb - 1) valid fold index"
    )
    parser.add_argument(
        "--train_frac",
        type=int,
        default=1,
        help="factor of division for the training set (few shot learning)",
    )
    parser.add_argument(
        "--exclude_eegclip_train_set",
        action="store_true",
        help="excludes the first 70% subjects for training and testing",
    )

    args = parser.parse_args()

    num_workers = args.num_workers
    train_frac = args.train_frac

    n_recordings_to_load = args.n_rec
    task_name = (
        args.task_name
    )  # ('pathological', 'age', 'gender', "epilep", "keppra", "dilantin", "seizure")
    if task_name in ["pathological", "age", "gender"]:
        target_name = task_name
    elif task_name == "under_50":
        target_name = "age"
    else:
        target_name = "report"
    mapping = {"M": 0, "F": 1} if target_name == "gender" else None
    # TODO : find a way to use several targets
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200

    exclude_eegclip_train_set = args.exclude_eegclip_train_set
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    nailcluster = socket.gethostname() == "vs3-0"
    weights = args.weights
    freeze_encoder = args.freeze_encoder
    num_workers = args.num_workers

    results_dir = EEGClip_config.results_dir
    tuh_data_dir = EEGClip_config.tuh_data_dir

    # TODO : use get_output_shape (requires to load the model first)
    n_preds_per_input = (
        519  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
    )

    seed = args.seed  # random seed to make results reproducible

    cuda = torch.cuda.is_available()
    set_random_seeds(seed=seed, cuda=cuda)
    torch.backends.cudnn.benchmark = True

    print(target_name)
    print("freeze encoder : ", freeze_encoder)

    # ## Load data
    dataset = TUHAbnormal(
        path=tuh_data_dir,
        recording_ids=range(
            n_recordings_to_load
        ),  # loads the n chronologically first recordings
        target_name=target_name,  # age, gender, pathology
        preload=False,
        add_physician_reports=True,
        # n_jobs=1
    )

    # Preprocess the data
    if not nailcluster:
        preprocess(dataset, EEGClip_config.preprocessors)

    # ## Data Splitting
    # TODO : split using train and test splits instead
    # TODO : maybe load TUH now on top of TUH Abnormal ?

    if args.crossval:
        subject_datasets = dataset.split("subject")
        n_subjects = len(subject_datasets)
        keys = list(subject_datasets.keys())

        folds = np.array_split(keys, args.folds_nb)

        train_keys = np.concatenate(
            [folds[i] for i in range(args.folds_nb) if i != args.fold_idx]
        )
        valid_keys = folds[args.fold_idx]

        if args.exclude_eegclip_train_set:
            train_keys, valid_keys = (
                valid_keys[: len(valid_keys) // 2],
                valid_keys[len(valid_keys) // 2 :],
            )

        print(len(train_keys), len(valid_keys))

        # subsample training set
        train_keys = random.sample(list(train_keys), len(train_keys) // train_frac)

        train_sets = [d for k in train_keys for d in subject_datasets[k].datasets]
        train_set = BaseConcatDataset(train_sets)

        valid_sets = [d for k in valid_keys for d in subject_datasets[k].datasets]
        valid_set = BaseConcatDataset(valid_sets)

    else:
        train_set = dataset.split("train")["True"]
        """
        subject_datasets = train_set.split('subject')
        n_subjects = len(subject_datasets)

        n_split = int(np.round(n_subjects * 0.75))
        train_keys = list(subject_datasets.keys())
        
        # first 75% of training set
        #train_sets = [d for i in range(n_split) for d in subject_datasets[train_keys[i]].datasets]
        
        # last 25% of training set
        train_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[train_keys[i]].datasets]
        
        
        train_sets = random.sample(list(train_sets), len(train_sets) // train_frac) 
        print('nb train subjects : ', len(train_sets))
        train_set = BaseConcatDataset(train_sets)
        """

        valid_set = dataset.split("train")["False"]

    window_train_set = create_fixed_length_windows(
        train_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=True,
        mapping=mapping,
    )

    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        mapping=mapping,
    )

    ### PREPROCESSING NECESSARY IF USING TUH_PRE
    if nailcluster:
        window_train_set.transform = lambda x: x * 1e6
        window_valid_set.transform = lambda x: x * 1e6

    train_loader = torch.utils.data.DataLoader(
        window_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    print("size train set : ", len(train_loader.dataset))
    print("size valid set : ", len(valid_loader.dataset))

    encoder_output_dim = 64  # size of the last layer of the EEG decoder
    n_chans = 21  # number of channels in the EEG data

    # ## Create model
    if weights == "eegclip":
        eegclipmodel = EEGClipModel.load_from_checkpoint(
            results_dir + "/models/EEGClip_75.ckpt"
        )
        EEGEncoder = torch.nn.Sequential(
            eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
        )

        """# classifier should have the same shape everywhere for fair comparison
            projectionhead = list(EEGEncoder.children())[-1]
            layer_sizes = []
            for layer in projectionhead.children():
                if hasattr(layer, 'out_features'):
                    layer_sizes.append(layer.out_features)
            encoder_output_dim = layer_sizes[-1]
            """
    elif weights == "random":
        """
        EEGEncoder = Deep4Net(
            in_chans=n_chans,
            n_classes=encoder_output_dim,
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
            )

        to_dense_prediction_model(EEGEncoder)

        """
        eegclipmodel = EEGClipModel.load_from_checkpoint(
            results_dir + "/models/EEGClip_100_medicalai/ClinicalBERT_64.ckpt"
        )

        EEGEncoder = torch.nn.Sequential(
            eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
        )
        for layer in EEGEncoder.modules():
            if hasattr(layer, "reset_parameters"):
                # print(layer)
                layer.reset_parameters()

    else:
        eegclipmodel = EEGClipModel.load_from_checkpoint(
            results_dir + "/models/EEGClip_75.ckpt"
        )
        # print(eegclipmodel)
        EEGEncoder = torch.nn.Sequential(
            eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
        )
        for layer in EEGEncoder.modules():
            if hasattr(layer, "reset_parameters"):
                # print(layer)
                layer.reset_parameters()
        eegclassifiermodel = EEGClassifierModel.load_from_checkpoint(
            results_dir + "/models/" + weights + "_75.ckpt",
            EEGEncoder=EEGEncoder,
            encoder_output_dim=64,
        )

        EEGEncoder = eegclassifiermodel.encoder

    print("encoder_output_dim", encoder_output_dim)
    # ## Run Training
    wandb_logger = WandbLogger(
        project="EEGClip_few_shot_2",
        save_dir=results_dir + "/wandb",
        log_model=False,
    )

    wandb_logger.experiment.config.update(
        {
            "freeze_encoder": freeze_encoder,
            "weights": weights,
            "task_name": task_name,
            "train_frac": train_frac,
        },
        # allow_val_change=True
    )
    trainer = Trainer(
        default_root_dir=results_dir + "/models",
        devices=1,
        accelerator="gpu",
        max_epochs=n_epochs,
        logger=wandb_logger,
        # checkpoint_callback=False # do not save model
    )
    trainer.fit(
        EEGClassifierModel(
            EEGEncoder,
            task_name=task_name,
            freeze_encoder=freeze_encoder,
            lr=lr,
            weight_decay=weight_decay,
            encoder_output_dim=encoder_output_dim,
        ),
        train_loader,
        valid_loader,
    )

    # trainer.save_checkpoint(results_dir + "/models/pathological_75.ckpt")
