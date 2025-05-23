import argparse
import socket

import mne
import numpy as np
import torch
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

import configs.preprocess_config as preprocess_config
from EEGClip.clip_models import EEGClipModel
from EEGClip.text_preprocessing import text_preprocessing

mne.set_log_level("ERROR")  # avoid messages everytime a window is extracted

import os

"""
This script is used to train the EEGClip model on the TUH EEG dataset.
"""


def run_training(
    lr: float,  # learning rate to train EEGClip model
    lr_frac_lm: float,  # learning rate frac for the text encoder
    text_encoder_name: str,  # name of text encoder
    weight_decay: float,  # weight decay to train EEGClip model
    string_sampling: bool,  # whether to use string sampling
    projected_emb_dim: int,  # dimension of projected embeddings
    num_fc_layers: int,  # number of fully connected layers
    target_name: str = "report",  # target to train EEGClip model on
    category: str = "all",
    text_encoder_trainable: bool = True,
    text_model_emb_dim: int = 768,
    n_recordings_to_load: int = 2993,  # number of recordings to load from TUH EEG dataset
    n_epochs: int = 8,  # number of epochs to train EEGClip model
    num_workers: int = 16,  # number of workers to use for data loading
    batch_size: int = 64,  # batch size to train EEGClip model
    crossval: bool = False,
    folds_nb: int = 5,
    fold_idx: int = 0,  # (0 to 4) fold idx of the validation set'
):
    print("category : ", category)
    nailcluster = (
        socket.gethostname() == "vs3-0"
    )  # check if we are on the nail cluster or on kislurm

    results_dir = preprocess_config.results_dir
    tuh_data_dir = preprocess_config.tuh_data_dir

    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    # TODO : use get_output_shape (requires to load the model first)
    n_preds_per_input = (
        519  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
    )

    cuda = torch.cuda.is_available()
    seed = 20210325  # random seed to make results reproducible
    set_random_seeds(seed=seed, cuda=cuda)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(num_workers)  # Sets the available number of threads
    # apparently this is needed to avoid a deadlock in the DataLoader
    # TODO : check if this is still needed
    # https://github.com/huggingface/transformers/issues/5486
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ## Load data
    dataset = TUHAbnormal(
        path=tuh_data_dir,
        recording_ids=range(
            n_recordings_to_load
        ),  # loads the n chronologically first recordings
        target_name=target_name,  # age, gender, pathology
        preload=False,
        add_physician_reports=True,
        n_jobs=num_workers,
    )

    # ## Preprocessing

    # text preprocessing
    dataset.set_description(
        text_preprocessing(dataset.description, processed_categories=category),
        overwrite=True,
    )

    # EEG preprocessing

    # Preprocess the data
    if not nailcluster:
        print("Preprocessing EEG data")
        preprocess(dataset, preprocess_config.preprocessors)

    # ## Data Splitting
    # TODO : split using train and test splits instead
    # TODO : maybe load TUH now on top of TUH Abnormal ?

    if crossval:
        subject_datasets = dataset.split("subject")
        n_subjects = len(subject_datasets)
        print(n_subjects)
        keys = list(subject_datasets.keys())

        folds = np.array_split(keys, folds_nb)

        train_keys = np.concatenate(
            [folds[i] for i in range(folds_nb) if i != fold_idx]
        )
        valid_keys = folds[fold_idx]

        # train_keys = keys[:int(n_subjects * 0.70)]
        # valid_keys = keys[int(n_subjects * 0.70):n_subjects]

        print(len(train_keys), len(valid_keys))

        train_sets = [d for k in train_keys for d in subject_datasets[k].datasets]
        train_set = BaseConcatDataset(train_sets)

        valid_sets = [d for k in valid_keys for d in subject_datasets[k].datasets]
        valid_set = BaseConcatDataset(valid_sets)

    else:
        train_set = dataset.split("train")["True"]
        """ use only first 75 percent of the train ds
        subject_datasets = train_set.split('subject')
        n_subjects = len(subject_datasets)

        n_split = int(np.round(n_subjects * 0.75))
        keys = list(subject_datasets.keys())
        train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
        train_set = BaseConcatDataset(train_sets)
        """

        valid_set = dataset.split("train")["False"]  # wrong. but wont be used anyways.

    window_train_set = create_fixed_length_windows(
        train_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=False,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=True,
    )

    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=False,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
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
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    n_chans = 21  # number of channels in the EEG data

    wandb_logger = WandbLogger(
        project="EEGClip",
        save_dir=results_dir + "/wandb",
        log_model=False,
        # checkpoint_name = 'checkpoint.ckpt',
        # tags = ["hpo_emb_size-"]
    )

    # ## Training
    trainer = Trainer(
        default_root_dir=results_dir + "/models",
        accelerator="gpu",
        devices=1,  # TODO : see why using 2 gpus kills the process
        strategy="auto",
        # strategy="ddp_find_unused_parameters_true",
        max_epochs=n_epochs,
        logger=wandb_logger,
    )
    trainer.fit(
        EEGClipModel(
            n_chans=n_chans,
            lr=lr,
            lr_frac_lm=lr_frac_lm,
            weight_decay=weight_decay,
            string_sampling=string_sampling,
            projected_emb_dim=projected_emb_dim,
            num_fc_layers=num_fc_layers,
            text_encoder_name=text_encoder_name,
            text_encoder_trainable=text_encoder_trainable,
            text_model_emb_dim=text_model_emb_dim,
        ),
        train_loader,
        valid_loader,
    )
    """
    trainer.save_checkpoint(results_dir + "/models/EEGClip_100_"+
                                            text_encoder_name +
                                            "_" +
                                            str(projected_emb_dim)+
                                            ".ckpt")
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEGClip on TUH EEG dataset.")
    parser.add_argument(
        "--n_rec",
        type=int,
        default=2993,
        help="Number of recordings to load from TUH EEG dataset.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs to train EEGClip model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size to train EEGClip model."
    )
    parser.add_argument(
        "--projected_emb_dim",
        type=int,
        default=64,
        help="Final embedding size for the EEGClip model.",
    )
    parser.add_argument(
        "--text_model_emb_dim",
        type=int,
        default=768,
        help="embedding size for the text model.",
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        default=3,
        help="nb layers in the projection modules",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-3, help="Learning rate to train EEGClip model."
    )
    parser.add_argument(
        "--lr_frac_lm",
        type=float,
        default=0,
        help="Learning rate for the LM module (as a fraction of --lr).",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="medicalai/ClinicalBERT",
        help="name of the text encoder",
    )
    parser.add_argument(
        "--category", type=str, default="all", help="report category to keep/exclude"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay to train EEGClip model.",
    )
    parser.add_argument(
        "--string_sampling",
        action="store_true",
        help="Whether to use string sampling : random sampling of sentences in each batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of workers to use for data loading.",
    )
    parser.add_argument(
        "--crossval", action="store_true", help="Whether to do crossvalidation"
    )
    parser.add_argument(
        "--folds_nb", type=int, default=5, help="nb of folds for cross-validation"
    )
    parser.add_argument(
        "--fold_idx", type=int, default=0, help="(0 to folds_nb - 1) valid fold index"
    )
    args = parser.parse_args()

    target_name = "report"  # ('report', 'pathological', 'age', 'gender')
    # TODO : find a way to use several targets

    run_training(
        target_name=target_name,
        n_recordings_to_load=args.n_rec,
        n_epochs=args.n_epochs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_frac_lm=args.lr_frac_lm,
        weight_decay=args.weight_decay,
        string_sampling=args.string_sampling,
        text_encoder_name=args.text_encoder_name,
        text_encoder_trainable=False,
        text_model_emb_dim=args.text_model_emb_dim,
        category=args.category,
        projected_emb_dim=args.projected_emb_dim,
        num_fc_layers=args.num_fc_layers,
        crossval=args.crossval,
        folds_nb=args.folds_nb,
        fold_idx=args.fold_idx,
    )
