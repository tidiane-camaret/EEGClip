"""launcher for training eegclip model
"""
import os
import pprint
import socket
from dataclasses import dataclass

import hydra
import mne
import numpy as np

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


@hydra.main(version_base=None, config_path="../configs/", config_name="base_exp")
def main(config):
    pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))
    args = Args()
    args.lr = config.eegclip.lr
    args.lr_frac_lm = config.eegclip.text_encoder.lr_frac_lm
    args.text_encoder_name = config.eegclip.text_encoder.pretrained_name
    args.weight_decay = config.eegclip.weight_decay
    args.string_sampling = config.eegclip.text_encoder.string_sampling
    args.projected_emb_dim = config.eegclip.projected_emb_dim
    args.num_fc_layers = config.eegclip.num_fc_layers
    args.target_name = config.dataset.target_name
    args.processed_categories = config.dataset.processed_categories
    args.text_encoder_trainable = config.eegclip.text_encoder.trainable
    args.text_encoder_emb_dim = config.eegclip.text_encoder.emb_dim
    args.text_encoder_max_token_len = config.eegclip.text_encoder.max_token_len
    args.n_recordings = config.dataset.n_recordings
    args.n_epochs = config.training.n_epochs
    args.num_workers = config.training.num_workers
    args.batch_size = config.training.batch_size
    args.crossval = config.training.cross_validation
    args.n_folds = config.training.n_folds
    args.fold_idx = config.training.fold_idx
    args.preprocessing = config.dataset.preprocessing
    args.seed = config.training.seed
    args.n_preds_per_input = config.eegclip.eeg_encoder.n_preds_per_input
    args.n_eeg_channels = config.eegclip.eeg_encoder.n_eeg_channels
    args.contrastive_loss_temperature = config.eegclip.contrastive_loss.temperature
    args.contrastive_loss_func = config.eegclip.contrastive_loss.func
    run_eegclip_training(args)


def run_eegclip_training(args):
    import torch
    lr = args.lr
    lr_frac_lm = args.lr_frac_lm
    text_encoder_name = args.text_encoder_name
    weight_decay = args.weight_decay
    string_sampling = args.string_sampling
    projected_emb_dim = args.projected_emb_dim
    num_fc_layers = args.num_fc_layers
    target_name = args.target_name
    processed_categories = args.processed_categories
    text_encoder_trainable = args.text_encoder_trainable
    text_encoder_emb_dim = args.text_encoder_emb_dim
    n_recordings_to_load = args.n_recordings
    n_epochs = args.n_epochs
    num_workers = args.num_workers
    batch_size = args.batch_size
    crossval = args.crossval
    n_folds = args.n_folds
    fold_idx = args.fold_idx
    contrastive_loss_temperature = args.contrastive_loss_temperature
    text_encoder_max_token_len = args.text_encoder_max_token_len
    contrastive_loss_func = args.contrastive_loss_func
    

    print("processed categories : ", processed_categories)
    nailcluster = (
        socket.gethostname() == "vs3-0"
    )  # check if we are on the nail cluster or on kislurm

    results_dir = preprocess_config.results_dir
    tuh_data_dir = preprocess_config.tuh_data_dir


    sfreq = args.preprocessing.sfreq
    n_minutes = args.preprocessing.n_minutes
    input_window_samples = args.preprocessing.input_window_samples
    # TODO : use get_output_shape (requires to load the model first)
    n_preds_per_input = (
        args.n_preds_per_input  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
    )

    cuda = torch.cuda.is_available()
    seed = args.seed  # random seed to make results reproducible
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
        text_preprocessing(dataset.description, processed_categories=processed_categories),
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

        folds = np.array_split(keys, n_folds)

        train_keys = np.concatenate(
            [folds[i] for i in range(n_folds) if i != fold_idx]
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
            n_chans=args.n_eeg_channels,
            lr=lr,
            lr_frac_lm=lr_frac_lm,
            weight_decay=weight_decay,
            string_sampling=string_sampling,
            projected_emb_dim=projected_emb_dim,
            num_fc_layers=num_fc_layers,
            text_encoder_name=text_encoder_name,
            text_encoder_trainable=text_encoder_trainable,
            text_encoder_emb_dim=text_encoder_emb_dim,
            contrastive_loss_temperature=contrastive_loss_temperature,
            text_encoder_max_token_len=text_encoder_max_token_len,
            contrastive_loss_func = contrastive_loss_func
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


@dataclass
class Args:
    target_name: str = "report"
    """which target (label) to load from TUH EEG dataset."""
    n_recordings: int = 2993
    """number of recordings to load from TUH EEG dataset."""
    n_epochs: int = 20
    """number of epochs to train EEGClip model."""
    batch_size: int = 64
    """batch size"""
    projected_emb_dim: int = 64
    """final embedding size after projection"""
    text_encoder_emb_dim: int = 1024
    """embedding size for the text encoder"""
    num_fc_layers: int = 3
    """nb layers in the projection modules"""
    lr: float = 5e-3
    """learning rate to train EEGClip model."""
    lr_frac_lm: float = 0
    """learning rate for the LM module (as a fraction of --lr)."""
    text_encoder_name: str = "medicalai/ClinicalBERT"
    """name of the text encoder for import"""
    text_encoder_trainable: bool = False
    """whether to train the text encoder"""
    processed_categories: str = "all"
    """report categories to keep for training, "all" or "none" or a single category"""
    weight_decay: float = 5e-4
    """weight decay to train EEGClip model."""
    string_sampling: bool = False
    """whether to use string sampling : random sampling of sentences in each batch"""
    num_workers: int = 20
    """number of workers to use for data loading."""
    crossval: bool = False
    """whether to do crossvalidation"""
    n_folds: int = 5
    """nb of folds for cross-validation"""
    fold_idx: int = 0
    """(0 to n_folds - 1) valid fold index"""


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main()  # data processing might error out due to multiple jobs doing the same thing
        print(e)
