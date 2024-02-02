import argparse
import copy
import socket

import mne
import numpy as np
import pandas as pd
import torch
import tqdm
from braindecode.datasets import TUHAbnormal
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import create_fixed_length_windows, preprocess
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds

from EEGClip.clip_models import EEGClipModel
from EEGClip.text_preprocessing import text_preprocessing

mne.set_log_level("ERROR")  # avoid messages everytime a window is extracted


import configs.EEGClip_config as EEGClip_config

"""
This script uses EEG-Clip representations to do recording retrieval :
given a EEG segment, retrive the corresponding text description, and vice-versa
metrics : median rank, recall@k

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an EEG classifier on the TUH EEG dataset."
    )
    parser.add_argument(
        "--n_rec",
        type=int,
        default=2993,
        help="Number of recordings to load from TUH EEG dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for data loading.",
    )

    args = parser.parse_args()

    num_workers = args.num_workers

    n_recordings_to_load = args.n_rec

    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200

    batch_size = 64

    nailcluster = socket.gethostname() == "vs3-0"

    results_dir = EEGClip_config.results_dir
    tuh_data_dir = EEGClip_config.tuh_data_dir

    # TODO : use get_output_shape (requires to load the model first)
    n_preds_per_input = (
        519  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
    )

    seed = 20210325  # random seed to make results reproducible

    cuda = torch.cuda.is_available()
    set_random_seeds(seed=seed, cuda=cuda)
    torch.backends.cudnn.benchmark = True

    # ## Load data
    dataset = TUHAbnormal(
        path=tuh_data_dir,
        recording_ids=range(
            n_recordings_to_load
        ),  # loads the n chronologically first recordings
        target_name="report",  # age, gender, pathology
        preload=False,
        add_physician_reports=True,
        n_jobs=1,
    )

    dataset.set_description(
        text_preprocessing(dataset.description, processed_categories="all"),
        overwrite=True,
    )

    # ## Preprocessing

    if not nailcluster:
        preprocess(dataset, EEGClip_config.preprocessors)

    # ## Data Splitting
    # TODO : split using train and test splits instead
    # TODO : maybe load TUH now on top of TUH Abnormal ?

    # dataset = dataset.split('train')['True']

    n_subjects = len(dataset.split("subject"))

    print("Nb subjects loaded : ", n_subjects)

    valid_set = dataset.split("train")["False"]

    window_valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=60 * sfreq,
        stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
        preload=True,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
    )

    ### PREPROCESSING NECESSARY IF USING TUH_PRE
    if nailcluster:
        window_valid_set.transform = lambda x: x * 1e6

    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(len(valid_loader.dataset))

    # ## Create model

    eegclipmodel = EEGClipModel.load_from_checkpoint(
        EEGClip_config.model_paths["eegclip"]
    )
    eegclipmodel.cuda()
    EEGEncoder = torch.nn.Sequential(
        eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
    )
    # get size of the last layer
    text_encoder_name = "medicalai/ClinicalBERT"

    for param in EEGEncoder.parameters():
        param.requires_grad = False

    embs_df = pd.read_csv(EEGClip_config.embs_df_path)
    embs_name = text_encoder_name
    for r in range(len(embs_df)):
        re = copy.copy(embs_df[embs_name][r])
        # convert the string to array
        re = re.replace("[", "")
        re = re.replace("]", "")
        re = re.replace(",", "")
        re = re.split()
        re = [float(i) for i in re]
        embs_df[embs_name][r] = re
    # iterate over the validation set and get the embeddings
    eeg_embs = []
    text_embs = []
    for batch in tqdm.tqdm(valid_loader):
        eeg, text, id = batch
        eeg = eeg.cuda()
        eeg = EEGEncoder(eeg)
        eeg = torch.mean(eeg, dim=2)
        eeg_embs.append(eeg.detach().cpu().numpy())

        text_emb = []
        for s in text:
            lookup = embs_df.loc[embs_df["report"] == s, text_encoder_name]

            emb = lookup.tolist()[0]
            text_emb.append(emb)
        text_emb = torch.Tensor(text_emb).to(device="cuda:0")
        text_emb = eegclipmodel.text_projection(text_emb)
        text_embs.append(text_emb.detach().cpu().numpy())

    eeg_embs = np.concatenate(eeg_embs)
    text_embs = np.concatenate(text_embs)

    print(eeg_embs.shape)
    print(text_embs.shape)
