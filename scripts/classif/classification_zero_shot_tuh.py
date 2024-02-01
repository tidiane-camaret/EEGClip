import argparse
import socket

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import tqdm
from braindecode.datasets import TUHAbnormal
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
)
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModel, AutoTokenizer

from EEGClip.clip_models import EEGClipModel
from EEGClip.text_preprocessing import text_preprocessing

mne.set_log_level("ERROR")  # avoid messages everytime a window is extracted


import config.EEGClip_config as EEGClip_config

"""
This script trains a classifier model zero-shot style
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
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for data loading.",
    )

    args = parser.parse_args()

    num_workers = args.num_workers

    n_recordings_to_load = args.n_rec
    mapping = None

    task_name = (
        args.task_name
    )  # ('pathological', 'age', 'gender', "epilep", "keppra", "dilantin", "seizure","pathological_gender")

    # DEFINE THE TARGET NAMES FOR THE TUHAbnormal CLASS
    if task_name in ["pathological", "age", "gender"]:
        target_name = task_name
    elif task_name == "under_50":
        target_name = "age"
    elif task_name == "pathological_gender":
        target_name = ("pathological","gender")
    else:
        target_name = "report"

    # DEFINE THE PROMPTS 
    if task_name == "gender":
        s0 = "The patient is male"
        s1 = "The patient is female"
        mapping = {"M": 0, "F": 1}

    if task_name == "pathological":
        s0 = "This is a normal recording."
        s1 = "This is an abnormal recording."

    if task_name == "under_50":
        s0 = "The patient is over 50 year old"
        s1 = "The patient is under 50 year old"

    if task_name == "epilep":
        s0 = "The patient does not have epilepsy"
        s1 = "The patient has epilepsy"

    if task_name == "seizure":
        s0 = "The patient does not have seizures"
        s1 = "The patient has seizures"

    if task_name == "medication":
        medication_list = ["keppra", "dilantin", "depakote"]

        s0 = "no anti-epileptic drugs were prescribed to the patient"  # "The patient is unlikely to have been prescribed anti-epileptic drugs (anticonvulsants, keppra, dilantin or depakote), used to control seizures"  #"keppra", "dilantin", "depakote"
        s1 = "anti-epileptic drugs medication were prescribed to the patient"  # "The patient is likely to have been prescribed anti-epileptic drugs (anticonvulsants, keppra, dilantin or depakote), used to control seizures"  #"keppra, dilantin or depakote"

    if task_name == "pathological_gender":
        s0 = "The patient is male and has no pathology"
        s1 = "The patient is male and has a pathology"
        s2 = "The patient is female and has no pathology"
        s3 = "The patient is female and has a pathology"
        mapping = {"M": 0, "F": 1, True: 0, False: 1}

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
        target_name=target_name,  # age, gender, pathology
        preload=False,
        add_physician_reports=False,
        n_jobs=args.num_workers,
    )
    dataset.set_description(
        text_preprocessing(dataset.description, processed_categories="all"),
        overwrite=True,
    )

    # ## Preprocessing


    # Preprocess the data
    if not nailcluster:
        preprocess(dataset, EEGClip_config.preprocessors)

    # ## Data Splitting
    # TODO : split using train and test splits instead
    # TODO : maybe load TUH now on top of TUH Abnormal ?


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
        mapping=mapping,
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
        EEGClip_config.model_paths["eegclip_bert"]
    )
    eegclipmodel.cuda()
    EEGEncoder = torch.nn.Sequential(
        eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
    )
    # get size of the last layer
    text_encoder_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    language_model = AutoModel.from_pretrained(text_encoder_name)

    def sentence_embedder(sentence):
        """
        desc_tokenized = bert_tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        outputs = bert_model(**desc_tokenized)
        emb = outputs.to_tuple()[0][0][0].detach().numpy().tolist()


        desc_tokenized = instructor_tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        outputs = instructor_model.encoder(**desc_tokenized)
        emb = outputs.to_tuple()[0][0][0]
        emb = eegclipmodel.text_projection(emb)
        emb = emb.detach().cpu().numpy()

        instruction = "Represent the medical report: "
        emb = instructor_model.encode([[instruction,sentence]])[0]
        emb = torch.Tensor(emb).to(device='cuda:0')
        emb = eegclipmodel.text_projection(emb)
        emb = emb.detach().cpu().numpy()
        """

        desc_tokenized = tokenizer(
            sentence,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        outputs = language_model(**desc_tokenized)
        emb = outputs.to_tuple()[0][0][0].detach().cpu().numpy()
        emb = torch.Tensor(emb).to(device="cuda:0")
        emb = eegclipmodel.text_projection(emb)
        emb = emb.detach().cpu().numpy()
        return emb

    s0_embed = sentence_embedder(s0)
    s1_embed = sentence_embedder(s1)
    if task_name == "pathological_gender":
        s2_embed = sentence_embedder(s2)
        s3_embed = sentence_embedder(s3)

    ## get embeddings for the validation set using the EEG encoder

    for param in EEGEncoder.parameters():
        param.requires_grad = False

    # iterate over the validation set and get the embeddings
    embeddings = []
    labels = []

    for batch in tqdm.tqdm(valid_loader):
        eeg, label, id = batch
        print(label)
        eeg = eeg.cuda()
        eeg = EEGEncoder(eeg)
        eeg = torch.mean(eeg, dim=2)
        embeddings.append(eeg.detach().cpu().numpy())
        labels.append(label)

    embeddings = np.concatenate(embeddings)
    print(labels)
    print(labels[0])
    labels = np.concatenate(labels, axis=-1)

    if task_name == "medication":
        labels = [
            1 if any(med in string.lower() for med in medication_list) else 0
            for string in labels
        ]

    if task_name == "epilep":
        labels = [
            0 if "epilep" not in l.lower() or "no epilep" in l.lower() else 1
            for l in labels
        ]

    if task_name == "seizure":
        labels = [
            0 if "seizure" not in l.lower() or "no seizure" in l.lower() else 1
            for l in labels
        ]

    if task_name == "under_50":
        labels = [0 if age >= 50 else 1 for age in labels]


    distance_classifier = []
    for r in embeddings:
        d0 = distance.cosine(r, s0_embed)
        d1 = distance.cosine(r, s1_embed)
        
        if task_name == "pathological_gender":
            d2 = distance.cosine(r, s2_embed)
            d3 = distance.cosine(r, s3_embed)

            d = np.argsort([d0, d1, d2, d3])
            distance_classifier.append(d)
        else:
            if d0 < d1:
                distance_classifier.append(0)
            else:
                distance_classifier.append(1)    
        


    print("label balance :", np.mean(distance_classifier))

    # compare to the actual labels
    print("Accuracy: ", balanced_accuracy_score(labels, distance_classifier))

    ## plot the embeddings in 2D using TSNE
    features2d = TSNE(n_components=2).fit_transform(embeddings)

    plt.scatter([a[0] for a in features2d], [a[1] for a in features2d], c=labels)
    plt.savefig(EEGClip_config.results_dir + "clip_graphs/tsne_map_"+args.task_name+".png")
