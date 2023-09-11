import os
import argparse
import socket
import pandas as pd
import numpy as np
import torch 
import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.manifold import TSNE

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

from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from EEGClip.classifier_models import EEGClassifierModel
from EEGClip.clip_models import EEGClipModel

import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted


import EEGClip_config

"""
This script is used to train a classifier model zero-shot style
on the TUH EEG dataset for different classification tasks :
pathological
age
gender
report-based (medication, diagnosis ...)

"""





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an EEG classifier on the TUH EEG dataset.')
    parser.add_argument('--task_name', type=str, default="pathological",
                        help='classification task name (pathological, age, gender, report-related tasks ....')    
    parser.add_argument('--n_rec', type=int, default=2993,
                        help='Number of recordings to load from TUH EEG dataset.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers to use for data loading.')


    args = parser.parse_args()

    num_workers = args.num_workers

    n_recordings_to_load = args.n_rec
    mapping = None
    task_name = args.task_name #('pathological', 'age', 'gender', "epilep", "keppra", "dilantin", "seizure")
    if task_name in ['pathological', 'age', 'gender']: 
        target_name = task_name 
    elif task_name == "under_50":
        target_name = "age"
    else:
        target_name = "report"

    if task_name =="gender":
        s0 = "The patient is male"
        s1 = "The patient is female"
        mapping = {'M': 0, 'F': 1}

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

        s0 = "no anti-epileptic drugs were prescribed to the patient" #"The patient is unlikely to have been prescribed anti-epileptic drugs (anticonvulsants, keppra, dilantin or depakote), used to control seizures"  #"keppra", "dilantin", "depakote"
        s1 = "anti-epileptic drugs medication was prescribed to the patient" #"The patient is likely to have been prescribed anti-epileptic drugs (anticonvulsants, keppra, dilantin or depakote), used to control seizures"  #"keppra, dilantin or depakote" 


    # TODO : find a way to use several targets
    n_max_minutes = 3
    sfreq = 100
    n_minutes = 2
    input_window_samples = 1200
    
    batch_size = 64

    nailcluster = (socket.gethostname() == "vs3-0")

    num_workers = args.num_workers
    
    results_dir = EEGClip_config.results_dir
    tuh_data_dir = EEGClip_config.tuh_data_dir

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
    
    n_split = int(np.round(n_subjects * 0.70))
    keys = list(subject_datasets.keys())

    train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
    train_set = BaseConcatDataset(valid_sets)

    valid_sets = [d for i in range(n_split, n_subjects) for d in subject_datasets[keys[i]].datasets]
    valid_set = BaseConcatDataset(valid_sets)

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
    valid_loader = torch.utils.data.DataLoader(
        window_valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    print(len(valid_loader.dataset))
    

    # ## Create model

    eegclipmodel = EEGClipModel.load_from_checkpoint(EEGClip_config.model_paths["eegclip"])
    eegclipmodel.cuda()
    EEGEncoder = torch.nn.Sequential(eegclipmodel.eeg_encoder,eegclipmodel.eeg_projection)
    # get size of the last layer
    instructor_model = INSTRUCTOR('hkunlp/instructor-xl')

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
        """
        instruction = "Represent the medical report: "
        emb = instructor_model.encode([[instruction,sentence]])[0]
        emb = torch.Tensor(emb).to(device='cuda:0')
        emb = eegclipmodel.text_projection(emb)
        emb = emb.detach().cpu().numpy()    
        return emb



    s0_embed = sentence_embedder(s0)
    s1_embed = sentence_embedder(s1)
    ## get embeddings for the validation set using the EEG encoder

    for param in EEGEncoder.parameters():
        param.requires_grad = False

    # iterate over the validation set and get the embeddings
    embeddings_train = []
    labels_train  = []

    for batch in tqdm.tqdm(train_loader):
        eeg, label, id = batch
        eeg = eeg.cuda()
        eeg = EEGEncoder(eeg)
        eeg = torch.mean(eeg, dim=2)
        embeddings_train.append(eeg.detach().cpu().numpy())
        labels_train.append(label)
    
    embeddings_train = np.concatenate(embeddings_train)
    labels_train = np.concatenate(labels_train)

    embeddings_valid = []
    labels_valid  = []

    for batch in tqdm.tqdm(valid_loader):
        eeg, label, id = batch
        eeg = eeg.cuda()
        eeg = EEGEncoder(eeg)
        eeg = torch.mean(eeg, dim=2)
        embeddings_valid.append(eeg.detach().cpu().numpy())
        labels_valid.append(label)
    
    embeddings_valid = np.concatenate(embeddings_valid)
    labels_valid = np.concatenate(labels_valid)

    
    if task_name == "medication":
        labels_valid = [1 if any(med in string.lower() for med in medication_list) else 0 for string in labels_valid]
    

    if task_name == "epilep":
        labels_valid = [0 if "epilep" not in l.lower() or "no epilep" in l.lower() else 1 for l in labels_valid]
    
    if task_name == "seizure":
        labels_valid = [0 if "seizure" not in l.lower() or "no seizure" in l.lower() else 1 for l in labels_valid]

    if task_name == "under_50":
        labels_valid = [0 if age >= 50 else 1 for age in labels_valid]
    
    distance_classifier = []
    for r in embeddings_valid:
        d0 = distance.cosine(r, s0_embed)
        d1 = distance.cosine(r, s1_embed)
        if d0 < d1:
            distance_classifier.append(0)
        else:
            distance_classifier.append(1)

    print("label balance :", np.mean(distance_classifier))

    # compare to the actual labels
    print("Accuracy: ", balanced_accuracy_score(labels_valid, distance_classifier))

    ## plot the embeddings in 2D using TSNE
    features2d = TSNE(n_components=2).fit_transform(embeddings_valid)
    
    plt.scatter([a[0] for a in features2d],
                [a[1] for a in features2d],
                c=labels_valid)
    plt.savefig(EEGClip_config.results_dir + "clip_graphs/tsne_map.png")

