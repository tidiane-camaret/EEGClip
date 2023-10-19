import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda")
"""
from braindecode.datasets import TUHAbnormal
num_workers = 32
tuh_data_dir = "/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/v2.0.0/edf/"

dataset = TUHAbnormal(
    path=tuh_data_dir,
    recording_ids=None,#range(n_recordings_to_load),  # loads the n chronologically first recordings
    target_name="report",  # age, gender, pathology
    preload=False,
    add_physician_reports=True,
    n_jobs=num_workers)

# ## Preprocessing

# text preprocessing
from EEGClip.text_preprocessing import text_preprocessing
embs_df = text_preprocessing(dataset.description)
embs_df = embs_df[["report"]]
"""
embs_df = pd.read_csv("/home/jovyan/EEGClip/scripts/text_preprocessing/embs_df.csv")

def sentence_embedder(sentence,tokenizer,model):
    
    desc_tokenized = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding='max_length').to(device)
    outputs = model(**desc_tokenized)
    emb = outputs.to_tuple()[0][0][0].detach().cpu().numpy()
    return emb


model_name = "medicalai/ClinicalBERT"
"""
bert-base-uncased
"BAAI/bge-large-en-v1.5"
medicalai/ClinicalBERT
hkunlp/instructor-xl
microsoft/BioGPT-Large-PubMedQA
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device)

embs = []

import numpy as np
for i, r in enumerate(embs_df['report']):
    if i % 100:
        print(i,'/',len(embs_df['report']))
    emb = sentence_embedder(r,tokenizer,model)
    
    embs.append(emb)

embs = np.array(embs).tolist() # important so that whole arrays are copied to the df
embs_df[model_name] = embs

embs_df.to_csv("/home/jovyan/EEGClip/scripts/text_preprocessing/embs_df.csv")