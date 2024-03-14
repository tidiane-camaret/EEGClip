import os

import pandas as pd
import torch

import configs.preprocess_config as preprocess_config

device = torch.device("cuda")


num_workers = 32

tuh_data_dir = preprocess_config.tuh_data_dir
"""
dataset = TUHAbnormal(
    path=tuh_data_dir,
    recording_ids=None,  # range(n_recordings_to_load),  # loads the n chronologically first recordings
    target_name="report",  # age, gender, pathology
    preload=False,
    add_physician_reports=True,
    n_jobs=num_workers,
)

# ## Preprocessing

# text preprocessing
from EEGClip.text_preprocessing import text_preprocessing

embs_df = text_preprocessing(dataset.description)
embs_df = embs_df[["report"]]
"""
embs_df = pd.read_csv("scripts/text_preprocessing/embs_df.csv")

from InstructorEmbedding import INSTRUCTOR

instructor_model = INSTRUCTOR("hkunlp/instructor-xl")


def sentence_embedder(sentence):  # , tokenizer, model):
    instruction = "Represent the medical report: "
    emb = instructor_model.encode([[instruction, sentence]])[0]
    emb = torch.Tensor(emb).to(device="cuda:0")
    emb = emb.detach().cpu().numpy()
    return emb


model_name = "hkunlp/instructor-xl"
print("model_name : ", model_name)
"""
bert-base-uncased
"BAAI/bge-large-en-v1.5"
medicalai/ClinicalBERT
hkunlp/instructor-xl
microsoft/BioGPT-Large-PubMedQA
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.to(device)
"""
embs = []

import numpy as np

for i, r in enumerate(embs_df["report"]):
    if i % 100:
        print(i, "/", len(embs_df["report"]))
    emb = sentence_embedder(r)  # , tokenizer, model)

    embs.append(emb)

embs = np.array(embs).tolist()  # important so that whole arrays are copied to the df
embs_df[model_name] = embs

embs_df.to_csv(
    os.path.join(preprocess_config.ROOT_DIR, "scripts/text_preprocessing/embs_df.csv")
)
