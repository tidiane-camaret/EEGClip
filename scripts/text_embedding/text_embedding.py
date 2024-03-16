import copy
import json

import configs.preprocess_config as preprocess_config

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

def sentence_embedder(sentence,tokenizer,model,max_length=512):
    with torch.no_grad():
        desc_tokenized = tokenizer(sentence, return_tensors="pt", max_length=max_length, padding=True, truncation=True).to(device)
        outputs = model(**desc_tokenized)
        emb = outputs.to_tuple()[0][0][0].detach().cpu().numpy()
    return emb


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
embs_df = pd.read_csv("scripts/text_preprocessing/embs_df.csv")


model_name = "Salesforce/SFR-Embedding-Mistral"
max_length = 512
"""
bert-base-uncased
"BAAI/bge-large-en-v1.5"
medicalai/ClinicalBERT
hkunlp/instructor-xl
microsoft/BioGPT-Large-PubMedQA
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
mixedbread-ai/mxbai-embed-large-v1
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
"""
embs = []

for i, r in enumerate(embs_df["report"]):
    if i % 100:
        print(i, "/", len(embs_df["report"]))
    emb = sentence_embedder(r, tokenizer, model, max_length)

    embs.append(emb)

embs_df[model_name] = np.array(embs).tolist()

embs_df.to_csv("scripts/text_preprocessing/embs_df.csv")

"""
### Encoding of zero-shot sentences

zc_sentences_dict = {
    "pathological": {
        "s0": "This is a normal recording, from an healthy patient",
        "s1": "This an pathological recording, from a diseased patient  ",
    },
    "gender": {"s0": "The patient is female", "s1": "The patient is male"},
    "under_50": {
        "s0": "The patient is over 50 years old",
        "s1": "The patient is under 50 years old",
    },
    "medication": {
        "s0": "The patient is not taking anti-epileptic medication",
        "s1": "The patient is taking anti-epileptic medication",
    },
}


zc_sentences_model_emb_dict = copy.deepcopy(zc_sentences_dict)
for label, sentences in zc_sentences_dict.items():
    for l, s in sentences.items():
        zc_sentences_model_emb_dict[label][l] = sentence_embedder(s, tokenizer, model, max_length).tolist()




with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
            zc_sentences_emb_dict = json.load(f)


zc_sentences_emb_dict[model_name] = zc_sentences_model_emb_dict


with open(preprocess_config.zc_sentences_emb_dict_path, "w") as f:
    json.dump(zc_sentences_emb_dict, f)
