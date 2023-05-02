import os
import gc
import random
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import torch
from torch import nn
import torch.nn.functional as F

from braindecode.models import Deep4Net
from braindecode.training.scoring import trial_preds_from_window_preds

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

import pytorch_lightning as pl
import wandb 

class CFG:
    """Configuration class for the EEGClip model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8

class TextEncoder(nn.Module):
    def __init__(self, 
                 text_encoder_name="bert-base-uncased",
                 text_encoder_pretrained=True,
                 text_encoder_trainable=False,
                ):
        super().__init__()
        if text_encoder_pretrained:
            self.model = AutoModel.from_pretrained(text_encoder_name, output_hidden_states=True)
        else:
            self.model = AutoModel(config=AutoConfig())

        for param in self.model.parameters():
            param.requires_grad = text_encoder_trainable
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

    def forward(self, string_batch):
        input_ids = self.tokenizer(string_batch, padding=True, truncation=True, return_tensors="pt").input_ids.to(CFG.device)
        outputs = self.model(input_ids)
        return outputs.last_hidden_state[:,0,:]