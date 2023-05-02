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
    text_model_emb_dim = 768
    projected_emb_dim = 256
    temperature = 1.0

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
    
class EEGEncoder(nn.Module):
    def __init__(self, 
                 eeg_model_emb_dim=128,
                 n_chans=21,
                 eeg_model_pretrained=False,
                 eeg_model_trainable=True,
                ):
        super().__init__()

        self.model = Deep4Net(
            in_chans=n_chans,
            n_classes=eeg_model_emb_dim, 
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
        ).create_network()
        
        if eeg_model_pretrained:
            self.model.load_state_dict(torch.load('deep4net_trained.pt'))
            
        for param in self.model.parameters():
            param.requires_grad = eeg_model_trainable

    def forward(self, x):
        return self.model(x)
    
class ProjectionHead(nn.Module):
    def __init__(self, 
                 input_dim=128,
                 output_dim=128,
                 dropout=0.1,
                 num_layers=3,
                ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                self.layers.append(nn.Linear(output_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x)))
        return x
    
class EEGClip(pl.LightningModule):
    def __init__(self, 
                 eeg_model_emb_dim=128,
                 text_model_emb_dim=768,
                 projected_emb_dim=64,
                 text_encoder_name="bert-base-uncased",
                 text_encoder_pretrained=True,
                 text_encoder_trainable=False,
                 eeg_model_pretrained=False,
                 eeg_model_trainable=True,
                 dropout=0.1,
                 num_proj_layers=3,
                 lr=1e-3,
                 weight_decay=1e-6,
                 num_classes=2,
                 n_chans=21,
                 **kwargs
                ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.n_chans = n_chans
        
        self.text_encoder = TextEncoder(
            text_encoder_name=text_encoder_name,
            text_encoder_pretrained=text_encoder_pretrained,
            text_encoder_trainable=text_encoder_trainable,
        )
        
        self.eeg_encoder = EEGEncoder(
            eeg_model_emb_dim=eeg_model_emb_dim,
            n_chans=n_chans,
            eeg_model_pretrained=eeg_model_pretrained,
            eeg_model_trainable=eeg_model_trainable,
        )
        
        self.text_projection = ProjectionHead(
            input_dim=text_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout=dropout,
            num_layers=num_proj_layers,
        )
        
        self.eeg_projection = ProjectionHead(
            input_dim=eeg_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout=dropout,
            num_layers=num_proj_layers,
        )

        
        
