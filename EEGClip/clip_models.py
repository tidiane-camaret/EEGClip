import os
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pytorch_lightning as pl

class CFG:
    debug = False
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    #image_encoder_lr = 1e-4
    category_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    eeg_embedding_dim = 768 #2048
    nb_categories = 2
    category_embedding_dim = 768
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both eeg encoder and text encoder
    trainable = True # for both eeg encoder and text encoder
    temperature = 1.0


    # for projection head; used for both eeg and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class CategoryEncoder(nn.Module):
    def __init__(self, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = nn.Embedding(CFG.nb_categories, CFG.category_embedding)
        else:
            self.model = nn.Embedding.from_pretrained(torch.rand((CFG.nb_categories, CFG.category_embedding)))
            
        for p in self.model.parameters():
            p.requires_grad = trainable

class EEGEncoder(nn.Module):
    def __init__(self, eeg_classifier, trainable=CFG.trainable):
        super().__init__()
        # TODO: add pretrained models
        self.model = torch.nn.Sequential(*list(eeg_classifier.children())[:-1])
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input):
        output = self.model(input)
        return output

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class EEGClipModule(pl.LightningModule):
    def __init__(
        self,
        temperature=CFG.temperature,
        eeg_embedding_dim=CFG.eeg_embedding_dim,
        text_embedding_dim=CFG.text_embedding_dim,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.text_encoder = TextEncoder()
        self.eeg_projection = ProjectionHead(embedding_dim=eeg_embedding_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim)
        self.temperature = temperature


    def forward(self, batch):
        x, y, z = batch
        eeg_features = self.eeg_encoder(x)
        text_features = self.text_encoder(y)

        eeg_embeddings = self.eeg_projection(eeg_features)
        text_embeddings = self.text_projection(text_features)

        return eeg_embeddings, text_embeddings

    def training_step(self, batch, batch_idx):
        eeg_embeddings, text_embeddings = self.forward(batch)

        logits = (text_embeddings @ eeg_embeddings.T) / self.temperature
        categories_similarity = eeg_embeddings @ eeg_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (categories_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        categories_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (categories_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]
