import random
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import torch
from torch import nn
import torch.nn.functional as F

from braindecode.models import Deep4Net
from braindecode.training.scoring import trial_preds_from_window_preds

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

import pytorch_lightning as pl

class CFG:
    """Configuration class for the EEGClip model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    text_model_emb_dim = 768
    temperature = 1.0
    classifiers_dict = {
        'knn': KNeighborsClassifier(n_neighbors=10),
        'logreg': LogisticRegression(random_state=0, max_iter=1000)
    }
    max_length_tokens = 512 

class TextEncoder(nn.Module):
    def __init__(self, 
                 text_encoder_name,
                 text_encoder_pretrained,
                 text_encoder_trainable,
                 string_sampling,
                ):
        super().__init__()
        if text_encoder_pretrained:
            self.model = AutoModel.from_pretrained(text_encoder_name, output_hidden_states=True)
        else:
            self.model = AutoModel(config=AutoConfig())

        for param in self.model.parameters():
            param.requires_grad = text_encoder_trainable
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.string_sampling = string_sampling
        self.target_token_idx = 0

    def forward(self, string_batch): #input_ids, attention_mask):


        #input_batch = input_batch.cpu().numpy()
        #print("nb of sentences : ", len(text_batch))

        string_batch = list(string_batch)

        trimmed_string_batch = string_batch # most descs have token lenght <512 [string[string.find('DESCRIPTION OF THE RECORD:'):string.find('IMPRESSION:')] for string in string_batch]

        tokenized_text = self.tokenizer(
            trimmed_string_batch, padding=True, truncation=True, max_length=CFG.max_length_tokens
        )

        output = self.model(input_ids=torch.IntTensor(tokenized_text["input_ids"]).to(CFG.device),
                            attention_mask=torch.IntTensor(tokenized_text["attention_mask"]).to(CFG.device))
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class EEGEncoder(nn.Module):
    def __init__(self, 
                 eeg_model_emb_dim,
                 n_chans,
                 eeg_model_pretrained,
                 eeg_model_trainable,
                ):
        super().__init__()

        self.model = Deep4Net(
            in_chans=n_chans,
            n_classes=eeg_model_emb_dim, 
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
        )
        
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
        embedding_dim = input_dim
        projection_dim = output_dim
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
    
class EEGClipModel(pl.LightningModule):
    def __init__(self, 
                 eeg_model_emb_dim=256,
                 text_model_emb_dim=768,
                 projected_emb_dim=64,
                 text_encoder_name="bert-base-uncased",
                 text_encoder_pretrained=True,
                 text_encoder_trainable=False,
                 eeg_model_pretrained=False,
                 eeg_model_trainable=True,
                 string_sampling=False,
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
            string_sampling=string_sampling,
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


        # save features and labels for classification
        self.features_train = []
        self.labels_train = []
        self.features_valid = []
        self.labels_valid = []

    def forward(self, batch):
        eeg_batch, string_batch, id_batch = batch

        #print("CALCULATING EEG FEATURES")
        eeg_features = self.eeg_encoder(eeg_batch)
        eeg_features = torch.mean(eeg_features, dim=2) # Average over output channels       
        text_features = self.text_encoder(string_batch)

        #print("PROJECTING EEG FEATURES")
        eeg_features_proj = self.eeg_projection(eeg_features)
        #print("PROJECTING TEXT FEATURES")
        text_features_proj = self.text_projection(text_features)

        # Extract the labels from the description string
        # TODO : add other labels
        labels = [1 if "abnormal" in string.lower() else 0 for string in string_batch]
        labels = torch.IntTensor(labels).to(CFG.device)

        return eeg_features, eeg_features_proj, text_features, text_features_proj, labels



    def loss_calculation(self, eeg_features_proj, text_features_proj):
        logits = (text_features_proj @ eeg_features_proj.T) / CFG.temperature
        targets = torch.eye(logits.shape[0]).to(CFG.device)
        # shape: (batch_size * batch_size)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        # shape: (batch_size)
        eeg_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # shape: (batch_size) 
        loss =  (eeg_loss + texts_loss) / 2.0 
        # shape: (batch_size)
        loss = loss.mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        eeg_features, eeg_features_proj, text_features, text_features_proj, labels = self.forward(batch)
        self.features_train.append(eeg_features_proj)
        self.labels_train.append(labels)
        loss = self.loss_calculation(eeg_features_proj, text_features_proj)
        self.log('train_loss', loss, prog_bar=True)

        return loss



    def validation_step(self, batch, batch_idx):
        eeg_features, eeg_features_proj, text_features, text_features_proj, labels = self.forward(batch)
        self.features_valid.append(eeg_features_proj)
        self.labels_valid.append(labels)
        loss = self.loss_calculation(eeg_features_proj, text_features_proj)
        self.log('val_loss', loss, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self):

        features_valid = self.features_valid
        labels_valid = self.labels_valid

        features_valid = torch.cat(features_valid).cpu()
        labels_valid = torch.cat(labels_valid).cpu()


        if self.features_train :
            features_train = torch.cat(self.features_train).cpu()
            labels_train = torch.cat(self.labels_train).cpu()

            print("balance in train set : ", torch.sum(labels_train)/labels_train.shape[0])
            print("balance in test set : ", torch.sum(labels_valid)/labels_valid.shape[0])

            # loop through classifiers
            for classifier_name, classifier in CFG.classifiers_dict.items():
                classifier.fit(features_train, labels_train)
                pred_labels = classifier.predict(features_valid)
                accuracy = balanced_accuracy_score(labels_valid.tolist(), pred_labels)
                self.log(classifier_name, accuracy, prog_bar=True)
                print(classifier_name, accuracy)

        self.features_train.clear()
        self.labels_train.clear()

        self.features_valid.clear()
        self.labels_valid.clear()
        
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]