import os
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pytorch_lightning as pl

class CFG:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recordings_df = pd.read_csv('/home/jovyan/EEGClip/data/TUH_Abnormal_EEG_rep.csv')


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

    eeg_embedding_dim = 128 #768 #2048
    nb_categories = 2
    category_embedding_dim = 768
    text_encoder_model = "distilbert-base-uncased"
    text_embedding_dim = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained_text_model = True
    trainable_text_model = False
    trainable_eeg_model = True

    temperature = 1.0


    # for projection head; used for both eeg and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained_text_model , trainable=CFG.trainable_text_model ):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        self.tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

        self.trimming = lambda sentence : sentence[sentence.find('DESCRIPTION OF THE RECORD:'):sentence.find('HR:')] 

    def forward(self, input_batch): #input_ids, attention_mask):

        #print("nb of ids : ", input.cpu().numpy().shape)

        input_batch = input_batch.cpu().numpy()

        text_batch = [CFG.recordings_df[CFG.recordings_df.SUBJECT == input].iloc[0]["DESCRIPTION OF THE RECORD"] for input in input_batch]

        #text_batch = [self.trimming(sentence) for sentence in text_batch]

        #print("nb of sentences : ", len(text_batch))

        tokenized_text = self.tokenizer(
            text_batch, padding=True, truncation=True, max_length=CFG.max_length
        )

        output = self.model(input_ids=torch.IntTensor(tokenized_text["input_ids"]).to(CFG.device),
                            attention_mask=torch.IntTensor(tokenized_text["attention_mask"]).to(CFG.device))
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

"""
class CategoryEncoder(nn.Module):
    def __init__(self, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = nn.Embedding(CFG.nb_categories, CFG.category_embedding)
        else:
            self.model = nn.Embedding.from_pretrained(torch.rand((CFG.nb_categories, CFG.category_embedding)))
            
        for p in self.model.parameters():
            p.requires_grad = trainable
"""
class EEGEncoder(nn.Module):
    def __init__(self, eeg_classifier_model, trainable=CFG.trainable_eeg_model):
        super().__init__()
        # TODO: add pretrained models
        self.model = eeg_classifier_model
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
        eeg_classifier_model, 
        lr,
        temperature=CFG.temperature,
        eeg_embedding_dim=CFG.eeg_embedding_dim,
        text_embedding_dim=CFG.text_embedding_dim,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_classifier_model)
        self.text_encoder = TextEncoder()
        self.eeg_projection = ProjectionHead(embedding_dim=eeg_embedding_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim)
        self.temperature = temperature
        self.eeg_classifier_model = eeg_classifier_model
        self.lr = lr


    def forward(self, batch):
        x, y, z = batch
        #print("CALCULATING EEG FEATURES")
        eeg_features = self.eeg_encoder(x)
        #print("CALCULATING TEXT FEATURES")
        text_features = self.text_encoder(y)
        #print("PROJECTING EEG FEATURES")
        eeg_embeddings = self.eeg_projection(eeg_features)
        #print("PROJECTING TEXT FEATURES")
        text_embeddings = self.text_projection(text_features)

        return eeg_embeddings, text_embeddings, y

    def training_step(self, batch, batch_idx):
        eeg_embeddings, text_embeddings, _ = self.forward(batch)
        #print("CALCULATING LOSS")

        logits = (text_embeddings @ eeg_embeddings.T) / self.temperature
        eeg_similarity = eeg_embeddings @ eeg_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (eeg_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        eeg_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (eeg_loss + texts_loss) / 2.0 # shape: (batch_size)
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]


    def validation_step(self, batch, batch_idx):
        eeg_embeddings, text_embeddings, y = self.forward(batch)

        input_batch = y.cpu().numpy()
        label_batch = [CFG.recordings_df[CFG.recordings_df.SUBJECT == input].iloc[0]["LABEL"] for input in input_batch]
        label_batch = [1 if label == "normal" else 0 for label in label_batch]
        label_batch = torch.IntTensor(label_batch)

        return text_embeddings, label_batch

    
    def validation_epoch_end(self, outputs):
            # get rid of last batch if it is smaller than batch_size
        if len(outputs) > 1:
            outputs = outputs[:-1]
        features, targets = zip(*outputs)
        features = torch.stack(features)
        targets = torch.stack(targets)
        features = features.reshape(-1, features.shape[-1])
        targets = targets.reshape(-1)
        features_train, features_test, targets_train, targets_test = train_test_split(features, targets, shuffle=True)
        print(torch.sum(targets_train)/targets_train.shape[0],torch.sum(targets_test)/targets_test.shape[0])
        
        neigh = KNeighborsClassifier(n_neighbors=min(targets.shape[0],20))
        neigh.fit(features_train.cpu(), targets_train.cpu())

        pred_labels_knn = neigh.predict(features_test.cpu())

        accuracy = (pred_labels_knn == targets_test.cpu().tolist()).mean().item()# / features.size(0)
        self.log('knn_acc', accuracy, prog_bar=True)
        return accuracy
        """

        logits = (text_embeddings @ eeg_embeddings.T) / self.temperature
        categories_similarity = eeg_embeddings @ eeg_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (categories_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        categories_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (categories_loss + texts_loss) / 2.0 # shape: (batch_size)
        loss = loss.mean()
        self.log('validation_loss', loss, on_epoch = True, prog_bar=True)
        return loss
        """





    """
    def training_epoch_end(self, outputs):
        # print losses
        losses = [i['loss'].item() for i in outputs]
        loss_avg = sum(losses)/len(losses)
        print(f'train loss = {loss_avg:.2f}')

        # update feature bank at the end of each training epoch
        self.eeg_encoder.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for batch in self.val_dataloader():
                eeg_embeddings, text_embeddings, y = self.forward(batch)
                input_batch = input_batch.cpu().numpy()
                label = [CFG.recordings_df[CFG.recordings_df.SUBJECT == input].iloc[0]["LABEL"] for input in input_batch]
                self.feature_bank.append(text_embeddings)
                self.targets_bank.append(label)
        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.eeg_encoder.train()
    """