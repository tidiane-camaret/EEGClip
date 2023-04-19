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

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import pytorch_lightning as pl
import wandb 


class CFG:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = False
    num_workers = 2
    weight_decay = 1e-3
    patience = 1
    factor = 0.8

    eeg_embedding_dim =  256#768 #256 #128 #768 #2048
    nb_categories = 2
    category_embedding_dim = 768
    text_encoder_model = "bert-base-uncased" #"AshtonIsNotHere/GatorTron-OG"#"distilbert-base-uncased" #"AshtonIsNotHere/GatorTron-OG" #"bert-base-uncased" #"microsoft/biogpt" #"microsoft/BioGPT-Large-PubMedQA" #"AshtonIsNotHere/GatorTron-OG" # "" " "distilbert-base-uncased" #"emilyalsentzer/Bio_ClinicalBERT"
    text_embedding_dim = 768 #
    text_tokenizer = text_encoder_model #"distilbert-base-uncased"
    max_length = 512

    pretrained_text_model = True
    trainable_text_model = False
    trainable_eeg_model = True

    temperature = 1.0


    # for projection head; used for both eeg and text encoders
    num_projection_layers = 1
    projection_dim = 64 
    dropout = 0.1


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained_text_model , trainable=CFG.trainable_text_model ):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoConfig.from_pretrained(model_name)
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)

    def forward(self, string_batch): #input_ids, attention_mask):


        #input_batch = input_batch.cpu().numpy()
        #print("nb of sentences : ", len(text_batch))

        string_batch = list(string_batch)

        trimmed_string_batch = string_batch # most descs have token lenght <512 [string[string.find('DESCRIPTION OF THE RECORD:'):string.find('IMPRESSION:')] for string in string_batch]

        tokenized_text = self.tokenizer(
            trimmed_string_batch, padding=True, truncation=True, max_length=CFG.max_length
        )

        output = self.model(input_ids=torch.IntTensor(tokenized_text["input_ids"]).to(CFG.device),
                            attention_mask=torch.IntTensor(tokenized_text["attention_mask"]).to(CFG.device))
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class CategoryEncoder(nn.Module):
    def __init__(self, pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = nn.Embedding(CFG.nb_categories, CFG.category_embedding_dim)
        else:
            self.model = nn.Embedding.from_pretrained(torch.rand((CFG.nb_categories, CFG.category_embedding_dim)))
            
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input):
        output = self.model(input)
        return output

class EEGEncoder(nn.Module):
    def __init__(self, n_chans = 21, trainable=CFG.trainable_eeg_model):
        super().__init__()
        # TODO: add pretrained models
        self.model = Deep4Net(
            in_chans=n_chans,
            n_classes=CFG.eeg_embedding_dim, 
            input_window_samples=None,
            final_conv_length=2,
            stride_before_pool=True,
        )
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
        lr,
        weight_decay,
        n_chans=21,
        temperature=CFG.temperature,
        eeg_embedding_dim=CFG.eeg_embedding_dim,
        text_embedding_dim=CFG.text_embedding_dim,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_chans)
        self.text_encoder = TextEncoder()
        self.category_encoder = CategoryEncoder()
        self.eeg_projection = ProjectionHead(embedding_dim=eeg_embedding_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim)
        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_features = []
        self.train_labels = []

        self.valid_features = []
        self.valid_labels = []

        self.save_hyperparameters()


    def forward(self, batch):
        eeg_batch, string_batch, id_batch = batch

        #print("CALCULATING EEG FEATURES")
        eeg_features = self.eeg_encoder(eeg_batch)
        #print(eeg_features.shape)
        eeg_features = torch.mean(eeg_features, dim=2)

        #eeg_features = self.category_encoder(labels)
        #print("CALCULATING TEXT FEATURES")
        #text_features = self.category_encoder(labels)
        
        text_features = self.text_encoder(string_batch)

        #print("PROJECTING EEG FEATURES")
        eeg_features_proj = self.eeg_projection(eeg_features)
        #print("PROJECTING TEXT FEATURES")
        text_features_proj = self.text_projection(text_features)

        # Extract the labels from the description string

        trimmed_string_batch = [string[string.find('IMPRESSION:'):string.find('CLINICAL CORRELATION:')] for string in string_batch]
        #print(string_batch)
        labels = [1 if "abnormal" in string.lower() else 0 for string in trimmed_string_batch]
        labels = torch.IntTensor(labels).to(CFG.device)

        return eeg_features, eeg_features_proj, text_features, text_features_proj, labels


    def loss_calculation(self, eeg_features_proj, text_features_proj):
        

        logits = (text_features_proj @ eeg_features_proj.T) / self.temperature
        
        # shape: (batch_size * batch_size)
        
        """
        eeg_similarity = eeg_features_proj @ eeg_features_proj.T
        # shape: (batch_size * batch_size)
        texts_similarity = text_features_proj @ text_features_proj.T
        # shape: (batch_size * batch_size)
        targets = F.softmax(
            (eeg_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        """
        targets = torch.eye(logits.shape[0]).to(CFG.device)

        # shape: (batch_size * batch_size)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        # shape: (batch_size)
        eeg_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # shape: (batch_size) 
        loss =  (eeg_loss + texts_loss) / 2.0 
        # shape: (batch_size)
        loss = loss.mean()

        #log the logit matrix
        #logits_image = wandb.Image(logits, caption="logit matrix")
        #targets_image = wandb.Image(targets, caption="targets matrix")

        #wandb.log({"logits image": logits_image,
        #           "targets image": targets_image})

        """

        eeg_embeddings = eeg_features_proj
        text_embeddings = text_features_proj
        print(eeg_embeddings.shape, text_embeddings.shape)
        batch_size = eeg_embeddings.shape[0]
        
        # concatenate EEG and text embeddings along the batch dimension
        embeddings = torch.cat([eeg_embeddings, text_embeddings], dim=0)
        
        # calculate cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        # exclude self-similarity and normalize temperature
        mask = torch.eye(batch_size * 2, dtype=torch.bool)
        similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
        similarity_matrix /= self.temperature
        
        # calculate contrastive loss
        targets = torch.arange(batch_size * 2, dtype=torch.long, device=similarity_matrix.device)
        print(similarity_matrix.shape, targets.shape)
        
        loss = nn.CrossEntropyLoss()(similarity_matrix, targets)
        loss = loss.mean()
        """

        return loss


    def training_step(self, batch, batch_idx):
        eeg_features, eeg_features_proj, text_features, text_features_proj, labels = self.forward(batch)
        
        self.train_features.append(eeg_features_proj)
        self.train_labels.append(labels)
        #+("CALCULATING LOSS")

        loss = self.loss_calculation(eeg_features_proj, text_features_proj)

        self.log('train_loss', loss, prog_bar=True)

        return loss



    def validation_step(self, batch, batch_idx):

        eeg_features, eeg_features_proj, text_features, text_features_proj, labels = self.forward(batch)
        self.valid_features.append(eeg_features_proj)
        self.valid_labels.append(labels)

        loss =  self.loss_calculation(eeg_features_proj, text_features_proj)

        self.log('val_loss', loss, prog_bar=True)

        return loss

    
    def validation_epoch_end(self, outputs):

        features_valid = self.valid_features
        targets_valid = self.valid_labels

        features_valid = torch.cat(features_valid).cpu()
        targets_valid = torch.cat(targets_valid).cpu()
        
        """
        features2d = TSNE(n_components=2).fit_transform(features_valid)
        plt.scatter([a[0] for a in features2d],
            [a[1] for a in features2d],
            c=targets_valid)

        plt.savefig("/home/jovyan/EEGClip/results/clip_graphs/tsne_map.png")
        
        self.logger.experiment.log({
            "2d projection of eeg embeddings": wandb.Image("/home/jovyan/EEGClip/results/clip_graphs/tsne_map.png") 
        })
        """


        if self.train_features :
            features_train = self.train_features
            targets_train = self.train_labels

            features_train = torch.cat(features_train).cpu()
            targets_train = torch.cat(targets_train).cpu()

            print("balance in train set : ", torch.sum(targets_train)/targets_train.shape[0])
            print("balance in test set : ", torch.sum(targets_valid)/targets_valid.shape[0])
            
            neigh_classifier = KNeighborsClassifier(n_neighbors=min(targets_train.shape[0],5))
            neigh_classifier.fit(features_train, targets_train)
            pred_labels_knn = neigh_classifier.predict(features_valid)
            knn_accuracy = balanced_accuracy_score(targets_valid.tolist(), pred_labels_knn)
            self.log('knn_acc', knn_accuracy, prog_bar=True)

            logreg_classifier = LogisticRegression(random_state=0, max_iter=1000, verbose=0)
            logreg_classifier.fit(features_train, targets_train)
            pred_labels_logreg = logreg_classifier.predict(features_valid)
            logreg_accuracy = balanced_accuracy_score(targets_valid.tolist(), pred_labels_logreg)
            self.log('logreg_acc', logreg_accuracy, prog_bar=True)

        self.train_features.clear()
        self.train_labels.clear()

        self.valid_features.clear()
        self.valid_labels.clear()
        
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]

class EEGClipClassifierModule(pl.LightningModule):
    """
    This module uses the encoders from the EEGClipModule to perform classification
    """
    def __init__(self, 
                lr,
                weight_decay,):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.eeg_clip_module = EEGClipModule(lr, weight_decay) #
        self.eeg_clip_module = EEGClipModule.load_from_checkpoint("/home/jovyan/results/models/eegclipmodel.ckpt",lr=lr, weight_decay=weight_decay)
        self.eeg_clip_module.freeze()

        #self.classifier = nn.Linear(CFG.projection_dim, 2)
        self.classifier = nn.Sequential(
            nn.Linear(CFG.projection_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, batch):

        eeg, labels, id_batch = batch
        eeg_features = self.eeg_clip_module.eeg_encoder(eeg)
        eeg_features = torch.mean(eeg_features, dim=2) #TODO : think about how to pool the features. Simple mean ? 
        eeg_features_proj = self.eeg_clip_module.eeg_projection(eeg_features)
        logits = self.classifier(eeg_features_proj)

        #labels = labels.long()  #pathological
        labels = [0 if l=="M" else 1 for l in labels]

        labels = torch.IntTensor(labels).to(CFG.device)
        return logits, labels
    
    def training_step(self, batch, batch_idx):
        logits, labels = self.forward(batch)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss, prog_bar=True)

        accuracy = balanced_accuracy_score(labels.tolist(), torch.argmax(logits, dim=1).tolist())
        self.log('train_acc', accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, labels = self.forward(batch)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        
        accuracy = balanced_accuracy_score(labels.tolist(), torch.argmax(logits, dim=1).tolist())
        self.log('val_acc', accuracy, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]
    
    
