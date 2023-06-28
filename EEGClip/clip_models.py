import random
import numpy as np
import pandas as pd
import re
import copy
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import torch
from torch import nn
import torch.nn.functional as F

from braindecode.models import Deep4Net
from braindecode.training.scoring import trial_preds_from_window_preds
from braindecode.models.util import to_dense_prediction_model

#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

import pytorch_lightning as pl



class CFG:
    """Configuration class for the EEGClip model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8
    text_model_emb_dim = 768
    projected_emb_dim = 256
    temperature = 1.0
    classifiers_dict = {
        #'knn': KNeighborsClassifier(n_neighbors=10),
        'logreg': LogisticRegression(random_state=0, max_iter=1000)
    }
    max_length_tokens = 512 

class TextEncoder(nn.Module):
    def __init__(self, 
                 text_encoder_name,
                 text_encoder_pretrained,
                 text_encoder_trainable,
                 string_sampling,
                 lookup_strings = True #use previously computed embeddings
                ):
        super().__init__()
        self.string_sampling = string_sampling
        self.lookup_strings = lookup_strings
        """
        if text_encoder_pretrained:
            self.model = AutoModel.from_pretrained(text_encoder_name, output_hidden_states=True)
        else:
            self.model = AutoModel(config=AutoConfig())
        
        #self.model = SentenceTransformer("hkunlp/instructor-xl")
        for param in self.model.parameters():
            param.requires_grad = text_encoder_trainable
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
        self.target_token_idx = 0
        """
        #TODO : add a config file for the path
        report_df_path = '/home/jovyan/EEGClip/scripts/text_analysis/report_df_embs.csv'
        report_df = pd.read_csv(report_df_path)
        embs_name = "embs_instructor"
        for r in range(len(report_df)):
            re = copy.copy(report_df[embs_name][r])
            # convert the string to array
            re = re.replace('[', '')
            re = re.replace(']', '')
            re = re.replace(',', '')
            re = re.split()
            re = [float(i) for i in re]
            report_df[embs_name][r] = re

        self.report_df = report_df


    def forward(self, string_batch):
        string_batch = list(string_batch)

        if self.string_sampling:
            for i, string in enumerate(string_batch):
                # look for the positions of \n occurences
                newlines = [m.start() for m in re.finditer('\n', string)]
                newlines.extend([0, len(string)])
                # sample a random position
                start, end = 0,0
                while end <= start:
                    start = random.choice(newlines)
                    end = random.choice(newlines)
                # sample a random substring
                string_batch[i] = string[start:end]

        if self.lookup_strings :
            embs = []
            for s in string_batch:
                lookup = self.report_df.loc[self.report_df['report'] == s, 'embs_instructor']
                
                emb = lookup.item()
                embs.append(emb)
            embs = torch.Tensor(embs).to(CFG.device)
                
        else:
            input_ids = self.tokenizer(string_batch, 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors="pt",
                                    max_length=CFG.max_length_tokens
                                    ).input_ids.to(CFG.device)
            outputs = self.model(input_ids)
            
            embs = outputs.last_hidden_state[:,0,:]
        


        
        return embs

    
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
        to_dense_prediction_model(self.model)
        if eeg_model_pretrained:
            self.model.load_state_dict(torch.load('deep4net_trained.pt'))
            
        for param in self.model.parameters():
            param.requires_grad = eeg_model_trainable

    def forward(self, x):
        eeg_features = self.model(x) 
        return eeg_features.transpose(1,2) #[B,N_pred,Enc_size]. Allows for projection afterwards
    
class ProjectionHead(nn.Module):
    def __init__(self, 
                 input_dim=128,
                 output_dim=128,
                 dropout=0.1,
                 num_fc_layers=2,
                 transpose=False,
                ):
        super().__init__()
        """ # TODO : this config sucks. it shouldnt, see why
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

        """
        ## Do the same, but with variable number of layers
        self.projection_layer = nn.Linear(input_dim, output_dim)
        self.fc_layers = nn.ModuleList()

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(output_dim, output_dim))
        self.layer_norm = nn.LayerNorm(output_dim) # TODO : what is the benefit of layer norm here?
        self.transpose = transpose
    def forward(self, x):
        x_proj = self.projection_layer(x)
        for layer in self.fc_layers:
            x_proj_fc = self.dropout(self.gelu(layer(x_proj)))
            x_proj = x_proj + x_proj_fc
            x_proj = self.layer_norm(x_proj)

        if self.transpose:
            x_proj = x_proj.transpose(1,2)#[B,Enc_size,N_pred]
        return x_proj




def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

        
class EEGClipModel(pl.LightningModule):
    def __init__(self, 
                 eeg_model_emb_dim=128,
                 text_model_emb_dim=768,
                 projected_emb_dim=64,
                 text_encoder_name="bert-base-uncased",
                 text_encoder_pretrained=True,
                 text_encoder_trainable=False,
                 eeg_model_pretrained=False,
                 eeg_model_trainable=True,
                 string_sampling=False,
                 dropout=0.1,
                 num_fc_layers=1,
                 lr=1e-3,
                 weight_decay=1e-6,
                 n_chans=21,
                 **kwargs
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
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
            num_fc_layers=num_fc_layers,
        )
        
        self.eeg_projection = ProjectionHead(
            input_dim=eeg_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout=dropout,
            num_fc_layers=num_fc_layers,
            transpose = True
        )


        # save features and labels for classification
        self.features_train = []
        self.labels_train = []
        self.ids_train = []
        self.features_valid = []
        self.labels_valid = []
        self.ids_valid = []

        self.report_list = []

    def forward(self, batch):
        eeg_batch, string_batch, id_batch = batch
        self.report_list.extend(list(string_batch))
        #print("CALCULATING EEG FEATURES")
        eeg_features = self.eeg_encoder(eeg_batch)     
        text_features = self.text_encoder(string_batch)

        #print("PROJECTING EEG FEATURES")
        eeg_features_proj = self.eeg_projection(eeg_features)
        eeg_features_proj = torch.mean(eeg_features_proj, dim=2)
        #print("PROJECTING TEXT FEATURES")
        text_features_proj = self.text_projection(text_features)

        # Extract the labels from the description string
        # TODO : add other labels
        string_batch = [string[string.find('IMPRESSION:'):string.find('CLINICAL CORRELATION:')] for string in string_batch]
        labels = [1 if "abnormal" in string.lower() else 0 for string in string_batch]
        #labels = [0 if "seizure" not in l.lower() or "no seizure" in l.lower() else 1 for l in string_batch]
        labels = torch.IntTensor(labels).to(CFG.device)

        #print("SHAPE OF ID BATCH: ", id_batch[0].shape)
        ids = id_batch[0]

        return eeg_features, eeg_features_proj, text_features, text_features_proj, labels, ids


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
        eeg_features, eeg_features_proj, text_features, text_features_proj, labels, ids = self.forward(batch)
        self.features_train.append(eeg_features_proj)
        self.labels_train.append(labels)
        self.ids_train.append(ids)
        loss = self.loss_calculation(eeg_features_proj, text_features_proj)
        self.log('train_loss', loss, prog_bar=True)

        return loss



    def validation_step(self, batch, batch_idx):
        eeg_features, eeg_features_proj, text_features, text_features_proj, labels, ids = self.forward(batch)
        self.features_valid.append(eeg_features_proj)
        self.labels_valid.append(labels)
        self.ids_valid.append(ids)
        loss = self.loss_calculation(eeg_features_proj, text_features_proj)
        self.log('val_loss', loss, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self):
        #report_list = list(set(self.report_list))
        #with open('parrot.pkl', 'wb') as f:
        #   pickle.dump(report_list, f)

        features_valid = torch.cat(self.features_valid).cpu()
        ids_valid = torch.cat(self.ids_valid).cpu()
        labels_valid = torch.cat(self.labels_valid).cpu()

        equal_to_extracted = (labels_valid == torch.cat(self.labels_valid).cpu())
        print("proportion of correctly extracted labels (train): ", torch.sum(equal_to_extracted)/equal_to_extracted.shape[0])



        if self.features_train :
            features_train = torch.cat(self.features_train).cpu()
            ids_train = torch.cat(self.ids_train).cpu()
            labels_train = torch.cat(self.labels_train).cpu()
            equal_to_extracted = (labels_train == torch.cat(self.labels_train).cpu())
            print("proportion of correctly extracted labels (valid): ", torch.sum(equal_to_extracted)/equal_to_extracted.shape[0])

            print("balance in train set : ", torch.sum(labels_train)/labels_train.shape[0])
            print("balance in valid set : ", torch.sum(labels_valid)/labels_valid.shape[0])

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

        self.ids_train.clear()
        self.ids_valid.clear()
        
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]

"""
def on_save_checkpoint(checkpoint):
    for key in list(checkpoint['state_dict'].keys()):
        if "text_encoder" in key:
            print('deleting ', key)
            del checkpoint['state_dict'][key]
    # pop the backbone here using custom logic
    del checkpoint['state_dict'][backbone_keys]

"""