import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, AutoModel, AutoTokenizer
import pytorch_lightning as pl
from medclip import MedCLIPTextModel

class CFG:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = False
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    #image_encoder_lr = 1e-4
    #category_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8

    eeg_embedding_dim = 128 #768 #2048
    nb_categories = 2
    category_embedding_dim = 768
    text_encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
    text_embedding_dim = 768
    text_tokenizer = 'emilyalsentzer/Bio_ClinicalBERT'
    max_length = 512

    pretrained_text_model = True
    trainable_text_model = False
    trainable_eeg_model = True

    temperature = 1.0


    # for projection head; used for both eeg and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

class TextEncoder(nn.Module):
    def __init__(self, recordings_df, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained_text_model , trainable=CFG.trainable_text_model ):
        super().__init__()
        #self.last_n_layer = 4
        proj_dim = 512,
        proj_bias = False
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        self.tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)

        self.trimming = lambda sentence : sentence[sentence.find('DESCRIPTION OF THE RECORD:'):sentence.find('HR:')]

        self.recordings_df = recordings_df
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_batch): #input_ids, attention_mask):

        #print("nb of ids : ", input.cpu().numpy().shape)

        #input_batch = list(input_batch)
        #print(input_batch)

        input_batch = input_batch.cpu().numpy()

        text_batch = [str(self.recordings_df[self.recordings_df.SUBJECT == input].iloc[0]["DESCRIPTION OF THE RECORD"]) for input in input_batch]

        #text_batch = [self.trimming(sentence) for sentence in text_batch]

        #print("nb of sentences : ", len(text_batch))

        tokenized_text = self.tokenizer(
            text_batch, padding=True, truncation=True, max_length=CFG.max_length
        )

        output = self.model(input_ids=torch.IntTensor(tokenized_text["input_ids"]).to(CFG.device),
                            attention_mask=torch.IntTensor(tokenized_text["attention_mask"]).to(CFG.device))
        #last_hidden_state = output.last_hidden_state
        #return last_hidden_state[:, self.target_token_idx, :]
        
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        embed = self.projection_head(embed)
        return embed

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


class MedCLIP(pl.LightningModule):
    def __init__(self, eeg_classifier_model, recordings_df, lr, eeg_embedding_dim=CFG.eeg_embedding_dim, text_embedding_dim = CFG.text_embedding_dim, temperature=CFG.temperature):
        super().__init__()
        self.save_hyperparameters()
        self.eeg_encoder = EEGEncoder(eeg_classifier_model)
        self.text_encoder = TextEncoder(recordings_df, proj_bias=False)
        self.eeg_projection = ProjectionHead(embedding_dim= eeg_embedding_dim)
        self.text_projection = ProjectionHead(embedding_dim = text_embedding_dim)
        self.temperature = temperature
        #self.loss_fn = cross_entropy
        self.eeg_classifier_model = eeg_classifier_model
        self.lr = lr
        self.recordings_df = recordings_df

    def forward(self, eeg, text):
        eeg_features = self.eeg_encoder(eeg)
        text_features = self.text_encoder(text)
        eeg_embeddings = self.projection_head(eeg_features)
        text_embeddings = self.text_projection_head(text_features)
        return eeg_embeddings, text_embeddings, text

    def training_step(self, batch, batch_idx):
        eeg, text = batch
        #eeg, text = self(eeg, text)
        #loss = self.loss_fn(eeg, text, reduction="mean")
        #self.log("train_loss", loss)
        #return loss
        eeg_embeddings, text_embeddings, _ = self.forward(batch)
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

    def validation_step(self, batch, batch_idx):
        eeg, text = batch
        #eeg, text = self(eeg, text)
        #loss = self.loss_fn(eeg, text, reduction="mean")
        #self.log("val_loss", loss)
        #return loss
        eeg_embeddings, text_embeddings, y = self.forward(batch)

        input_batch = y.cpu().numpy()
        label_batch = [self.recordings_df[self.recordings_df.SUBJECT == input].iloc[0]["LABEL"] for input in input_batch]
        label_batch = [1 if label == "normal" else 0 for label in label_batch]
        label_batch = torch.IntTensor(label_batch)

        return eeg_embeddings, label_batch
        

    def test_step(self, batch, batch_idx):
        eeg, text = batch
        eeg, text = self(eeg, text)
        loss = self.loss_fn(eeg, text, reduction="mean")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.eeg_encoder.parameters(), "lr": CFG.head_lr},
                {"params": self.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
                {"params": self.projection_head.parameters(), "lr": CFG.head_lr},
                {"params": self.text_projection_head.parameters(), "lr": CFG.head_lr},
            ],
            weight_decay=CFG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=CFG.patience, factor=CFG.factor)
        return [optimizer], [scheduler]
