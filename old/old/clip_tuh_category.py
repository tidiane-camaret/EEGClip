# %% [markdown]
# A CLIP implementation on medical reports, where the two modalities are :
# - natural language description of the report
# - classification of the report (normal/abormal)
# 
# Inspired by M.Shariatnia implementation : https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing#scrollTo=l9V91XcNi6lW
# 

# %%
import os
import gc
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
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# %%
tuh_data = pd.read_csv('/home/jovyan/EEGClip/data/TUH_Abnormal_EEG_rep.csv') #open the original dataset
tuh_data = tuh_data.drop([0]).dropna(subset=['DESCRIPTION OF THE RECORD']) #drop first line
tuh_data = tuh_data.rename(columns={"DESCRIPTION OF THE RECORD": "DESC"})
tuh_data['CAT'] = tuh_data.LABEL.astype('category').cat.codes

# %%
class CFG:
    debug = False
    #image_path = image_path
    #captions_path = captions_path
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
    #image_embedding = 2048
    nb_categories = 2
    category_embedding = 768
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = False # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

# %%
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# %%
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        """
        """

        self.descs = df.DESC.to_list()
        self.cats = df.CAT.to_list()

        self.encoded_descs = tokenizer(
            self.descs, padding=True, truncation=True, max_length=CFG.max_length
        )

        """
        self.tokenized_descs = tokenizer(
            self.descs, padding=True, truncation=True, max_length=CFG.max_length
        )
        """

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_descs.items()
        }

        item['category'] = self.cats[idx]

        return item


    def __len__(self):
        return len(self.descs)



# %%
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

# %%
class CategoryEncoder(nn.Module):
    def __init__(self, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = nn.Embedding(CFG.nb_categories, CFG.category_embedding)
        else:
            self.model = nn.Embedding.from_pretrained(torch.rand((CFG.nb_categories, CFG.category_embedding)))
            
        for p in self.model.parameters():
            p.requires_grad = trainable


    def forward(self, input):
        output = self.model(input)
        return output

# %%
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

# %%
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        category_embedding=CFG.category_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.category_encoder = CategoryEncoder()
        self.text_encoder = TextEncoder()
        self.category_projection = ProjectionHead(embedding_dim=category_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        category_features = self.category_encoder(batch["category"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        category_embeddings = self.category_projection(category_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ category_embeddings.T) / self.temperature
        categories_similarity = category_embeddings @ category_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (categories_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        categories_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (categories_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean(), category_features, text_features


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

# %%
def build_loaders(dataframe, tokenizer, mode):
    dataset = CLIPDataset(
        df=dataframe,
        tokenizer=tokenizer,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

# %%
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss, _, _ = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["category"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    category_features, text_features, labels = [], [], []
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss, category_features_batch, text_features_batch = model(batch)


        category_features.append(category_features_batch)
        text_features.append(text_features_batch)
        labels.append(batch["category"])

        count = batch["category"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    category_features = torch.cat(category_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    labels = torch.cat(labels, dim=0)

    features = text_features.cpu()
    targets = labels.cpu()
     
    features2d = TSNE(n_components=2).fit_transform(features)
    plt.scatter([a[0] for a in features2d],
        [a[1] for a in features2d],
        c=targets)

    plt.savefig("/home/jovyan/EEGClip/results/clip_graphs/category_tsne_map.png")



    
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, shuffle=True)
    print("balance in train set : ", torch.sum(targets_train)/targets_train.shape[0])
    print("balance in test set : ", torch.sum(targets_test)/targets_test.shape[0])
    
    neigh_classifier = KNeighborsClassifier(n_neighbors=min(targets.shape[0],5))
    neigh_classifier.fit(features_train, targets_train)
    pred_labels_knn = neigh_classifier.predict(features_test)
    knn_accuracy = balanced_accuracy_score(targets_test.tolist(), pred_labels_knn)
    print(knn_accuracy)
    return loss_meter

# %%
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(tuh_data, test_size=0.2)
tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

train_loader = build_loaders(train_df, tokenizer, mode="train")
valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

model = CLIPModel().to(CFG.device)

optimizer = torch.optim.AdamW([
        {"params": model.category_encoder.parameters(), "lr": CFG.category_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.category_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay} 
                               ]
                                , weight_decay=0.)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

step = "epoch"

best_loss = float('inf')

for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, valid_loader)
    
    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(), "best.pt")
        print("Saved Best Model!")
    
    lr_scheduler.step(valid_loss.avg)


