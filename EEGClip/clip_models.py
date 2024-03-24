import copy
import random
import re
import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from braindecode.models import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from torch import nn

# from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer

import configs.preprocess_config as preprocess_config
from EEGClip.loss_methods import ClipLoss, SigLipLoss

medication_list = ["keppra", "dilantin", "depakote"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifiers_dict = {
    #'knn': KNeighborsClassifier(n_neighbors=10),
    "logreg": LogisticRegression(random_state=0, max_iter=1000)
}


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        text_encoder_pretrained,
        text_encoder_trainable,
        string_sampling=False,
        lookup_strings=True,  # use previously computed embeddings
        max_token_len=512,
    ):
        super().__init__()
        self.string_sampling = string_sampling
        self.lookup_strings = lookup_strings
        self.text_encoder_name = text_encoder_name
        self.max_token_len = max_token_len
        with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
            self.zc_sentences_emb_dict = json.load(f)

        if self.lookup_strings:
            embs_df = pd.read_csv(preprocess_config.embs_df_path)
            embs_name = text_encoder_name
            for r in range(len(embs_df)):
                re = copy.copy(embs_df[embs_name][r])
                # convert the string to array
                re = re.replace("[", "")
                re = re.replace("]", "")
                re = re.replace(",", "")
                re = re.split()
                re = [float(i) for i in re]
                embs_df[embs_name][r] = re

            self.embs_df = embs_df
        else:
            if text_encoder_pretrained:
                self.model = AutoModel.from_pretrained(
                    text_encoder_name, output_hidden_states=True
                )
            else:
                self.model = AutoModel(config=AutoConfig())

            # self.model = SentenceTransformer("hkunlp/instructor-xl")
            print("trainable text encoder : ", text_encoder_trainable)
            for param in self.model.parameters():
                param.requires_grad = text_encoder_trainable

            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

    def forward(self, string_batch):
        string_batch = list(string_batch)

        if self.string_sampling:  # randomly sample strings from the report
            for i, string in enumerate(string_batch):
                # look for the positions of \n occurences
                newlines = [m.start() for m in re.finditer(",", string)]
                newlines.extend([0, len(string)])
                # sample a random position
                start, end = 0, 0
                while end <= start:
                    start = random.choice(newlines)
                    end = random.choice(newlines)
                # sample a random substring
                string_batch[i] = string[start:end]

        if self.lookup_strings:  # lookup precomputed embeddings (faster training)
            embs = []
            for s in string_batch:
                lookup = self.embs_df.loc[
                    self.embs_df["report"] == s, self.text_encoder_name
                ]

                emb = lookup.tolist()[0]
                embs.append(emb)
            embs = torch.Tensor(embs).to(device)

        else:
            # print(string_batch)
            # string_batch = [s.partition("pijule")[2] for s in string_batch]
            # print(string_batch)
            input_ids = self.tokenizer(
                string_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_token_len,
            ).input_ids.to(device)
            outputs = self.model(input_ids)

            embs = outputs.last_hidden_state[:, 0, :]

        return embs


class EEGEncoder(nn.Module):
    def __init__(
        self,
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
            self.model.load_state_dict(torch.load("deep4net_trained.pt"))

        for param in self.model.parameters():
            param.requires_grad = eeg_model_trainable

    def forward(self, x):
        eeg_features = self.model(x)
        return eeg_features.transpose(
            1, 2
        )  # [B,N_pred,Enc_size]. Allows for projection afterwards


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim=128,
        output_dim=128,
        dropout_rate=0.1,
        num_fc_layers=2,
        transpose=False,
    ):
        super().__init__()
        """
        self.projection_layer = nn.Linear(input_dim, output_dim)
        self.fc_layers = nn.ModuleList()

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(output_dim, output_dim))
        self.layer_norm = nn.LayerNorm(
            output_dim
        )  # TODO : what is the benefit of layer norm here?
        self.transpose = transpose

    def forward(self, x):
        x_proj = self.projection_layer(x)
        for layer in self.fc_layers:
            x_proj_fc = self.dropout(self.gelu(layer(x_proj)))
            x_proj = x_proj + x_proj_fc
            x_proj = self.layer_norm(x_proj)

        if self.transpose:
            x_proj = x_proj.transpose(1, 2)  # [B,Enc_size,N_pred]
        return x_proj
        """
        super(ProjectionHead, self).__init__()
        self.num_layers = num_fc_layers
        self.transpose = transpose
        layers = nn.ModuleList()

        # Input projection layer
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_fc_layers - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output projection layer
        layers.append(nn.Linear(output_dim, output_dim))

        self.layers = layers
        self.projection = nn.Sequential(*layers)
    def forward(self, x):
        for layer in self.layers:
            # check if layer is batchnorm
            if isinstance(layer, nn.BatchNorm1d) and self.transpose:
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
            
        return x
        #return self.projection(x)


class EEGClipModel(pl.LightningModule):
    def __init__(
        self,
        eeg_model_emb_dim=128,
        text_encoder_emb_dim=1024,
        projected_emb_dim=64,
        text_encoder_name="medicalai/ClinicalBERT",
        text_encoder_pretrained=True,
        text_encoder_trainable=True,
        eeg_model_pretrained=False,
        eeg_model_trainable=True,
        string_sampling=False,
        dropout_rate=0.1,
        num_fc_layers=1,
        lr=1e-3,
        lr_frac_lm=0,
        weight_decay=1e-6,
        n_chans=21,
        contrastive_loss_temperature=1,
        contrastive_loss_func="clip",
        text_encoder_max_token_len=512,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_lm = self.lr * lr_frac_lm
        self.weight_decay = weight_decay
        self.n_chans = n_chans
        self.contrastive_loss_temperature = contrastive_loss_temperature
        self.contrastive_loss_func = contrastive_loss_func
        self.text_encoder_emb_dim = text_encoder_emb_dim
        print(self.text_encoder_emb_dim)

        self.text_encoder = TextEncoder(
            text_encoder_name=text_encoder_name,
            text_encoder_pretrained=text_encoder_pretrained,
            text_encoder_trainable=text_encoder_trainable,
            string_sampling=string_sampling,
            max_token_len=text_encoder_max_token_len,
        )

        self.eeg_encoder = EEGEncoder(
            eeg_model_emb_dim=eeg_model_emb_dim,
            n_chans=n_chans,
            eeg_model_pretrained=eeg_model_pretrained,
            eeg_model_trainable=eeg_model_trainable,
        )

        self.text_projection = ProjectionHead(
            input_dim=self.text_encoder_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
        )

        self.eeg_projection = ProjectionHead(
            input_dim=eeg_model_emb_dim,
            output_dim=projected_emb_dim,
            dropout_rate=dropout_rate,
            num_fc_layers=num_fc_layers,
            transpose=True,
        )

        if contrastive_loss_func == "clip":
            self.loss_fn = ClipLoss()
            init_logit_scale = np.log(1 / 0.07)
            init_logit_bias = 0
        elif contrastive_loss_func == "siglip":
            self.loss_fn = SigLipLoss()
            init_logit_scale = np.log(10)
            init_logit_bias = -10

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

        # save features and labels for classification
        self.features_train = []
        self.labels_train = []

        self.features_valid = []
        self.labels_valid = []

        self.report_list = []

    def forward(self, batch):
        eeg_batch, string_batch, id_batch = batch
        self.report_list.extend(list(string_batch))
        # print("CALCULATING EEG FEATURES")
        eeg_features = self.eeg_encoder(eeg_batch)
        text_features = self.text_encoder(string_batch)

        # print("PROJECTING EEG FEATURES")
        eeg_features_proj = self.eeg_projection(eeg_features)
        eeg_features_proj = torch.mean(eeg_features_proj, dim=1) # average over the time dimension
        # print("PROJECTING TEXT FEATURES")
        text_features_proj = self.text_projection(text_features)

        # Extract the labels from the description string
        # TODO : add other labels

        labels_pathological = [
            (
                1
                if "true" in re.search(r"pathological: (\w+)", string).group(1).lower()
                else 0
            )
            for string in string_batch
        ]
        labels_gender = [
            1 if "m" in re.search(r"gender: (\w+)", string).group(1).lower() else 0
            for string in string_batch
        ]
        labels_under_50 = [
            1 if int(re.search(r"age: (\d+)", string).group(1)) < 50 else 0
            for string in string_batch
        ]
        labels_med = [
            1 if any([med in string.lower() for med in medication_list]) else 0
            for string in string_batch
        ]

        # stack the labels into an int tensor

        labels = torch.stack(
            [
                torch.IntTensor(labels_pathological),
                torch.IntTensor(labels_gender),
                torch.IntTensor(labels_under_50),
                torch.IntTensor(labels_med),
            ],
            dim=1,
        ).to(device)

        return (
            eeg_features,
            eeg_features_proj,
            text_features,
            text_features_proj,
            labels,
        )

    def training_step(self, batch, batch_idx):
        (
            eeg_features,
            eeg_features_proj,
            text_features,
            text_features_proj,
            labels,
        ) = self.forward(batch)
        self.features_train.append(eeg_features_proj)
        self.labels_train.append(labels)

        # loss = self.loss_calculation(eeg_features_proj, text_features_proj,self.contrastive_loss_temperature)
        loss = self.loss_fn(
            eeg_features_proj, text_features_proj, self.logit_scale, self.logit_bias
        )
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (
            eeg_features,
            eeg_features_proj,
            text_features,
            text_features_proj,
            labels,
        ) = self.forward(batch)
        self.features_valid.append(eeg_features_proj)
        self.labels_valid.append(labels)

        loss = self.loss_fn(
            eeg_features_proj, text_features_proj, self.logit_scale, self.logit_bias
        )
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # report_list = list(set(self.report_list))
        # with open('parrot.pkl', 'wb') as f:
        #   pickle.dump(report_list, f)

        features_valid = torch.cat(self.features_valid).cpu()

        labels_valid = torch.cat(self.labels_valid).cpu()

        equal_to_extracted = labels_valid == torch.cat(self.labels_valid).cpu()
        print(
            "proportion of correctly extracted labels (train): ",
            torch.sum(equal_to_extracted) / equal_to_extracted.shape[0],
        )

        if self.features_train:
            features_train = torch.cat(self.features_train).cpu()

            labels_train = torch.cat(self.labels_train).cpu()

            print(
                "balance in train set : ",
                torch.sum(labels_train) / labels_train.shape[0],
            )
            print(
                "balance in valid set : ",
                torch.sum(labels_valid) / labels_valid.shape[0],
            )

            # loop through classifiers
            for classifier_name, classifier in classifiers_dict.items():
                for label_idx, label_name in enumerate(
                    ["pathological", "gender", "under_50", "medication"]
                ):
                    # classification
                    classifier.fit(features_train, labels_train[:, label_idx])
                    preds = classifier.predict(features_valid)
                    balanced_acc = balanced_accuracy_score(
                        labels_valid[:, label_idx], preds
                    )
                    self.log(
                        f"val_acc_{classifier_name}_{label_name}",
                        balanced_acc,
                        prog_bar=True,
                    )
                    # zero shot classification
                    zero_shot_preds = []
                    emb_dict = self.text_encoder.zc_sentences_emb_dict[self.text_encoder.text_encoder_name][label_name]
                    s0, s1 = (
                        emb_dict["s0"],
                        emb_dict["s1"],
                    )
                    s0, s1 = torch.Tensor(s0).to(device), torch.Tensor(s1).to(device)
                    # add batch dimension to s0 and s1
                    s0, s1 = s0.unsqueeze(0), s1.unsqueeze(0)
                    with torch.no_grad():
                        s0, s1 = self.text_projection(s0), self.text_projection(s1)
                    s0, s1 = s0.cpu().detach().numpy(), s1.cpu().detach().numpy()
                    s0, s1 = s0[0], s1[0]
                    # (s0,s1) as a numpy array
                    s = np.array([s0, s1])
                    logits_per_eeg = features_valid @ s.T
                    zero_shot_preds = torch.argmax(logits_per_eeg, dim=1)
                    """
                    for f in features_valid:
                        d0 = distance.cosine(f, s0)
                        d1 = distance.cosine(f, s1)
                        pred_valid = 0 if d0 < d1 else 1
                        zero_shot_preds.append(pred_valid)
                    """
                    balanced_acc = balanced_accuracy_score(
                        labels_valid[:, label_idx], zero_shot_preds
                    )
                    self.log(
                        f"val_acc_zs_{classifier_name}_{label_name}",
                        balanced_acc,
                        prog_bar=True,
                    )

        self.features_train.clear()
        self.labels_train.clear()

        self.features_valid.clear()
        self.labels_valid.clear()

        return None

    def configure_optimizers(self):
        params = list(self.named_parameters())
        print("params")
        print([n for n, p in params])

        def is_backbone(n):
            return "text_encoder" in n

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], "lr": self.lr_lm},
            {"params": [p for n, p in params if not is_backbone(n)], "lr": self.lr},
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs - 1
        )
        return [optimizer], [scheduler]


def on_save_checkpoint(checkpoint):
    for key in list(checkpoint["state_dict"].keys()):
        if "text_encoder" in key:
            print("deleting ", key)
            del checkpoint["state_dict"][key]
