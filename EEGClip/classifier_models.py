import numpy as np
import torch
import pytorch_lightning as pl

from sklearn.metrics import balanced_accuracy_score
from braindecode.training.scoring import trial_preds_from_window_preds

class EEGClassifierModel(pl.LightningModule):
    """Model for classification for the TUH dataset.
    Can use pretrained encoders.

    Args:
        encoder (torch.nn.Module): pretrained encoder
        freeze_encoder (bool): whether to freeze encoder
       lr (float): learning rate
       weight_decay (float): weight decay
       n_classes (int): number of classes

    """
    def __init__(self,
                  EEGEncoder, 
                  freeze_encoder=True, 
                  lr=1e-3,
                  weight_decay=1e-6,
                  encoder_output_dim = 128,
                  n_classes = 2):
        super().__init__()
        self.encoder = EEGEncoder
        self.freeze_encoder = freeze_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes = n_classes

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = torch.nn.Linear(encoder_output_dim, self.n_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.pred_labels = []
        self.true_labels = []
        self.ids = []
        self.stop_ids = []


    def forward(self, batch):
        eeg_batch, labels_batch, id_batch = batch
        eeg_batch = self.encoder(eeg_batch)
        eeg_batch = torch.mean(eeg_batch, dim=2)

        eeg_batch = self.classifier(eeg_batch)
        pred_labels_batch = eeg_batch

        labels_batch = labels_batch.long()

        return pred_labels_batch, labels_batch, id_batch
    
    def training_step(self, batch, batch_nb):
        pred_labels_batch, labels_batch, _ = self.forward(batch)
        loss = self.loss_fn(pred_labels_batch, labels_batch)
        self.log('train_loss', loss)

        balanced_acc = balanced_accuracy_score(labels_batch.cpu().numpy(), torch.argmax(pred_labels_batch, dim=1).cpu().numpy())
        self.log('train_balanced_acc', balanced_acc)

        return loss
    
    def validation_step(self, batch, batch_nb):
        eeg_batch, labels_batch, id_batch = batch
        pred_labels_batch, labels_batch, _ = self.forward(batch)

        self.pred_labels.append(pred_labels_batch)
        self.true_labels.append(labels_batch)
        self.ids.append(id_batch[0])
        self.stop_ids.append(id_batch[2])

        loss = self.loss_fn(pred_labels_batch, labels_batch)
        self.log('val_loss', loss)

        balanced_acc = balanced_accuracy_score(labels_batch.cpu().numpy(), torch.argmax(pred_labels_batch, dim=1).cpu().numpy())
        self.log('val_balanced_acc', balanced_acc)

        return loss

    def on_validation_epoch_end(self):

        pred_labels = torch.cat(self.pred_labels).cpu()
        true_labels = torch.cat(self.true_labels).cpu()
        ids = torch.cat(self.ids).cpu()
        stop_ids = torch.cat(self.stop_ids).cpu()

        self.pred_labels = []
        self.true_labels = []
        self.ids = []
        self.stop_ids = []

        pred_labels = torch.unsqueeze(pred_labels, dim=1)
        print(pred_labels.shape, ids.shape, stop_ids.shape)
        pred_per_recording = trial_preds_from_window_preds(pred_labels,
                                                           ids,
                                                           stop_ids
                                                           )
        print(len(pred_per_recording))
        
        pred_per_recording = np.array([p.mean(axis=1).argmax(axis=0) for p in pred_per_recording])
        
        print(pred_per_recording.shape)

        true_per_recording = true_labels[np.diff(ids, prepend = [np.inf]) != 1]

        print(true_per_recording.shape)

        acc_per_recording = balanced_accuracy_score(true_per_recording, pred_per_recording)
        self.log('val_balanced_acc_per_recording', acc_per_recording)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]
    
    
"""
class EEGClassifierModel_old(pl.LightningModule):
    def __init__(self, eeg_classifier_model, lr):
        super().__init__()
        self.classifier = eeg_classifier_model #torch.nn.Sequential(*list(eeg_classifier_model.children())[:-1])
        self.loss_fn = torch.nn.CrossEntropyLoss() #NLLLoss()
        self.lr = lr

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_nb):
        x, y, z = batch
        print("INPUT SHAPE : ", x.shape)
        y_hat = self.classifier(x)
        print("OUTPUT SHAPE : ", y_hat.shape)
        
        y_hat = torch.mean(y_hat, dim=2) #TODO : what is the dimension we do the mean on ? (several preds per window ?)
        

        loss = self.loss_fn(y_hat, y.long())
        self.log('train_loss', loss) 

        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        #accuracy = accuracy.cpu().numpy()
        #print('train_acc : ', accuracy)
        self.log('train_acc', accuracy)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, z = batch
        #print("INPUT SHAPE : ", x.shape)
        y_hat = self.classifier(x)
        #print("OUTPUT SHAPE : ", y_hat.shape) 
        
        y_hat = torch.mean(y_hat, dim=2) #
        loss = self.loss_fn(y_hat, y.long())
        self.log('validation_loss', loss)

        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        #accuracy = accuracy.cpu().numpy()
        #print('validation_acc : ', accuracy)
        self.log('validation_acc', accuracy, on_epoch = True, prog_bar=True) #on_epoch automatically averages over epoch

        return loss 
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = self.trainer.max_epochs - 1)
        return [optimizer], [scheduler]

"""