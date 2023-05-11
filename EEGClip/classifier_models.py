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
                  lr = 5e-3,
                  weight_decay = 5e-4,
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

        self.preds = []
        self.true_labels = []
        self.ids = []
        self.stop_ids = []
        self.all_is = []


    def forward(self, batch):
        eeg_batch, labels_batch, id_batch = batch
        eeg_batch = self.encoder(eeg_batch)
        #print(eeg_batch.shape)
        #eeg_batch = torch.mean(eeg_batch, dim=2)

        #eeg_batch = 
        preds_batch = eeg_batch #self.classifier(eeg_batch) #shape [B, Enc_size, N_Preds]

        labels_batch = labels_batch.long()

        return preds_batch, labels_batch, id_batch
    
    def training_step(self, batch, batch_nb):
        preds_batch, labels_batch, _ = self.forward(batch)
        loss = self.loss_fn(torch.mean(preds_batch,dim=2), labels_batch)
        self.log('train_loss', loss, prog_bar=True)

        #balanced_acc = balanced_accuracy_score(labels_batch.cpu().numpy(), torch.argmax(torch.mean(preds_batch,dim=2), dim=1).cpu().numpy())
        #self.log('train_balanced_acc', balanced_acc)

        return loss
    
    def validation_step(self, batch, batch_nb):
        eeg_batch, labels_batch, id_batch = batch
        preds_batch, labels_batch, _ = self.forward(batch)

        self.preds.extend(preds_batch.cpu().numpy())
        self.true_labels.extend(labels_batch.cpu().numpy())
        self.ids.append(id_batch[0])
        self.stop_ids.append(id_batch[2])
        self.all_is.extend(id_batch)

        loss = self.loss_fn(torch.mean(preds_batch,dim=2), labels_batch)
        self.log('val_loss', loss)

        #balanced_acc = balanced_accuracy_score(labels_batch.cpu().numpy(), torch.argmax(torch.mean(preds_batch,dim=2), dim=1).cpu().numpy())
        #self.log('val_balanced_acc', balanced_acc, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        
        all_preds = self.preds
        all_ys = self.true_labels
        #ids = torch.cat(self.ids).cpu()
        #stop_ids = torch.cat(self.stop_ids).cpu()
        all_is = self.all_is

        self.preds = []
        self.true_labels = []
        self.ids = []
        self.stop_ids = []
        self.all_is = []

        print("balance in valid set : ", np.sum(all_ys)/len(all_ys))  
        
        all_preds = np.array(all_preds)
        all_ys = np.array(all_ys)
        all_is = [a.cpu() for a in all_is]
        crop_preds = np.mean(all_preds, axis=(2)).argmax(axis=1)
        acc_cropwise = np.mean((crop_preds == all_ys))
        trial_ys = all_ys[np.diff(torch.cat(all_is[0::3]), prepend=[np.inf]) != 1]
        #print(all_preds.shape, torch.cat(all_is[0::3]).shape)
        preds_per_trial = trial_preds_from_window_preds(all_preds, torch.cat(all_is[0::3]), 
                                                        torch.cat(all_is[2::3]),)
        trial_preds = np.array([p.mean(axis=1).argmax(axis=0) for p in preds_per_trial])
        acc_trial = np.mean(trial_preds  == trial_ys)
        """
        pred_labels = torch.unsqueeze(pred_labels, dim=1)
        #print(pred_labels.shape, ids.shape, stop_ids.shape)
        pred_per_recording = trial_preds_from_window_preds(pred_labels,
                                                           ids,
                                                           stop_ids
                                                           )
        #print(len(pred_per_recording))
        
        pred_per_recording = np.array([p.mean(axis=1).argmax(axis=0) for p in pred_per_recording])
        
        #print(pred_per_recording.shape)

        true_per_recording = true_labels[np.diff(ids, prepend = [np.inf]) != 1]

        #print(true_per_recording.shape)

        acc_per_recording = balanced_accuracy_score(true_per_recording, pred_per_recording)
        """
        self.log('val_acc_per_recording', acc_trial, prog_bar=True)
        self.log('val_acc_per_crop',acc_cropwise, prog_bar=True)
        
        

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