import torch
import pytorch_lightning as pl

class EEGClassifierModule(pl.LightningModule):
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
        #print("OUTPUT SHAPE : ", y_hat.shape)
        
        y_hat = torch.mean(y_hat, dim=2)
        

        loss = self.loss_fn(y_hat, y.long())
        self.log('train_loss', loss) 

        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        #accuracy = accuracy.cpu().numpy()
        #print('train_acc : ', accuracy)
        self.log('train_acc', accuracy)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, z = batch
        print("INPUT SHAPE : ", x.shape)
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

