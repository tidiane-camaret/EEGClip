import torch
import pytorch_lightning as pl

class EEGClassifierModule(pl.LightningModule):
    def __init__(self, classifier_model):
        super().__init__()
        self.classifier = classifier_model
        self.loss_fn = torch.nn.NLLLoss()

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_nb):
        x, y, z = batch
        y_hat = self.classifier(x)

        loss = self.loss_fn(y_hat, y.long())
        self.log('train_loss', loss)

        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        #print('train_acc : ', accuracy)
        self.log('train_acc', accuracy)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, z = batch
        y_hat = self.classifier(x)

        loss = self.loss_fn(y_hat, y.long())
        self.log('validation_loss', loss)

        accuracy = torch.sum(torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        print('validation_acc : ', accuracy)
        self.log('validation_acc', accuracy)

        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 0.0625 * 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 49)
        return [optimizer], [scheduler]