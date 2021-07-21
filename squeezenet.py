from pytorch_lightning import callbacks
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import dataset as dataset
import pytorch_lightning as pl
from torch.optim import Adam, optimizer
from pytorch_lightning.callbacks import EarlyStopping
from torch.multiprocessing import freeze_support
print(torchvision.__version__)
class SqueezeNetTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.squeezenet1_1()
    def forward(self,x):
        x =self.model(x)
        return x
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    def train_dataloader(self):
        return super().train_dataloader()
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),lr=.01)
        return {'optimizer':optimizer}
    def random_weights(self):
        self.model = models.squeezenet1_1(pretrained = False)
    def pretrained_weights(self):
        self.model = models.squeezenet1_1(pretrained = True)
    #modify the layers to fit our desired input/output
    def modify_model(self):
        self.model.classifier[1] = nn.Conv2d(512,1,(1,1),(1,1))


earlystop = EarlyStopping(monitor=None,patience=1,mode='min')

trainer =pl.Trainer(gpus=0,max_epochs=5,progress_bar_refresh_rate =1,flush_logs_every_n_steps=100)
trainer.fit(SqueezeNetTrainer(),dataset.pdfdataset())