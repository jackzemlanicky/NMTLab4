from pytorch_lightning import callbacks
from pytorch_lightning.trainer.trainer import Trainer
import torch
import torchvision.models as models
from torchvision.models.vgg import vgg11
import byteplot as bp
import pytorch_dataset_template as dataset
import pytorch_lightning as pl
from torch.optim import Adam, optimizer
from pytorch_lightning.callbacks import EarlyStopping
from torch.multiprocessing import freeze_support

if __name__=='__main__':
    print('loop')
    torch.multiprocessing.freeze_support()
class VGGTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vgg11()
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

earlystop = EarlyStopping(monitor=None,patience=1,mode='min')

trainer =pl.Trainer(gpus=0,max_epochs=5,progress_bar_refresh_rate =1,flush_logs_every_n_steps=100)
trainer.fit(VGGTrainer(),dataset.pdfdataset())