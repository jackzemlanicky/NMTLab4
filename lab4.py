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


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
model = Net()
print(model)
class MyTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
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
 # How does above class relate to below method and variable?

# Temporary hard-coded params until I can get this to work 
earlystop = EarlyStopping(monitor=None,patience=1,mode='min')

trainer =pl.Trainer(gpus=0,max_epochs=5,progress_bar_refresh_rate =0.5,flush_logs_every_n_steps=100)
trainer.fit(MyTrainer(),dataset.pdfdataset())