from pytorch_lightning.trainer.trainer import Trainer
import torchvision.models as models
from torchvision.models.vgg import vgg11
import byteplot as bp
import pytorch_dataset_template as dataset
import pytorch_lightning as pl
from torch.optim import Adam, optimizer


class VGGTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vgg11(False,True)
    def forward(self,x):
        x =self.model(x)
        return
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),lr=self.args.lr,weight_decay=self.args.weight_decay)
        return {'optimizer':optimizer}
 # How does above class relate to below method and variable?
trainer =Trainer
trainer.fit(VGGTrainer,dataset.train_dataloader)
