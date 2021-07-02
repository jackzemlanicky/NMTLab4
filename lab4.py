from pytorch_lightning.trainer.trainer import Trainer
import torchvision.models as models
import byteplot as bp
import pytorch_dataset_template as dataset
import pytorch_lightning as pl
m = models.vgg11(False,True)
print(m)
dataset.testcase_test_pdfdataset()
trainer =Trainer
trainer.fit(m,dataset,dataset)