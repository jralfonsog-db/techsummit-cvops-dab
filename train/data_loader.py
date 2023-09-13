
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# before 13.X DBR - install this on your cluster 
#from pyspark.ml.torch.distributor import TorchDistributor
from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

class DeltaDataModule(pl.LightningDataModule):
    """
    Creating a Data loading module with Delta Torch loader 
    """
    def __init__(self, train_path = None, test_path= None):
        super().__init__()
        self.num_classes = 2

        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_path = test_path
        self.train_path = train_path

    def dataloader(self, path: str, batch_size=32):
       
        return create_pytorch_dataloader(
            path,
            id_field="id",
            fields=[
                FieldSpec("content", load_image_using_pil=True, transform=self.transform),
                FieldSpec("label"),
            ],
            shuffle=True,
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(
            self.train_path,
            batch_size=16,
        )

    def val_dataloader(self):
        return self.dataloader(self.test_path)

    def test_dataloader(self):
        return self.dataloader(self.test_path)