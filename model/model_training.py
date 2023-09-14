import pytorch_lightning as pl
from torchvision import models
import torchvision.transforms as tf
from torchmetrics import Accuracy
import torch.nn.functional as F
import torchmetrics.functional as FM
import torch
import logging
import datetime as dt
from io import BytesIO
import mlflow


class CVModel(pl.LightningModule):
    """
    We are going to define our model class here 
    """
    def __init__(self, num_classes: int = 2, learning_rate: float = 2e-4, momentum:float=0.9, family:str='mobilenet'):
        super().__init__()

        self.save_hyperparameters() # LightningModule allows you to automatically save all the hyperparameters 
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = self.get_model(num_classes, learning_rate, family)
        self.family = family
 
    def get_model(self, num_classes, learning_rate, family):
        """
        
        This is the function that initialises our model.
        If we wanted to use other prebuilt model libraries like timm we would put that model here
        
        """
        
        if family == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
        elif family == 'resnext':
            model = models.resnext50_32x4d(pretrained=True)
        
        # Freeze parameters in the feature extraction layers and replace the last layer
        for param in model.parameters():
            param.requires_grad = False
    
        # New modules have `requires_grad = True` by default
        if family == 'mobilenet':
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif family == 'resnext':
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        return model

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["content"]
        y = batch["label"]
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        self.log("train_loss", torch.tensor([loss]), on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", torch.tensor([acc]), on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["content"]
        y = batch["label"]
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        self.log("val_loss", torch.tensor([loss]), prog_bar=True)
        self.log("val_acc", torch.tensor([acc]), prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):

        if self.family == 'mobilenet':
            params = self.model.classifier[1].parameters()
        elif self.family == 'resnext':
            params = self.model.fc.parameters()
        
        optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
        
        return optimizer
      

# Deltatorch makes it easy to load Delta Dataframe to torch and efficiently distribute it among multiple nodes.
# This requires deltatorch being installed.
class DeltaDataModule(pl.LightningDataModule):
    # Creating a Data loading module with Delta Torch loader
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        super().__init__()

        self.transform = tf.Compose([
            tf.Lambda(lambda x: x.convert("RGB")),
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        return self.dataloader(self.train_path, batch_size=64)

    def val_dataloader(self):
        return self.dataloader(self.test_path, batch_size=64)

    def test_dataloader(self):
        return self.dataloader(self.test_path, batch_size=64)
