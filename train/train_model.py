# Databricks notebook source
# MAGIC %pip install pytorch-lightning git+https://github.com/delta-incubator/deltatorch.git

# COMMAND ----------

from PIL import Image
import pytorch_lightning as pl
from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec
import torchvision.transforms as tf
import io
from io import BytesIO
import os
import torch
from torchvision import models
from torchmetrics import Accuracy
from torch.nn import functional as nnf
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.file_utils import TempDir
import cloudpickle
import base64
import pandas as pd

# COMMAND ----------

from utils import get_current_url, get_username, get_pat
from model_training import DeltaDataModule, CVModel
from model_serving import CVModelWrapper

# COMMAND ----------

# This belongs to model_training.py

import pytorch_lightning as pl
from torchvision import models
import torchvision.transforms as tf
from torchmetrics import Accuracy
import torch.nn.functional as F
import torchmetrics.functional as FM
import torch
import logging
import datetime as dt

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
# Note: For small dataset, a LightningDataModule exemple directly using hugging face transformers is also available in the _resources/00-init notebook.

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


# COMMAND ----------

dbutils.widgets.text("environment", "environment")
dbutils.widgets.text("catalog", "catalog")
dbutils.widgets.text("schema", "schema")
dbutils.widgets.text("model_name", "model_name")

environment = dbutils.widgets.get("environment")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

ckpt_path = f"/Volumes/{catalog}/{schema}/torch_checkpoints/"
training_dataset_augmented_table = "training_dataset_augmented"
train_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/train"
test_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/test"

# COMMAND ----------

MAX_EPOCH_COUNT = 20
BATCH_SIZE = 16
STEPS_PER_EPOCH = 2
# EARLY_STOP_MIN_DELTA = 0.01
EARLY_STOP_MIN_DELTA = 0.5
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 8

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train function

# COMMAND ----------

# MAGIC %md
# MAGIC ### Input example

# COMMAND ----------

images = spark.table(training_dataset_augmented_table).take(25)
input_example = pd.DataFrame(
  [
    base64.b64encode(images[0]["content"]).decode("ascii"),
    base64.b64encode(images[1]["content"]).decode("ascii"),
    base64.b64encode(images[2]["content"]).decode("ascii"),
    base64.b64encode(images[3]["content"]).decode("ascii"),
    base64.b64encode(images[4]["content"]).decode("ascii")
  ],
  columns=["data"]
)

# COMMAND ----------

def train_model(dm, num_gpus=1, single_node=True):
    # We put them into these environment variables as this is where mlflow will look by default
    os.environ["DATABRICKS_HOST"] = get_current_url()
    os.environ["DATABRICKS_TOKEN"] = get_pat()
    torch.set_float32_matmul_precision("medium")
    if single_node or num_gpus == 1:
        num_devices = num_gpus
        num_nodes = 1
        strategy = "auto"
    else:
        num_devices = 1
        num_nodes = num_gpus
        strategy = "ddp_notebook"  # check this is ddp or ddp_notebook

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        mode="min",
        monitor="val_loss",  # this has been saved under the Model Trainer - inside the validation_step function
        dirpath=ckpt_path,
        filename="sample-cvops-{epoch:02d}-{val_loss:.2f}",
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=EARLY_STOP_MIN_DELTA,
        patience=EARLY_STOP_PATIENCE,
        stopping_threshold=0.001,
        strict=True,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=True,
        log_rank_zero_only=True,
    )

    tqdm_callback = pl.callbacks.TQDMProgressBar(
        refresh_rate=STEPS_PER_EPOCH, process_position=0
    )
    # make your list of choices that you want to add to your trainer
    callbacks = [early_stop_callback, checkpoint_callback, tqdm_callback]

    # AutoLog does not work with DDP
    mlflow.pytorch.autolog(disable=False, log_models=False)
    with mlflow.start_run() as run:
        mlf_logger = MLFlowLogger(run_id=run.info.run_id)
        # Initialize a trainer
        trainer = pl.Trainer(
            default_root_dir=ckpt_path,
            accelerator="gpu",
            max_epochs=50,
            check_val_every_n_epoch=2,
            devices=num_devices,
            callbacks=[early_stop_callback, checkpoint_callback, tqdm_callback],
            strategy=strategy,
            num_nodes=num_nodes,
            logger=mlf_logger,
        )

        print(
            f"Global Rank: {trainer.global_rank} - Local Rank: {trainer.local_rank} - World Size: {trainer.world_size}"
        )
        # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = CVModel(2)
        # model.to(device)

        trainer.fit(model, dm)
        print("Training is done ")
        val_metrics = trainer.validate(
            model, dataloaders=dm.val_dataloader(), verbose=False
        )
        # index of the current process across all nodes and devices. Only log on rank 0
        if trainer.global_rank == 0:
            print("Logging our model")
            reqs = mlflow.pytorch.get_default_pip_requirements() + [
                "pytorch-lightning==" + pl.__version__
            ]
            # pytorch_model = CVTorchModelWrapper(model.model)  # We use our previously defined class to wrap our model including the overriden predict method
            model_wrapper = CVModelWrapper(model.model)

            # model signature
            img = input_example["data"]
            predict_df = model_wrapper.predict("", input_example)
            predict_example = predict_df[["score", "label"]]
            signature = infer_signature(img, predict_example)

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_wrapper,
                input_example=input_example,
                signature=signature,
                registered_model_name=model_name,
                pip_requirements=reqs,
            )

            # Save the test/validate transform as we'll have to apply the same transformation in our pipeline.
            # We could alternatively build a model wrapper to encapsulate these transformations as part as of our model (that's what we did with the huggingface implementation).
            with TempDir(remove_on_exit=True) as local_artifacts_dir:
                # dump tokenizer to file for mlflow
                transform_path = local_artifacts_dir.path("transform.pkl")
                with open(transform_path, "wb") as fd:
                    cloudpickle.dump(dm.transform, fd)
                mlflow.log_artifact(transform_path, "model")
            mlflow.set_tag("cvops", "pcb_torch")
            # log and returns model accuracy
            mlflow.log_metrics(val_metrics[0])

    return mlf_logger.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Training
# MAGIC
# MAGIC We'll introduce a environment check before training. If we are in `development`, we will use a single GPU. If we are in `preproduction`, we'll assume we are going to train a bigger model and so we'll use 2 GPU instead

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor
# delta_dataloader = DeltaDataModule(train_deltatorch_path, test_deltatorch_path)
# distributed = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True)
# distributed.run(train_model, delta_dataloader, 1, False)

# COMMAND ----------

delta_dataloader = DeltaDataModule(train_deltatorch_path, test_deltatorch_path)

run_id = train_model(delta_dataloader, 1, True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model in UC

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/{run_id}/model", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Serving
