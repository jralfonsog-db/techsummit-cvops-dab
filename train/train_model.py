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
from mlflow.utils.file_utils import TempDir
import cloudpickle
import base64
import pandas as pd

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
# MAGIC # Definitions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utils

# COMMAND ----------

# Below are initialization related functions
def get_cloud_name():
    return spark.conf.get("spark.databricks.clusterUsageTags.cloudProvider").lower()


def get_current_url():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()


def get_username() -> str:  # Get the user's username
    return (
        dbutils()
        .notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .apply("user")
        .lower()
        .split("@")[0]
        .replace(".", "_")
    )


cleaned_username = get_username()


def get_pat():
    return (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )


def get_request_headers() -> str:
    return {
        "Authorization": f"""Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}"""
    }


def get_instance() -> str:
    return (
        dbutils()
        .notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .apply("browserHostName")
    )

# COMMAND ----------

print(get_cloud_name())
print(get_current_url())
print(get_username())
print(get_request_headers())
print(get_instance())

# COMMAND ----------

def set_experiment_permission(experiment_path):
  url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply("api_url")
  import requests
  pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
  headers =  {"Authorization": "Bearer " + pat_token, 'Content-type': 'application/json'}
  status = requests.get(url+"/api/2.0/workspace/get-status", params = {"path": experiment_path}, headers=headers).json()
  if "object_id" not in status:
        print(f"error setting up shared experiment permission: {status}")
  else:
    #Set can manage to all users to the experiment we created as it's shared among all
    params = {"access_control_list": [{"group_name": "users","permission_level": "CAN_MANAGE"}]}
    permissions = requests.patch(f"{url}/api/2.0/permissions/experiments/{status['object_id']}", json = params, headers=headers)
    if permissions.status_code != 200:
      print("ERROR: couldn't set permission to all users to the autoML experiment")
          
def init_experiment_for_batch(demo_name, experiment_name):
  #You can programatically get a PAT token with the following
  pat_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
  url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
  #current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
  import requests
  xp_root_path = f"/dbdemos/experiments/{demo_name}"
  requests.post(f"{url}/api/2.0/workspace/mkdirs", headers = {"Accept": "application/json", "Authorization": f"Bearer {pat_token}"}, json={ "path": xp_root_path})
  xp = f"{xp_root_path}/{experiment_name}"
  print(f"Using common experiment under {xp}")
  mlflow.set_experiment(xp)
  set_experiment_permission(xp)
  return mlflow.get_experiment_by_name(xp)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## LightningDataModule from Delta files

# COMMAND ----------

# Deltatorch makes it easy to load Delta Dataframe to torch and efficiently distribute it among multiple nodes.
# This requires deltatorch being installed. 
# Note: For small dataset, a LightningDataModule exemple directly using hugging face transformers is also available in the _resources/00-init notebook.

class DeltaDataModule(pl.LightningDataModule):
    #Creating a Data loading module with Delta Torch loader 
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

# MAGIC %md
# MAGIC ## Model definition

# COMMAND ----------

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
        loss = nnf.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        self.log("train_loss", torch.tensor([loss]), on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", torch.tensor([acc]), on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["content"]
        y = batch["label"]
        pred = self(x)
        loss = nnf.cross_entropy(pred, y)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow model wrapper
# MAGIC
# MAGIC This wrapper will override the default predict method to allow our model to include the required image preprocessing to receive images in base64 for inference

# COMMAND ----------

id2label = {0: "normal", 1: "anomaly"}
def predict_byte_serie(content_as_byte_serie, model, device = torch.device("cpu")):
  #Transform the bytes as PIL images
  image_list = content_as_byte_serie.apply(lambda b: Image.open(io.BytesIO(b))).to_list()
  #Apply our transformations & stack them for our model
  vector_images = torch.stack([transform(b).to(device) for b in image_list])
  #Call the model to get our inference
  outputs = model(vector_images)
  #Returns a proper results with Score/Label/Name
  preds = torch.max(outputs, 1)[1].tolist()
  probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
  return pd.DataFrame({"score": probs, "label": preds, "labelName": [id2label[p] for p in preds]})

# COMMAND ----------

# Model wrapper
class CVTorchModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        # instantiate model in evaluation mode
        self.model.eval()

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          #Convert the base64 to PIL images
          images = images['data'].apply(lambda b: base64.b64decode(b))
          return predict_byte_serie(images, self.model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train function

# COMMAND ----------

xp = init_experiment_for_batch("techsummit-cvops", f"{environment}-experiment")

def train_model(dm, num_gpus=1, single_node=True):
    # We put them into these environment variables as this is where mlflow will look by default
    os.environ['DATABRICKS_HOST'] = get_current_url()
    os.environ['DATABRICKS_TOKEN'] = get_pat()
    torch.set_float32_matmul_precision("medium")
    if single_node or num_gpus == 1:
        num_devices = num_gpus
        num_nodes = 1
        strategy="auto"
    else:
        num_devices = 1
        num_nodes = num_gpus
        strategy = 'ddp_notebook' # check this is ddp or ddp_notebook

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        mode="min",
        monitor="val_loss", # this has been saved under the Model Trainer - inside the validation_step function 
        dirpath=ckpt_path,
        filename="sample-cvops-{epoch:02d}-{val_loss:.2f}"
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
        log_rank_zero_only=True
    )

    tqdm_callback = pl.callbacks.TQDMProgressBar(
        refresh_rate=STEPS_PER_EPOCH,
        process_position=0
    )
    # make your list of choices that you want to add to your trainer 
    callbacks = [early_stop_callback, checkpoint_callback, tqdm_callback]

    # AutoLog does not work with DDP 
    mlflow.pytorch.autolog(disable=False, log_models=False)
    with mlflow.start_run() as run:
      mlf_logger = MLFlowLogger(experiment_name=xp.name, run_id=run.info.run_id)
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
          logger=mlf_logger)

      print(f"Global Rank: {trainer.global_rank} - Local Rank: {trainer.local_rank} - World Size: {trainer.world_size}")
      # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
      model = CVModel(2)
      # model.to(device)
      
      trainer.fit(model, dm)
      print("Training is done ")
      val_metrics = trainer.validate(model, dataloaders=dm.val_dataloader(), verbose=False)
      #index of the current process across all nodes and devices. Only log on rank 0
      if trainer.global_rank == 0:
        print("Logging our model")
        reqs = mlflow.pytorch.get_default_pip_requirements() + ["pytorch-lightning==" + pl.__version__]
        # pytorch_model = CVTorchModelWrapper(model.model)  # We use our previously defined class to wrap our model including the overriden predict method
        pytorch_model = model.model
        mlflow.pytorch.log_model(artifact_path="model", pytorch_model=pytorch_model, pip_requirements=reqs)
        #Save the test/validate transform as we'll have to apply the same transformation in our pipeline.
        #We could alternatively build a model wrapper to encapsulate these transformations as part as of our model (that's what we did with the huggingface implementation).
        with TempDir(remove_on_exit=True) as local_artifacts_dir:
          # dump tokenizer to file for mlflow
          transform_path = local_artifacts_dir.path("transform.pkl")
          with open(transform_path, "wb") as fd:
            cloudpickle.dump(dm.transform, fd)
          mlflow.log_artifact(transform_path, "model")
        mlflow.set_tag("cvops", "pcb_torch")
        #log and returns model accuracy
        mlflow.log_metrics(val_metrics[0])
    
    return mlf_logger.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC # Training
# MAGIC
# MAGIC We'll introduce a environment check before training. If we are in `development`, we will use a single GPU. If we are in `preproduction`, we'll assume we are going to train a bigger model and so we'll use 2 GPU instead

# COMMAND ----------

delta_dataloader = DeltaDataModule(train_deltatorch_path, test_deltatorch_path)

if environment == "development":
  run_id = train_model(delta_dataloader, 1, True)
elif environment == "preproduction" or "production":
  run_id = train_model(delta_dataloader, 2, True)
else:
  print("Wrong environment. Please select 'development', 'preproduction' or 'production'.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model in UC

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/{run_id}/model", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Serving

# COMMAND ----------

from io import BytesIO
import base64
import pandas as pd

# Model wrapper
class RealtimeCVTorchModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        # instantiate model in evaluation mode
        self.model.eval()

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          #Convert the base64 to PIL images
          images = images['data'].apply(lambda b: base64.b64decode(b))
          return predict_byte_serie(images, self.model)


#Load it as CPU as our endpoint will be cpu for now
model_cpu = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(torch.device("cpu"))
rt_model = RealtimeCVTorchModelWrapper(model_cpu)

def to_base64(b):
  return base64.b64encode(b).decode("ascii")


df = spark.table(training_dataset_augmented_table)

#Let's try locally before deploying our endpoint to make sure it works as expected:
pdf = df.limit(10).toPandas()

#Transform our input as a pandas dataframe containing base64 as this is what our serverless model endpoint will receive.
input_example = pd.DataFrame(pdf["content"].apply(to_base64).to_list(), columns=["data"])
predictions = rt_model.predict(None, input_example)
display(predictions)
