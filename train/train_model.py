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

from utils import get_current_url, get_username, get_pat
from model_training import DeltaDataModule, CVModel

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

def train_model(dm, num_gpus=1, single_node=True):
    # We put them into these environment variables as this is where mlflow will look by default
    os.environ['DATABRICKS_HOST'] = get_current_url()
    os.environ['DATABRICKS_TOKEN'] = get_pat()
    torch.set_float32_matmul_precision("medium")
    if single_node or num_gpus == 1:
        num_devices = num_gpus
        num_nodes = 1
        strategy = "auto"
    else:
        num_devices = 1
        num_nodes = num_gpus
        strategy = 'ddp_notebook'  # check this is ddp or ddp_notebook

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        mode="min",
        monitor="val_loss",  # this has been saved under the Model Trainer - inside the validation_step function
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
            logger=mlf_logger)

        print(
            f"Global Rank: {trainer.global_rank} - Local Rank: {trainer.local_rank} - World Size: {trainer.world_size}")
        # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = CVModel(2)
        # model.to(device)

        trainer.fit(model, dm)
        print("Training is done ")
        val_metrics = trainer.validate(model, dataloaders=dm.val_dataloader(), verbose=False)
        # index of the current process across all nodes and devices. Only log on rank 0
        if trainer.global_rank == 0:
            print("Logging our model")
            reqs = mlflow.pytorch.get_default_pip_requirements() + ["pytorch-lightning==" + pl.__version__]
            # pytorch_model = CVTorchModelWrapper(model.model)  # We use our previously defined class to wrap our model including the overriden predict method
            pytorch_model = model.model
            mlflow.pytorch.log_model(artifact_path="model", pytorch_model=pytorch_model, pip_requirements=reqs)
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
            # Convert the base64 to PIL images
            images = images['data'].apply(lambda b: base64.b64decode(b))
            return predict_byte_serie(images, self.model)


# Load it as CPU as our endpoint will be cpu for now
model_cpu = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(torch.device("cpu"))
rt_model = RealtimeCVTorchModelWrapper(model_cpu)


def to_base64(b):
    return base64.b64encode(b).decode("ascii")


df = spark.table(training_dataset_augmented_table)

# Let's try locally before deploying our endpoint to make sure it works as expected:
pdf = df.limit(10).toPandas()

# Transform our input as a pandas dataframe containing base64 as this is what our serverless model endpoint will receive.
input_example = pd.DataFrame(pdf["content"].apply(to_base64).to_list(), columns=["data"])
predictions = rt_model.predict(None, input_example)
display(predictions)
