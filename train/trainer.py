import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, models

# from torchdelta.deltadataset import DeltaIterableDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# before 13.X DBR - install this on your cluster
# from pyspark.ml.torch.distributor import TorchDistributor
from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

import mlflow
import os
from dataclasses import dataclass

from data_loader import DeltaDataModule
from model_training import CVModel


import data_loader, model_training

MAX_EPOCH_COUNT = 20
BATCH_SIZE = 16
STEPS_PER_EPOCH = 2
EARLY_STOP_MIN_DELTA = 0.01
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 2

# create experiment if does not exist
def mlflow_set_experiment(experiment_path: str = None):
    try:
        print(f"Setting our existing experiment {experiment_path}")
        mlflow.set_experiment(experiment_path)
        experiment = mlflow.get_experiment_by_name(experiment_path)
    except:
        print("Creating a new experiment and setting it")
        experiment = mlflow.create_experiment(name=experiment_path)
        mlflow.set_experiment(experiment_id=experiment_path)


def trainer(
    dm=None,
    model=None,
    num_gpus=1,
    single_node=True,
    db_host=None,
    db_token=None,
    experiment_path="/Shared/techsummit_cvops/techsummit_cvops_mlflow",
    ckpt_path=None,
):
    import data_loader, model_training

    # We put them into these environment variables as this is where mlflow will look by default
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token
    #mlflow_set_experiment(experiment_path)
    #print(f"Your experiment is set to {experiment_path }")
    # mlflow.pytorch.autolog(disable=True)
    torch.set_float32_matmul_precision("medium")

    if single_node:
        num_devices = num_gpus
        num_nodes = 1
        strategy = "dp"
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
        stopping_threshold=0.5,
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

    # Initialize a trainer
    pl_trainer = pl.Trainer(
        default_root_dir=ckpt_path,
        accelerator="gpu",
        max_epochs=120,
        check_val_every_n_epoch=5,
        devices=num_devices,
        callbacks=[early_stop_callback, checkpoint_callback, tqdm_callback],
        strategy=strategy,
        num_nodes=num_nodes,
    )

    print(f"Global Rank: {pl_trainer.global_rank}")
    print(f"Local Rank: {pl_trainer.local_rank}")

    print(f"World Size: {pl_trainer.world_size}")

    pl_trainer.fit(model, dm)
    print("Training is done ")
    val_metrics = pl_trainer.validate(
        model, dataloaders=dm.val_dataloader(), verbose=False
    )
    print(val_metrics[0])

    if pl_trainer.global_rank == 0:
        # AutoLog does not work with DDP
        print("We are enabling AUTOLOG")
        # mlflow.pytorch.autolog(disable=False, log_models=True)
        mlflow.pytorch.autolog(
            log_every_n_epoch=10,
            log_every_n_step=None,
            log_models=True,
            log_datasets=True,
            disable=False,
        )
        from mlflow.models.signature import infer_signature

        with mlflow.start_run(run_name="testing_cvOps") as run:

            # Train the model ⚡🚅⚡
            print("We are logging our model")
            reqs = mlflow.pytorch.get_default_pip_requirements() + [
                "git+https://github.com/mshtelma/torchdelta",
                "pytorch-lightning==" + pl.__version__,
            ]

            mlflow.pytorch.log_model(
                artifact_path="model_cvops",
                pytorch_model=model.model,
                pip_requirements=reqs,
            )
            mlflow.set_tag("field_demos", "cvops")
            # log and returns model accuracy
            mlflow.log_metrics(val_metrics[0])

    return 'Finished'