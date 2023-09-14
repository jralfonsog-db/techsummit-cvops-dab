# Databricks notebook source
# MAGIC %pip install pytorch-lightning git+https://github.com/delta-incubator/deltatorch.git

# COMMAND ----------

from PIL import Image
import io
from io import BytesIO
import os
import torch
from torchvision import models
import torchvision.transforms as tf
from torchmetrics import Accuracy
from torch.nn import functional as nnf
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.file_utils import TempDir
import cloudpickle
import base64
import pandas as pd

from model_serving import CVModelWrapper

# COMMAND ----------

dbutils.widgets.text("environment", "environment")
dbutils.widgets.text("catalog", "catalog")
dbutils.widgets.text("schema", "schema")
dbutils.widgets.text("model_name", "model_name")
dbutils.widgets.text("experiment_name", "experiment_name")

environment = dbutils.widgets.get("environment")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = f"/Shared/{dbutils.widgets.get('experiment_name')}"

# COMMAND ----------

ckpt_path = f"/Volumes/{catalog}/{schema}/torch_checkpoints/"
training_dataset_augmented_table = "training_dataset_augmented"
train_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/train"
test_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/test"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

mlflow.set_experiment(experiment_name=experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve best model
# MAGIC

# COMMAND ----------

best_model = mlflow.search_runs(
    filter_string=f'attributes.status = "FINISHED"',
    order_by=["metrics.train_acc DESC"],
    max_results=1,
).iloc[0]

model_uri = f"runs:/{best_model.run_id}/model"
print(f"Your model_uri is: {model_uri}")

# COMMAND ----------

import os
import torch
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

#local_path = ModelsArtifactRepository(f"models:/{model_name}/Production").download_artifacts("") # download model from remote registry
local_path = mlflow.artifacts.download_artifacts(model_uri)
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

pytorch_model = torch.load(local_path+"/data/model.pth", map_location=torch.device(device))

# COMMAND ----------

class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # instantiate model in evaluation mode
        model.to(torch.device("cpu"))
        self.model = model.eval()

    def feature_extractor(self, image):
        transform = tf.Compose([
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        #return transform(Image.open(io.BytesIO(data)))
        return transform(Image.open(BytesIO(base64.b64decode(image))))

    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          id2label = {0: "normal", 1: "anomaly"}
          # add here check if this is a DataFrame 
          # if this is an image remove iterrows 
          pil_images = torch.stack([self.feature_extractor(row[0]) for _, row in images.iterrows()])
          #pil_images = images.map(lambda x: self.feature_extractor(x))
          pil_images = pil_images.to(torch.device("cpu"))
          outputs = self.model(pil_images)
          preds = torch.max(outputs, 1)[1]
          probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
          labels = [id2label[pred] for pred in preds.tolist()]

          return pd.DataFrame( data=dict(
            score=probs,
            label=preds,
            labelName=labels)
          )


# COMMAND ----------

model_wrapper = CVModelWrapper(pytorch_model)

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

predict_df = model_wrapper.predict("", input_example)
display(predict_df)

# COMMAND ----------

import mlflow
# Set the registry URI to "databricks-uc" to configure
# the MLflow client to access models in UC
mlflow.set_registry_uri("databricks-uc")

model_name = "ap.cv_ops.cvops_model"

from mlflow.models.signature import infer_signature,set_signature
img = input_example['data']
predict_sample = predict_df[['score','label']]
# To register models under UC you require to log signature for both 
# input and output 
signature = infer_signature(img, predict_sample)

print(f"Your signature is: \n {signature}")

with mlflow.start_run(run_name=model_name) as run:
    mlflowModel = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model_wrapper,
        input_example=input_example,
        signature=signature,
        registered_model_name=model_name,
    )

# COMMAND ----------

            # pytorch_model = CVTorchModelWrapper(model.model)  # We use our previously defined class to wrap our model including the overriden predict method
            model_wrapper = CVModelWrapper(model.model)

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

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/{run_id}/model", f"{catalog}.{schema}.{model_name}")
