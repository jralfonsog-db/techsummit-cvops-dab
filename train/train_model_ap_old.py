# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("catalog", "techsummit_cvops_development")
dbutils.widgets.text("schema", "pcb")
dbutils.widgets.text("model_name", "techsummit_cvops")
dbutils.widgets.text("volumes_table", "deltatorch_files")

environment = dbutils.widgets.get("environment")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
volumes_table = dbutils.widgets.get("volumes_table")

# COMMAND ----------


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, models
#from torchdelta.deltadataset import DeltaIterableDataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# before 13.X DBR - install this on your cluster 
#from pyspark.ml.torch.distributor import TorchDistributor
from deltatorch import create_pytorch_dataloader
from deltatorch import FieldSpec

import mlflow
import os
from dataclasses import dataclass



## Specifying the mlflow host server and access token 
# We put them to a variable to feed into horovod later on
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


# Full username, e.g. "<first>.<last>@databricks.com"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Short form of username
user = username.split("@")[0].replace(".","_")


ckpt_path = f"/Volumes/{catalog}/{schema}/torch_checkpoints/"
temp_write_path = f"/Volumes/{catalog}/{schema}/deltatorch_files"
train_path = f"{temp_write_path}/train"
test_path = f"{temp_write_path}/test"

train_length = spark.read.format("delta").load(train_path).count()
test_length = spark.read.format("delta").load(test_path).count()


# COMMAND ----------

from train import trainer
from data_loader import DeltaDataModule
from model_training import CVModel

dm = DeltaDataModule(train_path, test_path)
model = CVModel(dm.num_classes)

trainer.trainer(dm=dm, model=model, num_gpus=1, db_host = db_host, db_token=db_token, ckpt_path=ckpt_path)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Training 

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor
from train import trainer
from data_loader import DeltaDataModule
from model_training import CVModel

dm = DeltaDataModule(train_path, test_path)
model = CVModel(dm.num_classes)

# be sure you are on dbr 13+
#path_training = "./trainer.py"
#args = ["--num_gpus=1"] #"--learning_rate=0.001", "--batch_size=16"
# on lover than 13+ this will require you to install this manually 
#trainer.trainer(dm=dm, model=model, num_gpus=1, db_host = db_host, db_token=db_token, ckpt_path=ckpt_path)
distributed = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True)
distributed.run(trainer.trainer, dm, model, 1, db_host, db_token, ckpt_path)

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor
# from data_loader import DeltaDataModule
# from model_training import CVModel

# # be sure you are on dbr 13+
# path_training = "./trainer.py"
# args = ["--num_gpus=1"] #"--learning_rate=0.001", "--batch_size=16"
# # on lover than 13+ this will require you to install this manually 
# distributed = TorchDistributor(num_processes=2, local_mode=False, use_gpu=True)
# distributed.run(path_training , *args)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Scoring a sample

# COMMAND ----------

best_model = mlflow.search_runs(
    filter_string=f'attributes.status = "FINISHED"',
    order_by=["metrics.train_acc DESC"],
    max_results=1,
).iloc[0]

model_uri = "runs:/{}/model_cvops".format(best_model.run_id)
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

loaded_model = torch.load(local_path+"/data/model.pth", map_location=torch.device(device))

# COMMAND ----------

loaded_model

# COMMAND ----------


from PIL import Image
from torchvision import transforms
import numpy as np
import io

data_test = spark.read.format("delta").load(test_path).limit(2).toPandas()

# let's test that we have indded a bytes image 
img = data_test["content"].iloc[0]
Image.open(io.BytesIO(img))

# COMMAND ----------

# MAGIC  %md ### Pandas UDF 

# COMMAND ----------

import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import io
from pyspark.sql.functions import pandas_udf
from typing import Iterator

def feature_extractor(img):
    image = Image.open(io.BytesIO(img))
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


model_b = sc.broadcast(loaded_model)

@pandas_udf("struct<score: float, label: int, labelName: string>")
def apply_vit(images_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    model = model_b.value
    model = model.to(torch.device("cuda"))
    model.eval()
    id2label = {0: "normal", 1: "anomaly"}
    with torch.set_grad_enabled(False):
        for images in images_iter:
            pil_images = torch.stack(
                [
                    feature_extractor(b)
                    for b in images
                ]
            )
            pil_images = pil_images.to(torch.device("cuda"))
            outputs = model(pil_images)
            preds = torch.max(outputs, 1)[1].tolist()
            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
            yield pd.DataFrame(
                [
                    {"score": prob, "label": pred, "labelName": id2label[pred]}
                    for pred, prob in zip(preds, probs)
                ]
            )

# COMMAND ----------

prediction_table = "techsummit_cvops_development.pcb.training_dataset"
#df_prep_gold = spark.read.format("delta").load(f"test_path")
df_prep_gold = spark.read.table(f"{prediction_table}") 
df_prep_gold.write.saveAsTable(f"{catalog}.{schema}.circuit_board_gold")

# COMMAND ----------

display(df_prep_gold)

# COMMAND ----------

# with the Brodcasted model we won 40sec, but it's because we do not have a big dataset, in a case of a big set this could significantly speed up things. 
# also take into account that some models may use Batch Inference natively - check API of your Framework. 
# 
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 32)
# # Change naming here to the PRODUCTION or PREPRODUCTION 
predictions_df = spark.table(f"{catalog}.{schema}.circuit_board_gold").withColumn("prediction", apply_vit("content"))
display(predictions_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model Serving 

# COMMAND ----------

from io import BytesIO

class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # instantiate model in evaluation mode
        model.to(torch.device("cpu"))
        self.model = model.eval()

    def feature_extractor(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

images = spark.table("ap.cv_ops.circuit_board_gold").take(25)

loaded_model = torch.load(
    local_path + "/data/model.pth", map_location=torch.device("cpu")
)
wrapper = CVModelWrapper(loaded_model)

b64image1 = base64.b64encode(images[0]["content"]).decode("ascii")
b64image2 = base64.b64encode(images[1]["content"]).decode("ascii")
b64image3 = base64.b64encode(images[3]["content"]).decode("ascii")
b64image4 = base64.b64encode(images[4]["content"]).decode("ascii")
b64image24 = base64.b64encode(images[24]["content"]).decode("ascii")

df_input = pd.DataFrame(
    [b64image1, b64image2, b64image3, b64image4, b64image24], columns=["data"])

df = wrapper.predict("", df_input)
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model in UC 

# COMMAND ----------

## OLD CODE 


# class CVModelWrapper(mlflow.pyfunc.PythonModel):

#     def __init__(self, model):     
#         # instantiate model in evaluation mode
#         self.model = model.eval()

#     def prep_data(self, data):
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])
#         return transform(Image.open(io.BytesIO(data)))    

#     def predict(self, context, model_input):
#         id2label = {
#         0: "good_device",
#         1: "bad_device"}
#         #if we have a dataframe, take the first column as input (regardless of the name)
#         if isinstance(model_input, pd.DataFrame):
#             model_input = model_input.iloc[:, 0]
#         # transform input images
#         features = model_input.map(lambda x: self.prep_data(x))
#         #raise Exception(features)
#         # make predictions
#         outputs = []
#         for i in torch.utils.data.DataLoader(features):
#             with torch.no_grad():
#                 self.model.eval()  
#                 output = self.model(i)
#                 indexs = output.cpu().numpy().argmax()
#                 labels = id2label[indexs]
#                 outputs.append(labels)
#         #raise Exception(outputs)        
#         return pd.Series(outputs)

# # another way of using a custom wrapper with UDF 
# @pandas_udf("string")
# def cv_model_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
#     #load the model from the registry
#     model = CVModelWrapper(loaded_model.model)
#     for s in iterator:
#         yield model.predict('', s)


# from pyspark.sql.functions import struct, col
# spark.udf.register("cv_model_udf", cv_model_udf)

# spark_df = spark.read.format("delta").load(spark_test_path)
# df = spark_df.withColumn('predictions', cv_model_udf(struct(*map(col, ["image"]))))
# display(df)

# COMMAND ----------

# MAGIC %sql 
# MAGIC USE CATALOG ap;
# MAGIC GRANT CREATE_MODEL ON SCHEMA cv_ops TO `anastasia.prokaieva@databricks.com`

# COMMAND ----------

import mlflow
# Set the registry URI to "databricks-uc" to configure
# the MLflow client to access models in UC
mlflow.set_registry_uri("databricks-uc")

model_name = "ap.cv_ops.cvops_model"

from mlflow.models.signature import infer_signature,set_signature
img = df_input['data']
predict_sample = df[['score','label']]
# To register models under UC you require to log signature for both 
# input and output 
signature = infer_signature(img, predict_sample)

print(f"Your signature is: \n {signature}")

with mlflow.start_run(run_name=model_name) as run:
    mlflowModel = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapper,
        input_example=df_input,
        signature=signature,
        registered_model_name=model_name,
    )
##Alternatively log your model and register later 
# mlflow.register_model(model_uri, "ap.cv_ops.cvops_model")


# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(model_name, "Champion", 1)

# COMMAND ----------

model_version_uri = f"models:/{model_name}@Champion"
# Or another option
# model_version_uri = f"models:/{model_name}/1"
loaded_model_uc = mlflow.pyfunc.load_model(model_version_uri)
# champion_version = client.get_model_version_by_alias("prod.ml_team.iris_model", "Champion")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Testing Serving on GPU 
# MAGIC

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/cv_ops_tech_summit/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

score_model(df_input)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Checking inference tables 

# COMMAND ----------

#dbfs:/model-serving-logs-cvops-techsummit
