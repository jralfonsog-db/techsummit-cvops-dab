# Databricks notebook source
import os
import requests
import numpy as np
import pandas as pd
import json

# COMMAND ----------

from utils import get_current_url, get_pat

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

training_dataset_augmented_table = "training_dataset_augmented"

host = get_current_url()
pat = get_pat()

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = f'{host}/serving-endpoints/techsummit_cvops_development/invocations'
  headers = {'Authorization': f'Bearer {pat}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

images = spark.table(training_dataset_augmented_table).take(5)
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

score_model(input_example)
