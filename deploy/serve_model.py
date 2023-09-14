# Databricks notebook source
import mlflow

# COMMAND ----------

dbutils.widgets.text("catalog", "catalog")
dbutils.widgets.text("schema", "schema")
dbutils.widgets.text("model_name", "model_name")
dbutils.widgets.text("experiment_name", "experiment_name")

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

mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(model_uri=model_uri, name=f"{catalog}.{schema}.{model_name}")
