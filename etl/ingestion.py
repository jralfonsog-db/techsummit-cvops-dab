# Databricks notebook source
import pyspark.sql.functions as f

# COMMAND ----------

dbutils.widgets.text("catalog", "catalog")
dbutils.widgets.text("schema", "schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

images_path = f"/Volumes/{catalog}/{schema}/landing/Images/"
images_schema = f"/Volumes/{catalog}/{schema}/landing/stream/images_schema"
images_checkpoint = f"/Volumes/{catalog}/{schema}/landing/stream/images_checkpoint"
images_table = "images"

labels_path = f"/Volumes/{catalog}/{schema}/landing/labels/"
labels_schema = f"/Volumes/{catalog}/{schema}/landing/stream/labels_schema"
labels_checkpoint = f"/Volumes/{catalog}/{schema}/landing/stream/labels_checkpoint"
labels_table = "labels"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {images_table}")
spark.sql(f"DROP TABLE IF EXISTS {labels_table}")

# COMMAND ----------

# These lines here are meant to remove autoloader checkpoints for demo purposes

dbutils.fs.rm(images_checkpoint, True)
dbutils.fs.rm(labels_checkpoint, True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Image and label ingestion
# MAGIC
# MAGIC The following code ingests images and labels which land in our landing Volume

# COMMAND ----------

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.JPG")
    .option("recursiveFileLookup", "true")
    .option("cloudFiles.schemaLocation", images_schema)
    .load(images_path)
    .withColumn("filename", f.substring_index(f.col("path"), "/", -1))
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", images_checkpoint)
    .toTable(images_table)
    .awaitTermination()
)

# COMMAND ----------

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", True)
    .option("cloudFiles.schemaLocation", labels_schema)
    .load(f"/Volumes/cvops/pcb/landing/labels/")
    .withColumn("filename", f.substring_index(f.col("image"), "/", -1))
    .select("filename", "label")
    .withColumnRenamed("label", "labelDetail")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", labels_checkpoint)
    .toTable(labels_table)
    .awaitTermination()
)
