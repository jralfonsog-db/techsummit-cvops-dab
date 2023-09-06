# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.sql import Window

# COMMAND ----------

dbutils.widgets.text("catalog", "catalog")
dbutils.widgets.text("schema", "schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

images_bronze_table = "images"
labels_bronze_table = "labels"
training_dataset_table = "training_dataset"
training_dataset_augmented_table = "training_dataset_augmented"
train_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/train"
test_deltatorch_path = f"/Volumes/{catalog}/{schema}/deltatorch_files/test"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# These lines here are meant to remove tables for demo purposes

spark.sql(f"DROP TABLE IF EXISTS {training_dataset_table}")
spark.sql(f"DROP TABLE IF EXISTS {training_dataset_augmented_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Join labels and images

# COMMAND ----------

images = spark.read.table(images_bronze_table)
labels = spark.read.table(labels_bronze_table)

# COMMAND ----------

training_dataset = labels.withColumn(
    "label",
    f.when(labels["labelDetail"] == "normal", "normal").otherwise("damaged"),
).join(images, how="inner", on="filename")

training_dataset.write.saveAsTable(training_dataset_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Data augmentation
# MAGIC
# MAGIC We will create two Pandas UDF to augment our dataset:
# MAGIC - Crop and resize
# MAGIC - Rotate damaged images

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## UDFs

# COMMAND ----------

# DBTITLE 1,Crop and resize our images
from PIL import Image
import io
from pyspark.sql.functions import pandas_udf

IMAGE_RESIZE = 256

#Resize UDF function
@pandas_udf("binary")
def resize_image_udf(content_series):
  def resize_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    width, height = image.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the image
    image = image.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    image = image.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(resize_image)

# COMMAND ----------

# DBTITLE 1,Flip and add damaged images
import PIL
@pandas_udf("binary")
def flip_image_horizontal_udf(content_series):
  def flip_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    #Flip
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(flip_image)

# COMMAND ----------

# add the metadata to enable the image preview
image_meta = {"spark.contentAnnotation" : '{"mimeType": "image/jpeg"}'}

(
    spark.table(training_dataset_table)
    .withColumn("sort", f.rand())
    .orderBy("sort")
    .drop("sort")  # shuffle the DF
    .withColumn(
        "content",
        resize_image_udf(f.col("content")).alias("content", metadata=image_meta),
    )
    .write.mode("overwrite")
    .saveAsTable(training_dataset_augmented_table)
)

# COMMAND ----------

(
    spark.table(training_dataset_augmented_table)
    .filter("label == 'damaged'")
    .withColumn(
        "content",
        flip_image_horizontal_udf(f.col("content")).alias(
            "content", metadata=image_meta
        ),
    )
    .write.mode("append")
    .saveAsTable(training_dataset_augmented_table)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation for Delta Torch

# COMMAND ----------

training_dataset_augmented = spark.table(training_dataset_augmented_table)

# COMMAND ----------

train_test = (
  training_dataset_augmented
  .withColumnRenamed("label", "labelName")
  .withColumn("label", f.when(f.col("labelName").eqNullSafe("normal"), 0).otherwise(1))
)

train, test = train_test.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

w = Window().orderBy(f.rand())

train.withColumn("id", f.row_number().over(w)).write.mode("overwrite").save(train_deltatorch_path)
test.withColumn("id", f.row_number().over(w)).write.mode("overwrite").save(test_deltatorch_path)
