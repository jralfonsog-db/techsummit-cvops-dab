from os import path

import dlt
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, col
import pyspark.sql.functions as F

# catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
# schema = spark.sql("SELECT current_schema()").collect()[0][0]

# catalog = "cvops-test"
# schema = "pcb"

catalog = spark.conf.get("pipelines.catalog")
schema = spark.conf.get("pipelines.schema")

@dlt.table
@dlt.expect("No null images", "content is not null")
def pcb_images():
    pcb_images_path = f"/Volumes/{catalog}/{schema}/landing/Images/"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.JPG")
        .option("recursiveFileLookup", "true")
        .load(pcb_images_path)
        .withColumn("filename", F.substring_index(col("path"), "/", -1))
    )

@dlt.table
@dlt.expect("No null labels", "labelDetail is not null")
def pcb_labels():
    csv_path = f"/Volumes/{catalog}/{schema}/landing/labels/"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", True)
        .load(f"/Volumes/cvops/pcb/landing/labels/")
        .withColumn("filename", F.substring_index(col("image"), "/", -1))
        .select("filename", "label")
        .withColumnRenamed("label", "labelDetail")
    )