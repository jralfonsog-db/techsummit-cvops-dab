import dlt
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f

catalog = spark.conf.get("pipelines.catalog")
schema = spark.conf.get("pipelines.schema")


@dlt.table(
    comment=f"Training dataset without augmentation"
)
def training_dataset():
    pcb_images: DataFrame = dlt.read("pcb_images")
    pcb_labels: DataFrame = dlt.read("pcb_labels")

    return pcb_labels.withColumn(
        "label",
        f.when(pcb_labels["labelDetail"] == "normal", "normal").otherwise("damaged"),
    ).join(pcb_images, how="inner", on="filename")


@dlt.table(comment="Training dataset with random splits to be filtered after")
def training_dataset_with_splits():
    return (
        dlt.read("training_dataset")
        .withColumnRenamed("label", "labelName")
        .withColumn("label", f.when(f.col("labelName").eqNullSafe("normal"), 0).otherwise(1))
        .withColumn("isTrain", f.when(f.rand() < 0.7, 1).otherwise(0))
    )


w = Window().orderBy(f.rand())


@dlt.table(comment="Train dataset prepared for DeltaTorch")
def train_deltatorch():
    return (
        dlt.read("training_dataset_with_splits")
        .where(f.col("isTrain") == 1)
        .drop("isTrain")
        .withColumn("id", f.row_number().over(w))
    )


@dlt.table(comment="Test dataset prepared for DeltaTorch")
def test_deltatorch():
    return (
        dlt.read("training_dataset_with_splits")
        .where(f.col("isTrain") == 0)
        .drop("isTrain")
        .withColumn("id", f.row_number().over(w))
    )
