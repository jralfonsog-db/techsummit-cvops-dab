import dlt
import pyspark.sql.functions as f

catalog = spark.conf.get("pipelines.catalog")
schema = spark.conf.get("pipelines.schema")


@dlt.table(
    comment=f"Raw images from the PCB dataset, ingested from /Volumes/{catalog}/{schema}/landing/Images/"
)
@dlt.expect("No null images", "content is not null")
def pcb_images():
    pcb_images_path = f"/Volumes/{catalog}/{schema}/landing/Images/"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.JPG")
        .option("recursiveFileLookup", "true")
        .load(pcb_images_path)
        .withColumn("filename", f.substring_index(f.col("path"), "/", -1))
    )


@dlt.table(
    comment=f"Labels from the PCB dataset, ingested from /Volumes/{catalog}/{schema}/landing/labels/"
)
@dlt.expect("No null labels", "labelDetail is not null")
def pcb_labels():
    pcb_labels_path = f"/Volumes/{catalog}/{schema}/landing/labels/"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", True)
        .load(pcb_labels_path)
        .withColumn("filename", f.substring_index(f.col("image"), "/", -1))
        .select("filename", "label")
        .withColumnRenamed("label", "labelDetail")
    )
