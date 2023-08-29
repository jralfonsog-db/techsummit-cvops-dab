# Databricks notebook source
# MAGIC %md ## Top Medium Posts by Databricks Field Engineering

# COMMAND ----------

# MAGIC %md ### Read Data

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from PIL import Image
import io

IMAGE_RESIZE = 256


@pandas_udf("binary")
def resize_image_udf(content_series):
    def resize_image(content):
        """resize image and serialize back as jpeg"""
        # Load the PIL image
        image = Image.open(io.BytesIO(content))
        width, height = image.size  # Get dimensions
        new_size = min(width, height)
        # Crop the center of the image
        image = image.crop(
            ((width - new_size) / 2, (height - new_size) / 2, (width + new_size) / 2, (height + new_size) / 2))
        # Resize to the new resolution
        image = image.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
        # Save back as jpeg
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return output.getvalue()

    return content_series.apply(resize_image)


@pandas_udf("binary")
def flip_image_horizontal_udf(content_series):
    def flip_image(content):
        """resize image and serialize back as jpeg"""
        # Load the PIL image
        image = Image.open(io.BytesIO(content))
        # Flip
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        # Save back as jpeg
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return output.getvalue()

    return content_series.apply(flip_image)


def training_dataset_augmented():
    # add the metadata to enable the image preview
    image_meta = {"spark.contentAnnotation": '{"mimeType": "image/jpeg"}'}

    crop_resize_dataset = (
        dlt.read("training_dataset")
        .withColumn("sort", f.rand()).orderBy("sort").drop('sort')  # shuffle the DF
        .withColumn("content", resize_image_udf(f.col("content")).alias("content", metadata=image_meta))
    )

    flip_damaged = (
        crop_resize_dataset
        .filter("label == 'damaged'")
        .withColumn("content",
                    flip_image_horizontal_udf(f.col("content")).alias("content", metadata=image_meta))
    )

    return crop_resize_dataset.union(flip_damaged)
