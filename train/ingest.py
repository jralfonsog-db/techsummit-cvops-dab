from os import path

import dlt
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace


@dlt.table
@dlt.expect("No null links", "link is not null")
def pcb_images():
    csv_path = "dbfs:/data-asset-bundles-dais2023/fe_medium_posts_raw.csv"
    return spark.read.csv(csv_path, header=True)
  
@dlt.table
@dlt.expect("No null links", "link is not null")
def pcb_labels():
    csv_path = "dbfs:/data-asset-bundles-dais2023/fe_medium_posts_raw.csv"
    return spark.read.csv(csv_path, header=True)
  
@dlt.table
def training_dataset():
    pcb_images = dlt.read("pcb_images")
    pcb_labels = dlt.read("pcb_labels")
    df = pcb_images
    return df
  
def training_dataset_augmented():
  df = None
  return df

def delta_torch_dataset():
  df = None
  return df