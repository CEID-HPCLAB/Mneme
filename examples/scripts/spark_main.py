import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    MinMaxScaler,
    StringIndexer
)
from pyspark.ml import Pipeline
from pyspark import StorageLevel
from time import perf_counter as time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "-c","--cores", help="number of cores that dask uses", default = 32, type = int)
args = parser.parse_args()

PROCS  = args.cores; DATASET = "/path/to/file.csv"
file_size = os.path.getsize(DS)
BLOCK_SIZE = ceil(file_size / NUM_BLOCKS)

spark = (
    SparkSession.builder
    .master(f"local[{PROCS}]")
    .appName("SparkPreprocessingPipeline")
    .config("spark.sql.files.maxPartitionBytes", BLOCK_SIZE)
    .config("spark.sql.shuffle.partitions", PROCS)
    .getOrCreate()
)

df = (spark.read.option("header", True).option("inferSchema", True).csv(DATASET))

cols_low  = [f"x{i}" for i in range(350)]
cols_high = [f"x{i}" for i in range(350, 700)]

assembler_high = VectorAssembler(inputCols = cols_high, outputCol = "features_high")
assembler_low = VectorAssembler(inputCols = cols_low, outputCol = "features_low")

std_scaler = StandardScaler(inputCol = "features_low", outputCol = "std_features", withMean = True, withStd = True)

min_max_scaler = MinMaxScaler(inputCol = "features_high", outputCol = "mm_features")

label_indexer = StringIndexer(inputCol = "y0", outputCol = "label", stringOrderType = "alphabetDesc")

pipeline = Pipeline(stages = [assembler_low, assembler_high, std_scaler, min_max_scaler, label_indexer])
fitted_pipeline = pipeline.fit(df)

spark.stop()