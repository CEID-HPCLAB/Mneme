from dask.dataframe import read_csv
from dask.distributed import Client, LocalCluster
from dask_ml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import dask
from time import perf_counter as time
import os 
from math import ceil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "-c","--cores", help="number of cores that dask uses", default = 2, type = int)
parser.add_argument( "-t","--threads", help="number of threads per worker that dask spawns", default = 1, type = int)
parser.add_argument( "-ml","--memory_limit", help="the maximum amount of memory (in GB) that each worker can utilize during processing", default = "1.6", type = str)
args = parser.parse_args()

CORES  = args.cores; THREADS = args.threads; NUM_BLOCKS = 600
DS = "/path/to/file.csv"

file_size = os.path.getsize(DS)
BLOCK_SIZE = ceil(file_size / NUM_BLOCKS)

def main():
    cluster = LocalCluster(n_workers = CORES, threads_per_worker = THREADS)
    
    client = Client(cluster)
    
    num_idxs = [f"x{i}" for i in range(700)]
    cat_idxs = ["y0"]
    preprocessors = [(StandardScaler(), num_idxs[:350]), (MinMaxScaler(), num_idxs[350:]), (LabelEncoder(), cat_idxs)]
    
    dd_df = read_csv(DS, blocksize = BLOCK_SIZE)
    
    for preprocessor, features in preprocessors:
        preprocessor.fit(dd_df[features])

    client.close()

if __name__ == "__main__":
    main()
