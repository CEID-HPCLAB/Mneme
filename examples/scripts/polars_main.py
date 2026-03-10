import os

import polars as pl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from time import perf_counter as time
import subprocess
from math import ceil
import argparse
from time import sleep
import numpy as np
import resource

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-t","--threads", help = "number of threads that polar spawns", default = 2, type = int)
    args = parser.parse_args()

    DS = "/path/to/file.csv"
    THREADS = args.threads
    ROWS = 64000001 # total number of samples
    NUM_BLOCKS = 600
    BLOCK_SIZE = ceil(int(ROWS) / NUM_BLOCKS)

    os.environ["POLARS_MAX_THREADS"] = str(THREADS)
    
    num_idxs = [f"x{i}" for i in range(700)]
    preprocessors = [(StandardScaler(), num_idxs[:350]), (MinMaxScaler(), num_idxs[350:])]

    data = pl.read_csv_batched(DS, n_threads = THREADS, batch_size = BLOCK_SIZE)    

    unique_classes = {"y0": set()}

    batches = data.next_batches(THREADS)
    
    while batches:
        chunk = pl.concat(batches)
        for scaler, idxs in preprocessors:
            scaler.partial_fit(chunk[idxs])
        
        uniques = chunk["y0"].unique().to_list()
        unique_classes["y0"].update(uniques)
            
        batches = data.next_batches(THREADS)

if __name__ == '__main__':
    main()