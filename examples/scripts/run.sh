#!/bin/bash

python3 main.py --training_file "/home/hpclab01/Projects/tabular_datasets/train_data_120M.csv" --target_label_name "y0"\
                -nw 64 -nb 600 -niow 2 -nr 17978709 -bcache "/home/hpclab01/Projects/tabular_datasets/120M_offsets_600bl.dat"
