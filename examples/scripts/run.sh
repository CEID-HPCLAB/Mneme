#!/bin/bash

python3 main.py --training_file "/path/to/dataset.csv" --target_label_name "y0"\
                -nw 64 -nb 600 -niow 2 -nr 17978709 -bcache "/path/to/offsets_file.dat"
