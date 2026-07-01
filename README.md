# 🚀 Mneme: A Parallel Preprocessing Framework for Large Tabular Datasets

[![Python 3.10](https://img.shields.io/badge/Python-3.10-purple?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) 
![HPC](https://img.shields.io/badge/HPC-228B22?style=flat&logo=dna&logoColor=white)
![Data Preprocessing](https://img.shields.io/badge/Data%20Preprocessing-001594?style=flat&logo=dna&logoColor=white)

**Mneme** is a CPU-based high-level framework designed for high-performance out-of-core preprocessing of large-scale tabular datasets on single-node systems. Developed as a Python library, Mneme optimizes resource consumption by combining memory-efficient data loading with parallel computation techniques, allowing the preprocessing tasks to be executed concurrently. The library supports various data transformations, including normalization, categorical encoding, and missing value imputation, providing a robust solution for ML/DL pipelines in resource-constrained environments. 

> [!NOTE]
> For a detailed description of the Mneme framework, we refer the reader to the [original paper](https://link.springer.com/article/10.1007/s10766-026-00817-7) published in the *International Journal of Parallel Programming (IJPP)*. An early version of Mneme was also presented at the [18th International Symposium on High-level Parallel Programming and Applications (HLPP 2025)](https://hlpp-conference.github.io/hlpp-2025/), Innsbruck, Austria, 3-4 July 2025.

## Table of Contents
- [Installation](#installation)
- [Mneme](#mneme)
  - [Parallelism Scheme](#parallelism-scheme)
  - [API](#api)
  - [Demo Example](#demo-example)
  - [Integration into PyTorch workflows](#integration-into-pytorch-workflows)
  - [Unit Testing](#unit-testing)
- [Performance Evaluation](#performance-evaluation)
  - [Comparison with Dask, Spark and Polars](#comparison-with-dask-spark-and-polars)
  - [Comparison with NVTabular](#comparison-with-nvtabular)
- [File Structure](#file-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

Clone the repository:
```bash
git clone https://github.com/CEID-HPCLAB/Mneme
cd Mneme
```

Install the **Mneme** package and external dependencies:
```bash
# (Optional) Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate # POSIX (bash/zsh)

pip install .
```

## Mneme

### Parallelism Scheme
Mneme relies on a hybrid task-based parallelism scheme, inspired by the **MapReduce** paradigm, which combines **multiprocessing** and **multithreading**. The coordination and management of processes are accomplished using Python’s `multiprocessing` package, via a **process pool programming pattern** that incorporates adaptive task scheduling, akin to the dynamic scheduling policy of *OpenMP*. To circumvent the limitations imposed by the *Global Interpreter Lock (GIL)*, **Rust threads** are integrated via the CSV reading capabilities of the [Polars](https://pola.rs/) library.

### API
Built on top of the [scikit-learn](https://scikit-learn.org/stable/) ecosystem, Mneme adopts the widely used `fit()-transform()` API model, offering a simple, high-level, and intuitive interface. It provides a wide range of data preprocessing techniques within its `preprocessing` package, following a structure [similar to that of scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html). Specifically, the operators provided by Mneme are summarized in the following table.

| Operator         | Description |
|-----------------|-------------|
| [StandardScaler](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelstandardscaler.py)   | Z-score normalization |
| [MinMaxScaler](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelminmaxscaler.py)     | Scaling in a range specified by min and max values |
| [MaxAbsScaler](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelmaxabsscaler.py)     | Scaling in range [-1, 1] |
| [RobustScaler](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelrobustscaler.py)     | Median centering and interquartile range (IQR) scaling |
| [OneHotEncoder](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelonehotencoder.py)    | Binary transformation for nominal categorical features |
| [OrdinalEncoder](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelordinalencoder.py)   | Encoding of hierarchical categorical features |
| [LabelEncoder](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallellabelencoder.py)     | Encoding of categorical target variables |

Mneme also provides two pipeline structures: the **Imputation Pipeline** ([ParImputer](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelimputer.py)) and the **Preprocessing Pipeline** ([ParallelPipeline](https://github.com/CEID-HPCLAB/Mneme/blob/main/src/Mneme/preprocessing/parallelpreprocessor.py)). The Imputation Pipeline combines multiple `SimpleImputer` instances, each employing a different strategy on a distinct subset of features for handling missing values. The Preprocessing Pipeline enables the integration of various operators, as listed in the table above.

### Demo Example

The code listing below shows a representative example of applying Z-Score Normalization to 700 features of a dataset using the corresponding operator in Mneme.

Initially, a `BlockReader` component is employed to partition the dataset into blocks (**virtual files**). The path to a binary block offset file is provided, specifying where the computed offsets will be stored (`block_offset_save`). This file can be reused in subsequent runs, enabling faster preprocessing pipeline execution without recomputing the block offsets. 

Next, a StandardScaler (`ParStandardScaler`) is instantiated and fitted to the specified 700 input features (`num_idxs`). The fitting process is performed using four processes (`workers`), each employing two threads (IO_threads) for block fetching. The result is a fitted StandardScaler, ready to be used in subsequent pipeline stages for *on the fly* data transformation during batch-level fetching. 

By instantiating the corresponding operator object, the same procedure can be used to fit any standalone operator listed in the table above.

```python
from Mneme import BlockReader
from Mneme.preprocessing import ParStandardScaler

datafile = "/path/to/data.csv"

# dataset shape: 10M rows, 701 features (x0, x1, ..., x699, y0)
num_idxs = [f"x{i}" for i in range(700)]

workers = 4; IO_threads = 2; n_blocks = 100; offset_file = "./offsets.dat"

block_reader = BlockReader(datafile, num_blocks = n_blocks, block_offset_save = offset_file)

standard_scaler = ParStandardScaler(datafile, num_idxs = num_idxs)
standard_scaler.fit(block_reader = block_reader, num_workers = workers, IO_workers = IO_threads)
```

### Integration into PyTorch Workflows

The following code listing illustrates how Mneme’s preprocessing operators can be integrated with [PyTorch](https://pytorch.org/) during the training of a DNN model on a large-scale tabular dataset. 

Unlike the previous example, the binary block offsets are precomputed, and the path to the corresponding file is provided to the `BlockReader` upon instantiation (`block_offset_cache`). Subsequently, a Mneme preprocessing pipeline (`pipeline`), consisting of a `MinMaxScaler`, a `MaxAbsScaler`, and a `LabelEncoder`, is instantiated and fitted to the specified features. After the completion of the fitting phase, the raw data is transformed *on the fly* in **chunks** (i.e., batches) according to the fitted preprocessing pipeline before being sent to the GPU for training.

```python
import torch; import pandas as pd
from torch.utils.data import DataLoader, IterableDataset
from Mneme import BlockReader
from Mneme.preprocessing import ParallelPipeline, ParMinMaxScaler, ParMaxAbsScaler, ParLabelEncoder

class CSVDataset(IterableDataset):
    def __init__(self, path, pipeline, chunksize = 5000):
        self.path, self.pipeline, self.chunksize = path, pipeline, chunksize
    
    def __iter__(self):
        for chunk in pd.read_csv(self.path, chunksize = self.chunksize):
            # on-the-fly transformation using Mneme's fitted pipeline (self.pipeline)
            chunk = self.pipeline.transform(chunk)
            X, y = chunk[:, :-1], chunk[:, -1]
            for xi, yi in zip(X, y):
                yield torch.tensor(xi), torch.tensor(yi)

datafile = "/path/to/data.csv"

# dataset shape: 50M rows, 201 features (x0, x1, ..., x199, y0)
num_idxs = [f"x{i}" for i in range(200)]; cat_idxs = ["y0"]

workers = 4; IO_threads = 2; batch_size = 256; offset_file = "./offsets.dat"

block_reader = BlockReader(datafile, block_offset_cache = offset_file)

pipeline = ParallelPipeline({"InputFeatures": [        
                                            ParMinMaxScaler(num_idxs = num_idxs[:50]),   
                                            ParMaxAbsScaler(num_idxs = num_idxs[50:])],
                             "TargetVar": [ParLabelEncoder(cat_idxs = cat_idxs)]
                            },
                            datafile) # Mneme preprocessing pipeline

pipeline.fit(block_reader = block_reader, num_workers = workers, IO_workers = IO_threads)

model = DNN().to(torch.device("cuda")) # assume DNN: PyTorch (torch.nn) model class
dataset = CSVDataset(datafile, *cat_idxs, pipeline) # iterable-style PyTorch Dataset
dataloader = DataLoader(dataset, batch_size = batch_size) # PyTorch DataLoader

# assume train_loop() sets the training parameters (e.g., optimizer, criterion, epochs), and trains the model using the provided DataLoader
model_performance = train_loop(model, dataloader)
```

### Unit Testing

Mneme was extensively tested using the [pytest](https://docs.pytest.org/en/stable/) framework. For each operator, the statistics of the parallel fitting process were compared to those of the corresponding sequential version to ensure that the parallel execution did not alter the statistical properties of the data, wiping out any potential race conditions, hence verifying the correctness and reliability of the parallel implementations.


## Perfomance Evaluation

The performance of Mneme was evaluated and compared with state-of-the-art libraries in typical preprocessing workflow scenarios for ML/DL pipelines. The libraries chosen for comparing Mneme’s performance were [Dask-ML](https://ml.dask.org/) (Dask), [MLlib](https://spark.apache.org/mllib/) (Apache Spark), [Polars](https://pola.rs/), [Modin](https://modin.org/) (utilizing the Dask execution engine), and [NVTabular](https://github.com/nvidia-merlin/nvtabular) (NVIDIA Merlin). Experiments were carried out on two diverse computing systems, using five distinct tabular datasets: two synthetic datasets (126 GB, 117 GB) and three publicly available real-world datasets (101 GB, 50 GB, 151 GB) [[1](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), [2](https://www.kaggle.com/competitions/amex-default-prediction), [3](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)]. For brevity, only the results obtained for three of the five datasets on an NVIDIA DGX A100 system are presented below, unless otherwise stated. For the complete experimental evaluation, we refer the reader to the published [paper](https://link.springer.com/article/10.1007/s10766-026-00817-7).


> [!WARNING]
> To simulate the out-of-core execution scenario, the available physical memory of the server was restricted to 32 GB throughout all experiments.

> [!NOTE]
> Performance graphs also include results from an alternative Mneme implementation (*Mneme MP*) that uses native Pandas instead of Polars for block loading. This version differs from the standard implementation, as it omits multithreading during data loading.

> [!NOTE]
> Performance graphs do not include results from Modin, as we observed that, in alignment with [prior benchmarks](https://github.com/pola-rs/polars-benchmark), Modin struggles to support out-of-core processing (DNF) in resource-constrained environments.

### Comparison with Dask, Spark and Polars

As shown in Figure 1, Mneme consistently outperforms Dask and Spark. In particular, across both datasets, Mneme achieves at least **4×** speedup over Dask and Spark for all worker configurations.

Furthermore, Mneme demonstrates *better scalability* compared to Polars. Across both datasets, Mneme maintains parallel efficiency above **73%** for configurations with fewer than 16 workers, whereas Polars drops to 62% for the 126 GB dataset and 49% for the 101 GB dataset at 8 workers.

<br>
<p align="center">
  <picture>
    <source srcset = "figs/fig_dark.png" media = "(prefers-color-scheme: dark)">
    <source srcset = "figs/fig_light.png" media = "(prefers-color-scheme: light)">
    <img src = "figs/fig_light.png" width = "95%" alt = "Runtime performance (left: 126 GB, right: 101 GB)">
  </picture>
  <br>
  <em>Figure 1: Runtime performance on two datasets (left: 126 GB, right: 101 GB)</em>
</p>

### Comparison with NVTabular

Regarding the comparison of Mneme with the GPU-based library NVTabular, a 151 GB subset of the [Criteo dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) was used and the experiments were conducted on a general-purpose workstation (AMD Ryzen 9 7950X, 32 GB RAM, NVIDIA GeForce RTX 4070 Ti with 12 GB VRAM). Mneme was executed with eight processes, while NVTabular utilized the system’s NVIDIA GeForce RTX 4070 Ti GPU.

As shown in the table below, Mneme achieves up to **1.87×** speedup over NVTabular.

| Library    | Time (s) |
|-----------|----------|
| Mneme     | 143      |
| NVTabular | 267      |



## File Structure
```
Mneme/
├── examples/
│ ├── data/ # Scripts for generating synthetic tabular datasets
│ └── scripts/ 
|     # Scripts for running Mneme and the three CPU-based baseline libraries used for comparison.
│     # Also includes an example script demonstrating how Mneme can be integrated into an Optuna
│     # workflow for hyperparameter optimization of a PyTorch DNN pipeline.
│
├── src/ # Core implementation of Mneme
├── tests/ # Unit tests for core Mneme components (pytest)
│
├── pyproject.toml # Build system and project metadata configuration
├── requirements.txt # Python dependencies
├── setup.py # Installation script (setuptools-based)
```


## Citation
If you find Mneme useful for your research, please cite:

```bibtex
@article{Sofotasios2026,
  author  = {Sofotasios, Argiris and Metaxakis, Dimitris and Hadjidoukas, Panagiotis},
  title   = {Mneme: A Parallel Preprocessing Framework for Large Tabular Datasets},
  journal = {International Journal of Parallel Programming},
  year    = {2026},
  volume  = {54},
  number  = {4},
  pages   = {20},
  doi     = {10.1007/s10766-026-00817-7},
  url     = {https://doi.org/10.1007/s10766-026-00817-7}
}
```


## Acknowledgments
This research was carried out as part of [easyhpc@eco.plastics.industry](https://european-digital-innovation-hubs.ec.europa.eu/edih-catalogue/easyhpc) (MIS: 6001593), which is co-funded by the European Union under the Competitiveness Program (ESPA 2021–2027).    


## Contact
For questions, bug reports, or contributions, please open an issue or contact:

Argiris Sofotasios  
a.sofotasios@ac.upatras.com