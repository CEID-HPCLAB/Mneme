import argparse
from time import perf_counter as time
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler,
                                    MinMaxScaler, 
                                    MaxAbsScaler, 
                                    RobustScaler,
                                    LabelEncoder,
                                    OneHotEncoder,
                                    OrdinalEncoder)
from Mneme import LogLevel
from Mneme import BlockReader
from Mneme.preprocessing import (ParStandardScaler,
                                  ParMinMaxScaler,
                                  ParRobustScaler,
                                  ParMaxAbsScaler,
                                  ParMinMaxScaler,
                                  ParPreprocessor,
                                  ParLabelEncoder,
                                  ParOneHotEncoder,
                                  ParOrdinalEncoder,
                                  ParImputer)
from Mneme.utils import (reduce_std_scalers, reduce_minmax_scalers, 
                         reduce_maxabs_scalers, reduce_robust_scalers, 
                         reduce_label_encoders, reduce_ordinal_encoders,
                         reduce_onehot_encoders, reduce_imputers as reduce_fitted_imputers, set_log_level) 
from copy import deepcopy
import os


def str2bool(v: Union[str, bool]) -> bool:
    '''
    Converts a string to a boolean value.
    '''
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
    raise argparse.ArgumentTypeError('Boolean value expected!')


def fit_preprocessors_large_ds(data: pd.DataFrame, scalers: List[Tuple[Union[StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler], List[str]]] = [], 
                               encoders: List[Tuple[Union[LabelEncoder, OrdinalEncoder, OneHotEncoder], List[str]]] = [], 
                               print_stats: bool = False) -> None:
    '''
    Fits scalers and encoders on a large dataset sequentially (chunk by chunk).

    This function takes a large dataset and fits specified scalers and encoders on it. The fitting is done 
    sequentially, chunk by chunk, to accommodate large datasets that may not fit into memory. 

    Args:
        data (pd.DataFrame): The large dataset to fit the preprocessors on.
        scalers (List[Tuple[[Union[StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler], List[str]]], optional): List of tuples (scaler objects, features) to fit. Defaults to [].
        encoders (List[Tuple[Union[LabelEncoder, OrdinalEncoder, OneHotEncoder], List[str]], optional): List of tuples (encoder objects, features) to fit. Defaults to [].
        print_stats (bool, optional): If True, prints the computed statistics of the fitted preprocessors. Defaults to False.

    Returns:
        None
    '''
    
    partial_fits = [[] for _ in range(len(scalers + encoders))]
    
    total_fit_time = 0.0
    excluded_fit_time = 0.0

    start_fitting = time()
    # data = [data]
    
    label_encoder_indices = None

    for chunk in data:
        
        chunk_fit_start = time()
        scalers = [deepcopy(scaler) for scaler in scalers]
        encoders = [deepcopy(encoder) for encoder in encoders]
        
        # Fit each scaler - encoder and store it in the partial_fits list
        for index, scaler in enumerate(scalers):
            scaler, indices = scaler
            scaler_fit_start = time()
            scaler.fit(chunk[indices])
            scaler_fit_end = time()
            
            total_fit_time += scaler_fit_end - scaler_fit_start
            
            partial_fits[index].append(scaler)
            
        for index, encoder in enumerate(encoders, start = len(scalers)):
            encoder, indices = encoder
            if isinstance(encoder, LabelEncoder):
                label_encoder_indices = indices
                for cat_idx in indices:
                    chunk_reshaped = chunk[cat_idx].to_numpy().ravel()
                    encoder = deepcopy(encoder)
                    
                    encoder_fit_start = time()
                    encoder.fit(chunk_reshaped)
                    encoder_fit_end = time()
                    
                    total_fit_time += encoder_fit_end - encoder_fit_start
                    partial_fits[index].append(encoder)
                    
            else:
                
                encoder_fit_start = time()
                encoder.fit(chunk[indices].to_numpy())
                encoder_fit_end = time()
                
                total_fit_time += encoder_fit_end - encoder_fit_start
                partial_fits[index].append(encoder)
        
        chunk_fit_end = time()
        excluded_fit_time += chunk_fit_end - chunk_fit_start
    
    end_fitting = time()
    total_fit_time += end_fitting - start_fitting - excluded_fit_time 
    
    start_reduce = time()
    
    # Reduce the partial fitted scalers and encoders
    reduce_preprocessors(partial_fits, label_encoder_indices, print_stats)
    
    end_reduce = time()
    print(f"Sequential Times of Preprocessing Pipeline -> Fit: {total_fit_time:.6f} | Reduce: {end_reduce - start_reduce:.6f}")
    

def reduce_preprocessors(partial_fits: List[List], cat_idxs: List[str], print_stats: bool) -> None:
    '''
    Reduces the partial fitted preprocessors (scalers and encoders).

    This function takes a list of partially fitted preprocessors and reduces them to a single preprocessor 
    for each (preprocessor) type.
    
    Args:
        partial_fits (List[List]): A nested list where each inner list contains instances of a specific preprocessor that 
        have been partially fitted on different chunks of the input dataset.
        cat_idxs (list): List of categorical feature indices.
        print_stats (bool): If True, prints the attributes of the reduced preprocessors.

    Returns:
        None
    
    '''
    
    for preprocessors in partial_fits:
        # For each type, reduce the partial fitted preprocessors and print the attributes if required
        match preprocessors[0]:
           
            case StandardScaler():
                std_scaler_red = reduce_std_scalers(preprocessors)
                if print_stats:
                    for attr in vars(std_scaler_red):
                        print(f"{type(std_scaler_red).__name__ }.{attr} = { getattr(std_scaler_red, attr)}")
                    
            case MinMaxScaler():
                minmax_scaler_red = reduce_minmax_scalers(preprocessors)
                if print_stats:
                    for attr in vars(minmax_scaler_red):
                        print(f"{type(minmax_scaler_red).__name__ }.{attr} = { getattr(minmax_scaler_red, attr)}")
                    
            case MaxAbsScaler():
                maxabs_scaler_red = reduce_maxabs_scalers(preprocessors)
                if print_stats:
                    for attr in vars(maxabs_scaler_red):
                        print(f"{type(maxabs_scaler_red).__name__ }.{attr} = { getattr(maxabs_scaler_red, attr)}")
                     
            case OneHotEncoder():
                onehot_enc_red = reduce_onehot_encoders(preprocessors)
                if print_stats:
                    for attr in vars(onehot_enc_red):
                        print(f"{type(onehot_enc_red).__name__ }.{attr} = { getattr(onehot_enc_red, attr)}")
                
            case RobustScaler():
                robust_scaler_red = reduce_robust_scalers(preprocessors)
                if print_stats:
                    for attr in vars(robust_scaler_red):
                        print(f"{type(robust_scaler_red).__name__ }.{attr} = { getattr(robust_scaler_red, attr)}")

            case LabelEncoder():
                label_encs = [preprocessors[i:len(preprocessors):len(cat_idxs)] for i in range(0, len(cat_idxs))]
                for i, _ in enumerate(cat_idxs):
                    enc = reduce_label_encoders(label_encs[i])
                    if print_stats:
                        for attr in vars(enc):
                            print(f"{type(enc).__name__ }.{attr} [Feature: {_}] = { getattr(enc, attr)}")
            
            case OrdinalEncoder():
                ordinal_enc_red = reduce_ordinal_encoders(preprocessors)
                if print_stats:
                    for attr in vars(ordinal_enc_red):
                        print(f"{type(ordinal_enc_red).__name__ }.{attr} = { getattr(ordinal_enc_red, attr)}")
                
            case _:
                print(f"Fault! Didn't match preprocessor. Shouldn't happen!")
    
    
def fit_imputers_large_ds(data: pd.DataFrame, imputers_map: Dict[SimpleImputer, List[str]], print_stats: bool = False) -> None:
    '''
    Fits imputers on large datasets chunk by chunk.

    This function takes a large dataset and fits specified imputers on it. The fitting is done 
    sequentially, chunk by chunk, to accommodate large datasets that may not fit into memory. 

    Args:
        data (pd.DataFrame): The DataFrame to fit the imputers on.
        imputers_map (Dict[SimpleImputer, List[str]]): A mapping of imputers to the indices of the columns they should be fitted on.
        print_stats (bool, optional): If True, prints the computed statistics of the fitted imputers. Defaults to False.

    Returns:
        None
    '''
    
    partial_fits = [[] for _ in range(len(imputers_map))]
    
    total_fit_time = 0.0
    excluded_fit_time = 0.0

    start_fitting = time()
    indices = list(imputers_map.values())
    
    # Iterate over each chunk in the data
    for chunk in data:
        
        chunk_fit_start = time()
        imputers = [deepcopy(imputer) for imputer in imputers_map.keys()]
        
        # Fit each imputer and store it in the partial_fits list
        for index, imputer in enumerate(imputers):
            
            imputer_fit_start = time()
            imputer.fit(chunk[indices[index]].to_numpy())
            imputer_fit_end = time()
            
            imputer.n_samples_seen_ = chunk[indices[index]].shape[0]
            if imputer.strategy == 'mean':
                imputer.nan_vals = np.isnan(chunk[indices[index]]).sum()
    
            total_fit_time += imputer_fit_end - imputer_fit_start
            
            partial_fits[index].append(imputer)
            
        chunk_fit_end = time()
        excluded_fit_time += chunk_fit_end - chunk_fit_start 
    
    end_fitting = time()
    total_fit_time += end_fitting - start_fitting - excluded_fit_time 
    
    start_reduce = time()
    
    # Reduce the partial fitted imputers
    reduce_imputers(partial_fits, print_stats)
    
    end_reduce = time()
    print(f"Sequential Times of Imputer -> Fit: {total_fit_time:.6f} | Reduce: {end_reduce - start_reduce:.6f}")
    
    
def reduce_imputers(partial_fits: List[List[SimpleImputer]], print_stats: bool) -> None:
    '''
    Reduces the partial fits of each imputer to a single fit.

    This function takes a list of lists of partially fitted imputers and reduces each list to a single fitted imputer. 

    Args:
        partial_fits (List[List[SimpleImputer]]): A list of lists of partially fitted imputers.
        print_stats (bool): If True, prints the attributes of the reduced imputers.

    Returns:
        None
    '''
    
    for index, partial_fit_imp in enumerate(partial_fits):
        
        fitted_imp = reduce_fitted_imputers(imputers = partial_fit_imp, strategy = partial_fit_imp[0].strategy)
        
        if print_stats:
            print(f"SimpleImputer{index}.statistics_: {fitted_imp.statistics_}")
            print(f"SimpleImputer{index}.strategy: {fitted_imp.strategy}")
            print(f"{'='*35}")
   


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_file", help = "training file", required = True) 
    parser.add_argument("--target_label_name", help = "target label name, empty string for last column", type = str, required = True)
    parser.add_argument("--read_chunk_size", help = "read chunk size", default = 1000, type = int, required = False)
    parser.add_argument("-nr", "--n_rows", help = "number of rows", default = -1, type = int, required = False)
    parser.add_argument("--method", help = "get num rows method", default = 4, type = int, required = False)
    parser.add_argument("-nw", "--num_workers", help = "num workers", default = 1, type = int, required = False)
    parser.add_argument("-nb", "--num_blocks", help = "num blocks", default = -1, type = int, required = False)
    parser.add_argument( "-bcache", "--block_offset_cache", help = "precomputed block offsets", default="", type = str, required = False)
    parser.add_argument( "-bsave","--block_offset_save", help = "save block offsets", default = "", type = str, required = False)
    parser.add_argument("-niow","--IO_workers", help = "Threads spawned by polars", default = 1, type = int, required = False)
    args = parser.parse_args()
    
    print(args)

    target_label_name= args.target_label_name.split(",")
    training_file = args.training_file
    read_chunk_size = args.read_chunk_size
    n_rows = args.n_rows
    method = args.method
    num_workers = args.num_workers
    num_blocks = args.num_blocks
    block_offset_cache = args.block_offset_cache
    block_offset_save = args.block_offset_save
    IO_workers = args.IO_workers
    os.environ["POLARS_MAX_THREADS"] = str(IO_workers)

    # setting to BENCHMARK for more detailed logs 
    set_log_level(LogLevel.BENCHMARK)

    ######################################################
    ##  Example Dataset INFO
    ##  700 numerical (input) features (x0, x1, ..., x699)
    ##  1 categorical feature for output variable (y0)
    ######################################################

    # Numerical (input) features
    num_idxs = [f"x{i}" for i in range(700)]
    
    # Target variable
    cat_idxs = ['y0']
    
    block_reader_loader = BlockReader(training_file,
                                       target_label_name, num_blocks = num_blocks, n_rows = n_rows, 
                                       read_chunk_size = read_chunk_size, method = method, 
                                       block_offset_save = block_offset_save, block_offset_cache = block_offset_cache)
    
    
    imputers_map = {
        SimpleImputer(missing_values = np.nan, strategy = "mean") :  num_idxs,
        SimpleImputer(missing_values = np.nan, strategy = "most_frequent") : cat_idxs,
        }
    
    # Parallel Imputing Pipeline
    par_imputer = ParImputer(imputers_map, custom_block_reader = block_reader_loader, IO_workers = IO_workers, num_workers = num_workers)
    par_imputer.parallel_fit()
    # par_imputer.print()
    
    # Sequential Imputer Pipeline
    df = pd.read_csv(training_file, chunksize = block_reader_loader.block_size)
    fit_imputers_large_ds(df, imputers_map, print_stats = False)
    
    # Parallel Preprocessing Pipeline      
    par_preprocessor = ParPreprocessor(preprocessors = {"InputVar": [ParStandardScaler(data_file = training_file, 
                                                                                       num_idxs = num_idxs[:300] ), 
                                                                     ParStandardScaler(data_file = training_file, 
                                                                                       num_idxs = num_idxs[300:450], with_mean = False) ,
                                                                     ParMinMaxScaler(data_file = training_file,
                                                                                       num_idxs = num_idxs[450:600]),
                                                                     ParMaxAbsScaler(data_file = training_file,
                                                                                       num_idxs = num_idxs[600:]),
                                                                    ],
                                                                     
                                                        "TargetVar": [ParLabelEncoder(data_file = training_file,
                                                                                    cat_idxs = cat_idxs),
                                                        ]},
                                       block_reader= block_reader_loader, num_workers = num_workers, 
                                       imputer = par_imputer, IO_workers = IO_workers)
    
    par_preprocessor.parallel_fit()
    # par_preprocessor.print()
    
    # Sequential Preprocessing Pipeline with Scikit-Learn
    df = pd.read_csv(training_file, chunksize = block_reader_loader.block_size)
    scalers = [(StandardScaler(), num_idxs[:300]), (StandardScaler(with_mean = False), num_idxs[300:450]), 
               (MinMaxScaler(), num_idxs[450:600]),(MaxAbsScaler(), num_idxs[600:])]
    encoders = [(LabelEncoder(), cat_idxs)]
    fit_preprocessors_large_ds(data = df, scalers = scalers, encoders = encoders, print_stats = False)
    
    par_maxabs_scaler = ParMaxAbsScaler(data_file = training_file, num_idxs = num_idxs)
    par_maxabs_scaler._fit(use_parallel = True, block_reader = block_reader_loader, num_workers = num_workers, IO_workers = IO_workers)  
    
    seq_maxabs_scaler = ParMaxAbsScaler(data_file = training_file, num_idxs = num_idxs)
    seq_maxabs_scaler._fit(use_parallel = True, block_reader = block_reader_loader, num_workers = 1, IO_workers = 1) 
    
    par_ord_enc = ParOrdinalEncoder(data_file = training_file, cat_idxs = cat_idxs)
    par_ord_enc._fit(use_parallel = True, block_reader = block_reader_loader, num_workers = num_workers, IO_workers = IO_workers)

    seq_ord_enc = ParOrdinalEncoder(data_file = training_file, cat_idxs = cat_idxs)
    seq_ord_enc._fit(use_parallel = True, block_reader = block_reader_loader, num_workers = 1, IO_workers = 1)
    
if __name__ == '__main__':
    main()