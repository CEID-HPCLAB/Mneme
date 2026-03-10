import pandas as pd
import numpy as np
from time import perf_counter as time
from itertools import islice
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from sklearn.impute import SimpleImputer
from functools import partial
from Mneme import BlockReader
from Mneme.utils import reduce_imputers
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


class ParImputer():
    '''
    A class used to perform parallel imputation of missing values on large datasets using multiple imputers.

    This class is designed to handle large datasets that cannot fit into memory. It facilitates parallel imputation
    of data blocks using multiple imputers. It allows for efficient processing of large datasets 
    by dividing the imputation tasks into smaller chunks and processing them in parallel using multiple processes. 
    The partial results are then combined to obtain the final imputation pipeline on the whole dataset.
    '''
      
    def __init__(self, imputers_map: dict, custom_block_reader: BlockReader, num_workers: int = 2, IO_workers: int = 1) -> None:
          
        self.block_reader = custom_block_reader; self.num_workers = num_workers; self.IO_workers = IO_workers
        
        self.reduced_imputers = []
        self.imputers_ = {key:{"cols":val, "metadata": key.get_params()} for key, val in imputers_map.items()}
        
        t0 = time()
        
        # The inverse map is a dictionary where the keys are the numeric indexes 
        # of the columns and the values are the corresponding column names
        inverse_map = self._inverse_map(self.block_reader.feature_idxs_map)
        
        t1 = time()
        _Mneme_logger.benchmark(f"[INFO] Time for creating the inverse map: {t1 - t0:.6f}")
        
        for imputer, imputer_vals in self.imputers_.items():
            if isinstance(imputer_vals["cols"][0], str):
                self.imputers_[imputer]["feature_names"] = imputer_vals["cols"]
            else:
                self.imputers_[imputer]["feature_names"] = [inverse_map[col][0] for col in imputer_vals["cols"]]

    
    def _inverse_map(self, feature_idxs_map: Dict[str, int]) -> Dict[int, List[str]]:
        '''
        Create an inverse mapping from the given feature indexes map.

        This method takes a dictionary mapping feature names to their corresponding indexes and returns a new dictionary 
        where the keys are the indexes and the values are lists of feature names that correspond to each index.

        Args:
            feature_idxs_map (Dict[str, int]): A dictionary mapping feature names to their corresponding indexes.

        Returns:
            Dict[int, List[str]]: A dictionary where the keys are the numeric indexes of the columns and the values 
            are lists of feature names that correspond to each index.
        '''
        
        inv_map = defaultdict(list) 
        
        for num_idx, feature_name in feature_idxs_map.items():
            # For each item, append the feature name to the list of features for the corresponding index in the inverse map 
            inv_map[feature_name].append(num_idx)
            
        return inv_map


    def fit(self) -> None:
        '''
        Fits a series of imputers in parallel using multiple processes.

        Returns:
            None
        '''
        
        gathered_partial_fits = {}
        
        self._nonconst_imputer_ind = []; self.reduced_imputers = [None] * len(self.imputers_)

        for index, imputer in enumerate(self.imputers_.keys()):
            
            if (imputer.strategy == "constant"):
                features = self.imputers_[imputer]["feature_names"]
                imputer.statistics_ = np.full(len(features), imputer.fill_value); imputer._fit_dtype = imputer.statistics_.dtype 
                imputer.n_features_in_ = len(features); imputer.feature_names_in_ = np.array(features); self.reduced_imputers[index] = imputer
            else:
                self._nonconst_imputer_ind.append(index); key_name = type(imputer).__name__ + f"_{index}"
                gathered_partial_fits[key_name] = list()

        t0 = time()
        
        if len(gathered_partial_fits) != 0:
            pool = mp.Pool(processes = self.num_workers)
            partial_fits = pool.map(self.process_chunk, self.block_reader.block_offsets)
            
            pool.close(); pool.join()
            
            t1 = time()
            
            for partial_fit_dict in partial_fits:
                for key in partial_fit_dict.keys():
                    gathered_partial_fits[key].extend(partial_fit_dict[key])
            
            self.partial_fits = {key: list(val) for key, val in gathered_partial_fits.items()}
                
            self.reduce()

            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Imputer Pipeline Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
    
    
    def print(self) -> None:
        '''
        Print the attributes of each imputer in the pipeline.

        This method iterates over the reduced imputers in the pipeline. For each imputer, 
        it prints all of its attributes along with their values.

        Returns:
            None
        '''
        
        for index, imputer in enumerate(self.reduced_imputers):
            for attr in vars(imputer):
                print(f"pimputer_{index}.{attr} = {getattr(imputer, attr)}")
            print(f"{'='*35}")
        
      
    def reduce(self) -> None:
        '''
        Reduces the partial fits obtained from parallel fitting into final imputers.
        
        Returns:
            None
        '''
        
        for index, imputers_fits_list in enumerate(self.partial_fits.values()):
            imputer_metadata = list(islice(self.imputers_.values(), index, index + 1))
            imputer_strategy = imputer_metadata[0]["metadata"]["strategy"]
            self.reduced_imputers[self._nonconst_imputer_ind[index]] = reduce_imputers(list(imputers_fits_list), imputer_strategy)
            
            
    def process_chunk(self, block_offset: int) -> Dict[str, List[Any]]:
        '''
        Process a chunk of data and obtain the computed partial fits for each imputer.

        Args:
            block_offset (int): The offset in bytes from the beginning of the file to load the chunk of data.

        Returns:
            Dict[str, List[Any]]: A dictionary containing partial fits for each imputer.
        '''
        
        training_file = self.block_reader.training_file; block_size = self.block_reader.block_size
        
        partial_fits = {}
        
        for index, imputer in enumerate(self.imputers_.keys()):
            if index not in self._nonconst_imputer_ind:
                continue
            
            key_name = type(imputer).__name__ + f"_{index}"; partial_fits[key_name] = list()   
        
        chunk = self._load_chunk(training_file, block_size, block_offset)
        
        for index, (imputer, imputer_args) in enumerate(self.imputers_.items()):
            if index not in self._nonconst_imputer_ind:
                continue
            
            imputer_callable = type(imputer); imputer = imputer_callable(**imputer_args["metadata"])
            
            if imputer_args["metadata"]["strategy"] == "mean":
                imputer.statistics_ = np.array(chunk.select(pl.col(imputer_args["feature_names"]).mean()).row(0), dtype = np.float64)
                nan_vals = chunk[imputer_args["feature_names"]].null_count().to_numpy()
                imputer.nan_vals = nan_vals.reshape(-1); imputer.n_samples_seen_ = chunk.shape[0]
                imputer.n_features_in_ = len(imputer_args["feature_names"]); imputer.feature_names_in_ = np.array(imputer_args["feature_names"])
            
            else:
                imputer.fit(chunk[imputer_args["feature_names"]])
                imputer.n_samples_seen_ = chunk.shape[0]
        
            imputer_name = type(imputer).__name__ + f"_{index}"
            partial_fits[imputer_name].append(imputer)
            
        del(chunk)
        
        return partial_fits
        
    
    def transform(self, data: np.ndarray, custom_cols_mapping: Union[Optional[dict], None] = None) -> None:
        '''
        Transform the missing values in the data using the fitted imputers.

        This method applies the transformations to the missing values in the data using the fitted imputers.
        It iterates over each fitted imputer and transforms the corresponding columns in the data. The transformed
        data is updated in-place.

        Args:
            data (numpy.ndarray): The input data with missing values to be transformed.
            custom_cols_mapping (dict, optional): A dictionary mapping column names to their numerical indices. 
                This parameter is used when the training dataset include columns that will not be used by the imputer pipeline

        Returns:
            None

        Notes:
            - The transformation is performed in-place, meaning the original data array is modified.
            - Any warning about 'feature_names_in' when fitted with Pandas DataFrame containing named columns is ignored.
        
        '''
        
        # Ignore warning about 'feature_names_in' when fitted with Pandas DataFrame containing named columns
        warnings.filterwarnings("ignore", category = UserWarning)
        
        for imputer, imputer_key in zip(self.reduced_imputers, self.imputers_.keys()):
            # Retrieve the numerical indices corresponding to the features for the current imputer
            feature_idxs = self._set_feature_mapping(imputer_key, custom_cols_mapping)
            # Transform the missing values using the fitted imputer
            transformed_data = imputer.transform(data[:, feature_idxs])
            # Update the corresponding columns in the input data with the transformed values
            data[:, feature_idxs] = transformed_data    
            
        warnings.resetwarnings()  


    def _transform(self, chunk: pl.DataFrame, custom_cols_mapping: Union[Optional[dict], None] = None) -> None:
        '''
        Transform the missing values in the input data using the fitted imputers.

        Args:
            chunk (pl.DataFrame): The input data with missing values to be transformed.
            custom_cols_mapping (dict, optional): A dictionary mapping column names to their numerical indices. 
                This parameter is used when the training dataset include columns that will not be used by the imputer pipeline

        Returns:
            None

        Notes:
            - The transformation is performed in-place, meaning the original data array is modified.
        '''

        warnings.filterwarnings("ignore", category = UserWarning)
        
        for imputer, imputer_key in zip(self.reduced_imputers, self.imputers_.keys()):
            feature_idxs = self._set_feature_mapping(imputer_key, custom_cols_mapping)
            for n, col_idx in enumerate(feature_idxs):
                chunk_imp = chunk[:, col_idx].fill_null(imputer.statistics_[n])    
                chunk = chunk.replace_column(col_idx, chunk_imp)
         
        warnings.resetwarnings()    

    
    def _set_feature_mapping(self, imputer_idx: SimpleImputer, custom_cols_mapping: Union[dict, None]) -> List[int]:
        '''
        Set the the numerical indices corresponding to the feature indices for the specific 
        imputer based on the given custom_cols_mapping.
       
        Args:
            imputer_idx (SimpleImputer): The imputer for which the feature mapping is set.
            custom_cols_mapping (Union[dict, None]): A dictionary mapping column names to their numerical indices,
                or None if no custom mapping is provided.

        Returns:
            List[int]: A list of numerical indices corresponding to the feature indices used by the imputer.
        '''
        
        if custom_cols_mapping is None and isinstance(self.imputers_[imputer_idx]["cols"][0], int):
            return self.imputers_[imputer_idx]["cols"]
        
        cols = self.imputers_[imputer_idx]["cols"]
        if custom_cols_mapping is not None and isinstance(cols[0], int):
            cols = [key for key, value in self.block_reader.feature_idxs_map.items() 
                    if value in cols]
        
        feature_idxs_map = self.block_reader.feature_idxs_map if custom_cols_mapping is None else custom_cols_mapping
        contains = lambda idx: idx in cols
        feature_num_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())
        
        return [feature_pair[1] for feature_pair in feature_num_idx_dict] 
    
    
    def _load_chunk(self, training_file: str, block_size: int, block_offset: int) -> pl.DataFrame:
        '''
        Load a chunk of data from the training file.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The number of rows to read from the file.
            block_offset (int): The offset in bytes from the beginning of the file.

        Returns:
            pl.DataFrame: A DataFrame containing the loaded chunk of data.
        '''

        chunk_features = []
        
        for imputer in self.imputers_.keys():
            chunk_cols = self.imputers_[imputer]["feature_names"] 
            chunk_features.extend(chunk_cols)
        
        # Open the training file and seek to the specified block offset
        with open(training_file, 'r') as dfile:
            dfile.seek(block_offset)
            feature_idxs_map = self.block_reader.feature_idxs_map
            contains = lambda idx: idx in chunk_features
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            try:
                chunk = pl.read_csv(dfile, has_header = False, n_rows = block_size, new_columns = list(feature_num_idx_dict.keys()),
                                    n_threads = self.IO_workers, columns = list(feature_num_idx_dict.values()))
            
            except Exception as e:
                print(e)

        return chunk