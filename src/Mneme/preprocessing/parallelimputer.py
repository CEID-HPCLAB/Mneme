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
      
    def __init__(self, imputers_map: dict, custom_block_reader: BlockReader,
                 num_workers: int = 2, IO_workers: int = 1) -> None:
        '''
        Initializes a ParallelImputer object with a dictionary of imputers and a custom block reader.

        Args:
            imputers_map (dict): A dictionary mapping each imputer to its corresponding columns/features names.
            custom_block_reader (BlockReader): An instance of the BlockReader class that provides necessary information
            for reading data in chunks (blocks).
            num_workers (int, optional): The number of worker processes to use for parallel processing. This determines 
            the number of data blocks that can be processed simultaneously. If not provided, defaults to 2.
            IO_workers (int, optional): The number of threads to use for IO operations. If not provided, 
            defaults to 1.
        '''
        
        self.block_reader = custom_block_reader
        self.num_workers = num_workers
        self.IO_workers = IO_workers
        
        # Initialize an empty list for storing the imminent (final) reduced imputers
        self.reduced_imputers = []
        self.imputers_ = {key:{"cols":val, "metadata": key.get_params()} for key, val in imputers_map.items()}
        
        t0 = time()
        
        # The inverse map is a dictionary where the keys are the numeric indexes 
        # of the columns and the values are the corresponding column names.
        inverse_map = self._inverse_map(self.block_reader.feature_idxs_map)
        
        t1 = time()
        _Mneme_logger.benchmark(f"[INFO] Time for creating the inverse map: {t1 - t0:.6f}")
        
        # Mapping each imputer to its corresponding columns' names
        for imputer, imputer_vals in self.imputers_.items():
            # If the columns are represented as strings, assign them directly as feature names
            if isinstance(imputer_vals["cols"][0], str):
                self.imputers_[imputer]["feature_names"] = imputer_vals["cols"]
            else:
                # The columns are represented as indexes, we should map them to their corresponding feature names
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


    def parallel_fit(self) -> None:
        '''
        Fits a series of imputers in parallel using multiple processes.

        The method uses the multiprocessing library to create a pool of worker processes. Each worker process is 
        responsible for fitting a chunk of data using a specific pipeline of imputers. The results of these fits are stored 
        in a list of dictionaries, where each dictionary contains the partial fits produced by a worker process.

        After all the worker processes have completed their tasks, the method collects the partial fits from the list of 
        dictionaries and extends the corresponding lists in the `gathered_partial_fits` dictionary.

        After all partial fits have been gathered, the method sets the partial fits of each imputer and 
        calls the reduce method to combine the partial fits into a final fit.
        
        Note: This function uses the 'multiprocessing' library and is designed to work on systems with multiple cores.

        Returns:
            None
        '''
        
        gathered_partial_fits = {}
        
        # Iterate over the imputers and enumerate them to get the index
        for index, imputer in enumerate(self.imputers_.keys()):
            # Create a unique key for each imputer based on its type and index
            key_name = type(imputer).__name__ + f"_{index}"
            # Initialize an empty list for the partial fits of this imputer
            gathered_partial_fits[key_name] = list()

        t0 = time()
        
        pool = mp.Pool(processes = self.num_workers)
        partial_fits = pool.map(self.process_chunk, self.block_reader.block_offsets)
        
        # Close the multiprocessing pool and wait for all processes to finish
        pool.close()
        pool.join()
        
        t1 = time()
        
        for partial_fit_dict in partial_fits:
            # Iterate over each partial fitted imputer
            for key in partial_fit_dict.keys():
                # Extend the corresponding list in the gathered_partial_fits with the corresponding partial fitted imputer of this chunk 
                gathered_partial_fits[key].extend(partial_fit_dict[key])
        
        self.partial_fits = {key: list(val) for key, val in gathered_partial_fits.items()}
            
        # Reduce the partial fits to obtain the final fit of each imputer
        self.reduce()

        t2 = time()
        _Mneme_logger.benchmark(f"Parallel Times of Imputer Pipeline -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
    
    
    def print(self) -> None:
        '''
        Print the attributes of each imputer in the pipeline.

        This method iterates over the reduced imputers in the pipeline. For each imputer, 
        it prints all of its attributes along with their values.

        Returns:
            None
        '''
        
        # For each imputer in the pipeline
        for index, imputer in enumerate(self.reduced_imputers):
            # Print all attributes of the imputer
            for attr in vars(imputer):
                print(f"pimputer_{index}.{attr} = {getattr(imputer, attr)}")
            print(f"{'='*35}")
        
      
    def reduce(self) -> None:
        '''
        Reduces the partial fits obtained from parallel fitting into final imputers.

        This method iterates over the partial fits dictionary, which contains the partial fits of each imputer.
        It retrieves the metadata for each imputer and extracts the imputer strategy.
        Then, it calls the reduce_imputers function to combine the partial fits (of each imputer) into a final imputer,
        based on the imputer strategy.
        
        Returns:
            None
        '''
        
        for index, imputers_fits_list in enumerate(self.partial_fits.values()):
            imputer_metadata = list(islice(self.imputers_.values(), index, index+1))
            # Extract the imputer strategy from the metadata
            imputer_strategy = imputer_metadata[0]["metadata"]["strategy"]
            # Reduce the partial fits (of each imputer) into a final imputer based on the strategy
            self.reduced_imputers.append(reduce_imputers(list(imputers_fits_list), imputer_strategy))
            
            
    def process_chunk(self, block_offset: int) -> Dict[str, List[Any]]:
        '''
        Process a chunk of data and obtain the computed partial fits for each imputer.

        This method loads a chunk of data from the training file based on the given block offset. It then iterates
        over each imputer, fits it to the chunk of data and obtains a partial fit. The partial fit is stored
        in a dictionary with keys representing the type and index of the imputer.
        
        Args:
            block_offset (int): The offset in bytes from the beginning of the file to load the chunk of data.

        Returns:
            Dict[str, List[Any]]: A dictionary containing partial fits for each imputer.
        '''
        
        training_file = self.block_reader.training_file
        block_size = self.block_reader.block_size
        
        partial_fits = {}
        
        # Create empty lists to store partial fits for each imputer
        for index, imputer in enumerate(self.imputers_.keys()):
            key_name = type(imputer).__name__ + f"_{index}"
            partial_fits[key_name] = list()   
        
        # Load the chunk of data from the training file
        chunk = self._load_chunk(training_file, block_size, block_offset)
        
        # Fit each imputer to the chunk of data and obtain a partial fit
        for index, (imputer, imputer_args) in enumerate(self.imputers_.items()):
            imputer_callable = type(imputer)
            imputer = imputer_callable(**imputer_args["metadata"])
            
            # Fit the imputer to the corresponding features of the loaded chunk of data
            imputer.fit(chunk[imputer_args["feature_names"]])
            imputer.n_samples_seen_ = chunk.shape[0]
            
            # If the imputer strategy is mean, calculate and store the count of NaN values
            # Τhis is necessary for the incremental mean and variance algorithm (T. Chan, G. Golub, R. LeVeque. 
            # Algorithms for computing the sample variance: recommendations, The American Statistician, Vol. 37, No. 3, pp. 242-247), 
            # which is used in the reduce process
            if imputer_args["metadata"]["strategy"] == "mean":
                #polar version of counting nan values
                nan_vals = chunk[imputer_args["feature_names"]].null_count().to_numpy()
                imputer.nan_vals = nan_vals.reshape(-1)
                
            imputer_name = type(imputer).__name__ + f"_{index}"
            # Add the partial fitted imputer to the corresponding partial fits list
            partial_fits[imputer_name].append(imputer)

        # Cleanup: delete the chunk to free up memory
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
        
        # Check if custom_cols_mapping is None and the columns are specified by their numerical indices
        if custom_cols_mapping is None and isinstance(self.imputers_[imputer_idx]["cols"][0], int):
            # Return the numerical indices of the columns used by the imputer
            return self.imputers_[imputer_idx]["cols"]
        
        cols = self.imputers_[imputer_idx]["cols"]
        # Check if custom_cols_mapping is provided and the columns are specified by their numerical indices
        if custom_cols_mapping is not None and isinstance(cols[0], int):
            # Create a list with the feature names corresponding to the numerical indices of the columns used by the imputer
            cols = [key for key, value in self.block_reader.feature_idxs_map.items() 
                    if value in cols]
        
        # Use custom_cols_mapping if the preprocessing pipeline contains less features than the training dataset    
        # and, consequently, the incoming data for transforming (by the imputer) contains fewer columns and a 
        # different (numerical indexing) mapping compared to the original training dataset 
        # (upon which the feature_idxs_map have been calculated by the block reader).
        feature_idxs_map = self.block_reader.feature_idxs_map if custom_cols_mapping is None else custom_cols_mapping
        contains = lambda idx: idx in cols
        # Filter the feature_idxs_map to include only the features used by the imputer
        feature_num_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())
        
        return [feature_pair[1] for feature_pair in feature_num_idx_dict] 
    
    
    def _load_chunk(self, training_file: str, block_size: int, block_offset: int) -> pl.DataFrame:
        '''
        Load a chunk of data from the training file.
        
        This method loads a chunk of data from the training file specified by `training_file`.
        It seeks to the specified `block_offset` in the file, reads `block_size` rows and extracts
        the columns specified by `columns`. These columns are determined by the feature names
        of the imputers included in the pipeline.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The number of rows to read from the file.
            block_offset (int): The offset in bytes from the beginning of the file.

        Returns:
            pl.DataFrame: A DataFrame containing the loaded chunk of data.
        '''

        chunk_features = []
        
        # Collect the feature names associated with the imputers
        for imputer in self.imputers_.keys():
            chunk_cols = self.imputers_[imputer]["feature_names"] 
            chunk_features.extend(chunk_cols)
        
        # Open the training file and seek to the specified block offset
        with open(training_file, 'r') as dfile:
            dfile.seek(block_offset)
            feature_idxs_map = self.block_reader.feature_idxs_map
            contains = lambda idx: idx in chunk_features
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
       
            # Read the chunk of data from the file, selecting only the required columns
            chunk = pl.read_csv(dfile, has_header = False, n_rows = block_size, new_columns = list(feature_num_idx_dict.keys()),
                                n_threads = self.IO_workers, columns = list(feature_num_idx_dict.values()))
        
        
        return chunk