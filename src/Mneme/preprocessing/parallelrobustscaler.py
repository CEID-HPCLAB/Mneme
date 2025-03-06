import pandas as pd
from sklearn.preprocessing import RobustScaler
from time import perf_counter as time
import numpy as np
from typing import List, Dict, Optional, Union
from functools import partial
from Mneme.utils import reduce_robust_scalers, _copy_attr
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


__all__ = [
    'ParRobustScaler'
]

class ParRobustScaler(RobustScaler):
    '''
    This class extends the RobustScaler from sklearn.preprocessing to provide a parallelized version of the RobustScaler.
    It scales the data using statistics that are robust to outliers by removing the median and scaling 
    the data according to the quantile range. 
    
    In the current implementation, there are some differences compared to the original implementation of the 
    RobustScaler provided by scikit-learn. Because it was not possible (as yet) to parallelize the median calculation algorithm, 
    the median is calculated as the median of the individual medians of the data chunks (blocks). An alternative 
    implementation considers the median to be the mean value of the individual medians of the data chunks (blocks). 
    Also, the interquartile range is defined as the mean of the individual interquartile ranges of the data chunks (blocks), 
    not the interquartile range of the entire dataset. Improving these specific algorithms is a future goal.
    
    This class is designed to work with large datasets that do not fit into memory, 
    by processing chunks of the input training data in parallel, making use of distributed computing.
    It implements methods for fitting the scaler incrementally in parallel, 
    reducing the partial fitted scalers into the final scaler and transforming data.
    '''
    
    def __init__(self, data_file: str, num_idxs: List[str], with_centering: Optional[bool] = True, 
                 with_scaling: Optional[bool] = True, quantile_range: Optional[tuple] = (25.0, 75.0), 
                 copy: Optional[bool] = True, unit_variance: Optional[bool] = False) -> None:
        '''
        Initializes the ParRobustScaler object.
        
        Args:
            data_file (str): The path to the data file to be scaled.
            num_idxs (List[str]): The numerical indices of the features to be scaled.
            with_centering (Optional[bool], default=True): If True, center the data before scaling.
            with_scaling (Optional[bool], default=True): If True, scale the data to interquartile range.
            quantile_range (Optional[tuple], default=(25.0, 75.0)): Quantile range used to scale the data.
            copy (Optional[bool], default=True): If True, the original data is copied and scaled. 
                                                 If False, the original data is overwritten.
            unit_variance (Optional[bool], default=False): If True, scale the data to unit variance.
        
        Returns:
            None
        '''
    
        # Call the constructor of the superclass (RobustScaler)
        super(ParRobustScaler, self).__init__(with_centering = with_centering, with_scaling = with_scaling, 
                                              quantile_range = quantile_range, copy = copy, unit_variance = unit_variance)
        self.data_file = data_file
        self.num_idxs = num_idxs
        self.scalers_ = []

    
    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Fits a RobustScaler to a chunk of data and stores it.

        This method takes a chunk of data, fits a RobustScaler to the features specified by num_idxs 
        and stores the partial fitted scaler to the list of scalers.

        Args:
            chunk_data (pl.DataFrame): A chunk of data to be processed.
        
        Returns:
            None
        '''
        
        scaler = RobustScaler(with_centering = self.with_centering, with_scaling = self.with_scaling, 
                              quantile_range = self.quantile_range, copy = self.copy, unit_variance = self.unit_variance)
        
        # Fit the scaler to the corresponding features in the chunk of data
        scaler.fit(chunk_data[self.num_idxs])
        
        self.scalers_.append(scaler)
        
    
    def reduce(self) -> None:
        '''
        Combines the partial fitted scalers from all chunks into a final scaler.

        This method uses the reduce_robust_scalers function to combine the scalers fitted to each chunk of data 
        into a final scaler and copies the attributes of the final scaler to the current ParRobustScaler object. 
        
        Returns:
            None
        '''
        
        # Combine the scalers fitted to each chunk of data into a final scaler
        final_scaler = reduce_robust_scalers(self.scalers_)
        
        # Copy the attributes of the final scaler to the ParRobustScaler object
        _copy_attr(self, final_scaler)
        
        # Delete the list of the partial fitted scalers and the (combined) final scaler
        del(self.scalers_)
        del(final_scaler)
    
    
    def get_partial_fit(self) -> RobustScaler:
        '''
        Returns the partial fitted scaler.
        
        It's important to note that this method also removes the returned scaler from the list. This is necessary 
        to ensure that if the same process takes on another task, the `scalers_` list does not still contain the 
        partially fitted scalers from the previous tasks. If they weren't removed, the list would contain all 
        the partial fitted scalers from tasks that the specific process has completed, resulting in us not knowing 
        which is the partial fitted scaler of the specific task and therefore not returning the correct scaler.

        Returns:
            tmp_scalers (RobustScaler): The partial fitted scaler.
        '''
        
        tmp_scalers = self.scalers_.pop()
        return tmp_scalers
    
    
    def set_partial_fits(self, partial_fits: List[RobustScaler]) -> None:
        '''
        Gather the partial fitted scalers (from the different chunks of data) into the 'scalers_' list.
        
        Args:
            partial_fits (List[RobustScaler]): A list of partial fitted scalers.
            
        Returns:
            None
        
        '''
        
        self.scalers_ = list(partial_fits)
    
    
    def print(self, use_parallel: bool = True) -> None:
        '''
        Print the attributes of the fitted scaler.
        
        Args:
        use_parallel (bool, optional): If True, the scaler is considered as parallel. Defaults to True.
        
        Returns:
            None
        
        '''
        
        kind_scaler = "parrobscaler" if use_parallel else "robscaler"
        
        # Print all attributes of the scaler
        for attr in vars(self):
            print(f"{kind_scaler}.{attr} = {getattr(self, attr)}")
            

    def _fit(self, use_parallel: Optional[bool] = False, block_reader: Union[Optional[BlockReader], None] = None, 
             num_workers: Optional[int] = 2, IO_workers: Optional[int] = 1, num_blocks: Optional[int] = 100,
             chunk_size: Optional[int] = 5000, imputer: Union[Optional[ParImputer], None] = None) -> None: 
        '''
        Read and fit blocks of data incrementally.
        
        This method reads the data file in chunks and fits the scaler to each chunk. If an imputer is provided, 
        it also applies the imputation to each chunk before fitting the scaler. The method can operate in either 
        sequential or parallel mode, depending on the value of the use_parallel parameter.
        
        Args:
            use_parallel (bool, optional): Whether to use parallel processing. Defaults to False.
            block_reader (BlockReader, optional): A BlockReader object for reading data in blocks. If None, a new BlockReader 
                will be created. Defaults to None.
            num_workers (int, optional): The number of worker processes to use for parallel processing. Defaults to 2.
            IO_workers (int, optional): The number of threads to use for IO-bound operations. Defaults to 1.
            num_blocks (int, optional): The number of blocks to divide the data into for parallel processing. This is only used 
            in the parallel version of the method. In the sequential version, this parameter is not taken into account 
            as the data is read in chunks determined by the chunk_size parameter. Defaults to 100.
            chunk_size (int, optional): The size of the chunks to read the data in. This is only used in the sequential version 
            of the method. In the parallel version, the block size is determined solely by the num_blocks parameter. Defaults to 5000.
            imputer (ParImputer, optional): An imputer to apply to the data before fitting the scaler. Defaults to None.
        
        Returns:
            None   
        '''
        
        if not use_parallel:
            # Set chunk_size to the block_size of block_reader if it exists, otherwise keep the provided chunk_size
            chunk_size = block_reader.block_size if block_reader is not None else chunk_size
            
            # Read data from the provided CSV file in chunks
            data = pd.read_csv(self.data_file, chunksize = chunk_size)
            
            # If a block reader is not provided, create a new one
            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            # Set the feature mapping by identifying the numerical indexes for the given column names (num_idxs)
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            t0 = time()
            
            if imputer is not None:
                
                # If imputer is provided, transform each chunk before fitting
                for chunk_data in data:
                    chunk_data_np = chunk_data.to_numpy()
                    # Apply the provided fitted imputer to the chunk
                    imputer.transform(chunk_data_np)
                    chunk_data = pd.DataFrame(chunk_data_np, columns = chunk_data.columns, copy = False)
                    # Fit the scaler to the chunk
                    self.process_chunk(chunk_data)
                    
            else:
                # Otherwise, directly process each chunk
                for chunk_data in data:
                    self.process_chunk(chunk_data)
            
            t1 = time()
            
            # Combine the fitted scalers into a final scaler
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Sequential Times of Standalone RobustScaler -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
        else:
            # Parallel fitting 
            
            t0 = time()
            # If a block reader is not provided, create a new one
            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            block_size, columns, block_offsets, num_blocks, feature_idxs_map = \
            block_reader.block_size, block_reader.columns, block_reader.block_offsets,\
            block_reader.num_blocks, block_reader.feature_idxs_map
    
            # Set the feature mapping by identifying the numerical indexes for the given column names (num_idxs)
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            
            pool = mp.Pool(processes = num_workers)
            # Create a partial function for fitting the scaler to the corresponding block of data
            partial_func = partial(self._partial_fit, 
                                   args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            self.scalers_ = pool.map(partial_func, range(num_blocks))
            
            # Close the multiprocessing pool and wait for all processes to finish
            pool.close()
            pool.join()
            
            t1 = time()
            
            # Combine the partial fitted scalers into a final scaler
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Times of Standalone RobustScaler -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")

    
    def _partial_fit(self, index: int, args: tuple) -> RobustScaler:
        '''
        Perform fitting on a specific block of data.

        This method loads a block of data, fits a RobustScaler on it and returns the fitted scaler. The block of data 
        is determined by the value of the index, which specifies the block offset. The fitting is performed only on the 
        columns that correspond to this specific scaler (num_idxs), not all columns.

        Args:
            index (int): The index of the block of data to fit the scaler to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer 
                          and the number of IO threads.

        Returns:
            RobustScaler: The scaler fitted to the block of data.
        '''
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        # Load the block of data
        chunk = self._load_chunk(self.data_file, block_size, columns, self.num_idxs, block_offsets[index], 
                                 feature_idxs_map, imputer, IO_workers)
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [Standalone ParRobustScaler]: {t1-t0:.6f}")
        
        # Fit a RobustScaler on the loaded chunk and return the scaler
        scaler = RobustScaler(with_centering = self.with_centering, with_scaling = self.with_scaling, 
                              quantile_range = self.quantile_range, copy = self.copy, unit_variance = self.unit_variance)
        scaler.fit(chunk[self.num_idxs])
        
        return scaler
    
    
    def transform(self, data: np.ndarray) -> None:
        '''
        Transform the data in-place using the final fitted scaler.

        This method applies the transformation to the specified columns of the data array in place. It first extracts
        the columns specified by `self.feature_idxs_` and transforms them using the inherited `transform` method
        from the superclass. The transformed values are then assigned back to the original data array at the same
        column indices.

        Args:
            data (numpy.ndarray): The input data array to be transformed.

        Returns:
            None.
        '''
        
        # Ignore warnings about feature_names_in when fitted with Pandas Dataframe containing named columns
        warnings.filterwarnings("ignore", category = UserWarning)
        
        # Extract the specified columns for transformation and apply the transformation 
        transformed_data = super().transform(data[:, self.feature_idxs_])
        # Assign the transformed values back to the original data array at the same column indices
        data[:, self.feature_idxs_] = transformed_data     
        
        warnings.resetwarnings()  
    
    
    def _set_feature_mapping(self, feature_idxs_map: Dict[str, int]) -> None:
        '''
        Set the feature mapping based on the provided feature indices map.

        This method filters the feature indices map based on the indices stored in `self.num_idxs`. It creates a
        new list containing the numerical indices corresponding to the feature indices stored in `self.num_idxs`
        and assigns it to `self.feature_idxs_`.

        Args:
            feature_idxs_map (Dict[str, int]): A dictionary mapping feature names to their corresponding numerical indices.

        Returns:
            None
        '''
        
        # Define a lambda (helper) function to check if an index is present in self.num_idxs
        contains = lambda idx: idx in self.num_idxs
        
        # Filter the feature indices map based on the indices stored in self.num_idxs
        feature_num_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())
        
        # Store the numerical indices corresponding to the feature indices stored in self.num_idxs
        self.feature_idxs_ = [feature_pair[1] for feature_pair in feature_num_idx_dict]
        
    
    def _load_chunk(self, training_file: str, block_size: int, columns: List[str], num_idxs: List[str], 
                    block_offset: int, feature_idxs_map: Dict[str, int], imputer: Union[ParImputer, None], IO_workers: int) -> pl.DataFrame:
        '''
        Load a block of data from the training file.

        This method reads a block of data from the training file, applies any necessary transformations (imputer) and returns 
        the processed data. The block of data is determined by the block offset and block size. If an imputer is provided, 
        it is used to fill in missing values in the data.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The size of the block of data to read.
            columns (List[str]): The column names in the training data.
            num_idxs (List[str]): A list of strings representing the names of the features to be scaled.
            block_offset (int): The offset to start reading the block of data from.
            feature_idxs_map (Dict[str, int]): A dictionary mapping feature names to their corresponding numerical indexes.
            imputer (Union[ParImputer, None]): An optional imputer to fill in missing values in the data.
            IO_workers (int): Specifies the number of threads to use for the IO-bound operation (loading the block of data).

        Returns:
            pl.DataFrame: The processed block of data.
        '''
        
        # Nested function to provide a mapping of features with their corresponding numerical indices.
        def _compute_mapping(columns: list) -> dict:
            '''
            Compute a mapping from column names to their numeric indices.
            
            Args:
                columns (list): The column names.

            Returns:
                dict: A dictionary mapping column names to their numeric indices.
            '''
            
            # The keys of the dictionary are the column names and the values are the corresponding indices  
            # print(columns)
            col_indices = pd.Series(columns).reset_index(drop = True).to_dict()
            
            # Reverse the key-value pairs in the dictionary
            cols_mapping = {v: k for k, v in col_indices.items()}
            
            return cols_mapping
        
        
        def _compute_features(imputer: ParImputer, num_idxs: List[str], columns: List[str]) -> List[str]:
            '''
            This function determines the union of features from the ParImputer object and the num_idxs of the ParRobustScaler 
            (the features to be scaled) and returns them in the order they appear.

            Args:
                imputer (ParImputer): The ParImputer object.
                num_idxs (List[str]): A list of strings representing the names of the features to be scaled.
                columns (List[str]): The column names in the training data.

            Returns:
                List[str]: A list containing the union of features from the ParImputer and ParRobustScaler in the order they appear.
            '''
            
            features = []
            
            # Iterate over the imputers and extract their columns
            for imputer_idx in imputer.imputers_.keys():
                cols = imputer.imputers_[imputer_idx]["cols"]
                
                # If columns are represented by numerical indices, convert them to column names
                if isinstance(cols[0], int):
                    cols = [key for key, value in feature_idxs_map.items() if value in cols] 
                
                features.extend(cols)
            
            features.extend(num_idxs)
                
            ordered_list = []
            # Create a list containing the union of features from the ParImputer and ParRobustScaler 
            # in the order they appear.
            for el in columns:
                if el in features and el not in ordered_list:
                    ordered_list.append(el)
            
            return ordered_list
            
        
        cols = num_idxs if imputer is None else _compute_features(imputer, num_idxs, columns)
        
        with open(training_file, 'r') as dfile:
            # Move the file pointer to the specified block offset
            dfile.seek(block_offset)
            contains = lambda idx: idx in cols
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            # Read the corresponding block of data from the file. The size of the block is determined by the 'block_size' 
            # parameter. We only read the columns that will be used by the pipeline's preprocessors.
            dfX_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, 
                                    n_threads = IO_workers, new_columns = list(feature_num_idx_dict.keys()),
                                    columns = list(feature_num_idx_dict.values()))
            
        
            
        # Impute potentially missing values in the block of data if an imputer is provided
        if imputer is not None:
            # Compute custom column mapping if the number of columns in dfX_train differs from the columns of the 
            # total training dataset. This is done when the columns intended for fitting the scaler are fewer 
            # than the columns in the entire dataset

            custom_cols_mapping = _compute_mapping(columns = dfX_train.columns)\
                                  if len(columns) != len(dfX_train.columns)\
                                  else None
            data_np = dfX_train.to_numpy()
            
            # Apply the transformation using the imputer and the updated columns mapping
            imputer.transform(data_np, custom_cols_mapping)
            # Convert the transformed numpy array back to a dataframe with the same columns
            dfX_train = pl.DataFrame(data_np, schema = dfX_train.columns)
        
        return dfX_train