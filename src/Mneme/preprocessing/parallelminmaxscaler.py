import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from time import perf_counter as time
import numpy as np
from typing import List, Dict, Optional, Union
from functools import partial
from Mneme.utils import reduce_minmax_scalers, _copy_attr
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger



__all__ = [
    'ParMinMaxScaler'
]

class ParMinMaxScaler(MinMaxScaler):
    '''
    ParMinMaxScaler extends the MinMaxScaler from sklearn.preprocessing to provide a parallelized version of the MinMaxScaler. 
    It scales the data to lie within the range specified by feature_range by transforming the data using the 
    formula: 
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min, where min, max = feature_range.
    
    This transformation is often used as an alternative to zero mean,
    unit variance scaling. It is useful when the data is not normally distributed, 
    and the user wants to preserve the shape of the distribution.
 
    This class is designed to work with large datasets that do not fit into memory, 
    by processing chunks of the input training data in parallel, making use of distributed computing.
    It implements methods for fitting the scaler incrementally in parallel, 
    reducing the partial fitted scalers into the final scaler and transforming data.
    '''
    
    def __init__(self, num_idxs: List[str], data_file: str = None, feature_range: Optional[tuple] = (0,1), 
                 copy: Optional[bool] = True, clip: Optional[bool] = False) -> None:
        
        super(ParMinMaxScaler, self).__init__(feature_range = feature_range, copy = copy, clip = clip)
        
        self.data_file = data_file; self.num_idxs = num_idxs; self.scalers_ = []
    
    
    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Fits a MinMaxScaler to a chunk of data and stores it.

        Args:
            chunk_data (pl.DataFrame): A chunk of data to be processed.
        
        Returns:
            None
        '''
        
        scaler = MinMaxScaler(feature_range = self.feature_range, copy = self.copy, clip = self.clip)
        scaler.fit(chunk_data[self.num_idxs])
        
        self.scalers_.append(scaler)
        
    
    def reduce(self) -> None:
        '''
        Combines the partial fitted scalers from all chunks into a final scaler.
        
        Returns:
            None
        '''

        final_scaler = reduce_minmax_scalers(self.scalers_)
        _copy_attr(self, final_scaler)
        
        del(self.scalers_); del(final_scaler)

    
    def get_partial_fit(self) -> MinMaxScaler:
        '''
        Returns the partial fitted scaler.

        Returns:
            tmp_scalers (MinMaxScaler): The partial fitted scaler.
        '''
        
        tmp_scalers = self.scalers_.pop()
        return tmp_scalers
    
    
    def set_partial_fits(self, partial_fits: List[MinMaxScaler]) -> None:
        '''
        Gather the partial fitted scalers (from the different chunks of data) into the 'scalers_' list.
        
        Args:
            partial_fits (List[MinMaxScaler]): A list of partial fitted scalers.
            
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
        
        kind_scaler = "pmmscaler" if use_parallel else "mmscaler"
        
        for attr in vars(self):
            print(f"{kind_scaler}.{attr} = {getattr(self, attr)}")
    
    
    def fit(self, block_reader: Union[Optional[BlockReader], None] = None, num_workers: Optional[int] = 2, IO_workers: Optional[int] = 1, 
            num_blocks: Optional[int] = 100, chunk_size: Optional[int] = 5000, imputer: Union[Optional[ParImputer], None] = None) -> None: 
        '''
        Read and fit blocks of data incrementally.
        
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
        
        if num_workers == 1 and IO_workers == 1:
            chunk_size = block_reader.block_size if block_reader is not None else chunk_size
            
            data = pd.read_csv(self.data_file, chunksize = chunk_size)
            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
        
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            t0 = time()
            
            if imputer is not None:
                for chunk_data in data:
                    chunk_data_np = chunk_data.to_numpy()
                    imputer.transform(chunk_data_np)
                    chunk_data = pd.DataFrame(chunk_data_np, columns = chunk_data.columns, copy = False)
                    self.process_chunk(chunk_data)
                    
            else:
                for chunk_data in data:
                    self.process_chunk(chunk_data)
            
            t1 = time()

            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Sequential Standalone MinMaxScaler Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
        else:

            t0 = time()

            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            block_size, columns, block_offsets, num_blocks, feature_idxs_map = \
            block_reader.block_size, block_reader.columns, block_reader.block_offsets,\
            block_reader.num_blocks, block_reader.feature_idxs_map
    
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            
            pool = mp.Pool(processes = num_workers)
            partial_func = partial(self._partial_fit, args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            
            self.scalers_ = pool.map(partial_func, range(num_blocks))
            
            pool.close(); pool.join()
            
            t1 = time()
            
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Standalone MinMaxScaler Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")

    
    def _partial_fit(self, index: int, args: tuple) -> MinMaxScaler:
        '''
        Perform fitting on a specific block of data.

        Args:
            index (int): The index of the block of data to fit the scaler to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer 
                          and the number of IO threads.
        Returns:
            MinMaxScaler: The scaler fitted to the block of data.
        '''
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        chunk = self._load_chunk(self.data_file, block_size, columns, self.num_idxs, block_offsets[index], 
                                 feature_idxs_map, imputer, IO_workers)
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [Standalone ParMinMaxScaler]: {t1-t0:.6f}")
        
        scaler = MinMaxScaler(feature_range = self.feature_range, copy = self.copy, clip = self.clip)
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
    
        warnings.filterwarnings("ignore", category = UserWarning)
        
        transformed_data = super().transform(data[:, self.feature_idxs_])
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
        
        contains = lambda idx: idx in self.num_idxs
        feature_num_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())

        self.feature_idxs_ = [feature_pair[1] for feature_pair in feature_num_idx_dict]
        
    
    def _load_chunk(self, training_file: str, block_size: int, columns: List[str], num_idxs: List[str], 
                    block_offset: int, feature_idxs_map: Dict[str, int], imputer: Union[ParImputer, None], IO_workers: int) -> pl.DataFrame:
        '''
        Load a block of data from the training file.

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
        
        def _compute_mapping(columns: list) -> dict:
            '''
            Compute a mapping from column names to their numeric indices.
            
            Args:
                columns (list): The column names.

            Returns:
                dict: A dictionary mapping column names to their numeric indices.
            '''
            
            col_indices = pd.Series(columns).reset_index(drop = True).to_dict()
            cols_mapping = {v: k for k, v in col_indices.items()}
            
            return cols_mapping
        
        
        def _compute_features(imputer: ParImputer, num_idxs: List[str], columns: List[str]) -> List[str]:
            '''
            This function determines the union of features from the ParImputer object and the num_idxs of the ParMinMaxScaler 
            (the features to be scaled) and returns them in the order they appear.

            Args:
                imputer (ParImputer): The ParImputer object.
                num_idxs (List[str]): A list of strings representing the names of the features to be scaled.
                columns (List[str]): The column names in the training data.

            Returns:
                List[str]: A list containing the union of features from the ParImputer and ParMinMaxScaler in the order they appear.
            '''
            
            features = []
            
            for imputer_idx in imputer.imputers_.keys():
                cols = imputer.imputers_[imputer_idx]["cols"]
                
                if isinstance(cols[0], int):
                    cols = [key for key, value in feature_idxs_map.items() if value in cols] 
                
                features.extend(cols)
            
            features.extend(num_idxs)
                
            ordered_list = []
            for el in columns:
                if el in features and el not in ordered_list:
                    ordered_list.append(el)
            
            return ordered_list
            
        
        cols = num_idxs if imputer is None else _compute_features(imputer, num_idxs, columns)
        
        with open(training_file, 'r') as dfile:
            dfile.seek(block_offset)
            contains = lambda idx: idx in cols
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            dfX_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, n_threads = IO_workers, 
                                    new_columns = list(feature_num_idx_dict.keys()), columns = list(feature_num_idx_dict.values()))
            

        if imputer is not None:
            custom_cols_mapping = _compute_mapping(columns = dfX_train.columns)\
                                  if len(columns) != len(dfX_train.columns)\
                                  else None
            data_np = dfX_train.to_numpy()
            
            imputer.transform(data_np, custom_cols_mapping)
            dfX_train = pl.DataFrame(data_np, schema = dfX_train.columns)
        
        return dfX_train