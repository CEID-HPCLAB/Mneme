import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from time import perf_counter as time
import numpy as np
from typing import List, Dict, Optional, Union
from functools import partial
from Mneme.utils import reduce_ordinal_encoders, _copy_attr
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


__all__ = [
    'ParOrdinalEncoder'
]

class ParOrdinalEncoder(OrdinalEncoder):
    '''
    ParOrdinalEncoder is a parallelized version of the OrdinalEncoder from sklearn.preprocessing. It encodes 
    categorical features as an integer array.

    This class is designed to work with large datasets that do not fit into memory, by processing chunks 
    of the input training data in parallel, making use of distributed computing. It implements methods 
    for fitting the encoder incrementally in parallel, reducing the partial fitted encoders into the 
    final encoder and transforming data.
    '''
    
    def __init__(self, cat_idxs: List[str], data_file: str = None, categories: Union[str, List[str]] = 'auto',
                 dtype: Optional[np.dtype] = np.float64, handle_unknown: Optional[str] = 'error', 
                 unknown_value: Optional[Union[np.float64, int]] = None, 
                 encoded_missing_value: Optional[Union[np.float64, int]] = np.nan) -> None:
      
        super(ParOrdinalEncoder, self).__init__(categories = categories, dtype = dtype, handle_unknown = handle_unknown,
                                                unknown_value = unknown_value, encoded_missing_value = encoded_missing_value)
        
        self.data_file = data_file; self.cat_idxs = cat_idxs; self.ordinal_encoders_ = []


    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Fits a OrdinalEncoder to a chunk of data and stores it.

        This method takes a chunk of data, fits a OrdinalEncoder to the features specified by cat_idxs 
        and stores the partial fitted encoder to the list of encoders.

        Args:
            chunk_data (pl.DataFrame): A chunk of data to be processed.
        
        Returns:
            None
        '''
        
        encoder = OrdinalEncoder(categories = self.categories, dtype = self.dtype, handle_unknown = self.handle_unknown,
                                 unknown_value = self.unknown_value, encoded_missing_value = self.encoded_missing_value)
        
        encoder.fit(chunk_data[self.cat_idxs])
        
        self.ordinal_encoders_.append(encoder)
    

    def reduce(self) -> None:
        '''
        Combines the partial fitted encoders from all chunks into a final encoder.

        This method uses the reduce_ordinal_encoders function to combine the encoders fitted to each chunk of data 
        into a final encoder and copies the attributes of the final encoder to the current ParOrdinalEncoder object. 
        
        Returns:
            None
        '''
        
        final_encoder = reduce_ordinal_encoders(self.ordinal_encoders_)
        _copy_attr(self, final_encoder)
        
        del(self.ordinal_encoders_); del(final_encoder)
    
    
    def get_partial_fit(self) -> OrdinalEncoder:
        '''
        Returns the partial fitted encoder.

        Returns:
            tmp_ordinal_encoders (OrdinalEncoder): The partial fitted encoder.
        '''
        tmp_ordinal_encoders = self.ordinal_encoders_.pop()
        return  tmp_ordinal_encoders
    
    
    def set_partial_fits(self, partial_fits: List[OrdinalEncoder]) -> None:
        '''
        Gather the partial fitted encoders (from the different chunks of data) into the 'ordinal_encoders_' list.
        
        Args:
            partial_fits (List[OrdinalEncoder]): A list of partial fitted encoders.
            
        Returns:
            None
        '''
        
        self.ordinal_encoders_ = list(partial_fits) 
    
    
    def print(self, use_parallel: bool = True) -> None:
        '''
        Print the unique categories of the fitted encoder.
        
        Args:
        use_parallel (bool, optional): If True, the encoder is considered as parallel. Defaults to True.
        
        Returns:
            None
        
        '''
        
        kind_encoder = "poe" if use_parallel else "oe"
        
        for index in range(len(self.categories_)):
            print(f"{kind_encoder}: Feature = {self.feature_names_in_[index]}, Categories = {self.categories_[index]}")

    
    def fit(self, block_reader: Union[Optional[BlockReader], None] = None, 
            num_workers: Optional[int] = 2, IO_workers: Optional[int] = 1, num_blocks: Optional[int] = 100,
            chunk_size: Optional[int] = 5000, imputer: Union[Optional[ParImputer], None] = None) -> None: 
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
            imputer (ParImputer, optional): An imputer to apply to the data before fitting the encoder. Defaults to None.
        
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
            _Mneme_logger.benchmark(f"Sequential Standalone OrdinalEncoder Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
        else:
            
            t0 = time()

            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            block_size, columns, block_offsets, num_blocks, feature_idxs_map = \
            block_reader.block_size, block_reader.columns, block_reader.block_offsets,\
            block_reader.num_blocks, block_reader.feature_idxs_map
    
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            pool = mp.Pool(processes = num_workers)
            partial_func = partial(self._partial_fit, args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            
            self.ordinal_encoders_ = pool.map(partial_func, range(num_blocks))
            
            pool.close(); pool.join()
            
            t1 = time()
            
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Standalone OrdinalEncoder Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")

    
    def _partial_fit(self, index: int, args: tuple) -> OrdinalEncoder:
        '''
        Perform fitting on a specific block of data.

        Args:
            index (int): The index of the block of data to fit the encoder to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer 
                          and the number of IO threads.

        Returns:
            OrdinalEncoder: The encoder fitted to the block of data.
        '''
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        chunk = self._load_chunk(self.data_file, block_size, columns, self.cat_idxs, block_offsets[index], 
                                 feature_idxs_map, imputer, IO_workers)
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            print(f"[INFO] Indicative time for loading the data of a block [Standalone ParOrdinalEncoder]: {t1-t0:.6f}")
        
        encoder = OrdinalEncoder(categories = self.categories, dtype = self.dtype, handle_unknown = self.handle_unknown,
                                 unknown_value = self.unknown_value, encoded_missing_value = self.encoded_missing_value)
        encoder.fit(chunk[self.cat_idxs])
        
        return encoder

    
    def transform(self, data: np.ndarray) -> None:
        '''
        Transform the data in-place using the final fitted encoder.

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

        This method filters the feature indices map based on the indices stored in `self.cat_idxs`. It creates a
        new list containing the numerical indices corresponding to the feature indices stored in `self.cat_idxs`
        and assigns it to `self.feature_idxs_`.

        Args:
            feature_idxs_map (Dict[str, int]): A dictionary mapping feature names to their corresponding numerical indices.

        Returns:
            None
        '''
        
        contains = lambda idx: idx in self.cat_idxs
        feature_cat_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())

        self.feature_idxs_ = [feature_pair[1] for feature_pair in feature_cat_idx_dict]
    
    
    def _load_chunk(self, training_file: str, block_size: int, columns: List[str], cat_idxs: List[str], 
                    block_offset: int, feature_idxs_map: Dict[str, int], imputer: Union[ParImputer, None], IO_workers: int) -> pl.DataFrame:
        '''
        Load a block of data from the training file.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The size of the block of data to read.
            columns (List[str]): The column names in the training data.
            cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
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
        
        
        def _compute_features(imputer: ParImputer, cat_idxs: List[str], columns: List[str]) -> List[str]:
            '''
            This function determines the union of features from the ParImputer object and the cat_idxs of the ParOrdinalEncoder 
            (the features to be encoded) and returns them in the order they appear.

            Args:
                imputer (ParImputer): The ParImputer object.
                cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
                columns (List[str]): The column names in the training data.

            Returns:
                List[str]: A list containing the union of features from the ParImputer and ParOrdinalEncoder in the order they appear.
            '''
            
            features = []
            
            for imputer_idx in imputer.imputers_.keys():
                cols = imputer.imputers_[imputer_idx]["cols"]
                
                if isinstance(cols[0], int):
                    cols = [key for key, value in feature_idxs_map.items() if value in cols] 
                
                features.extend(cols)
            
            features.extend(cat_idxs)
                
            ordered_list = []
            for el in columns:
                if el in features and el not in ordered_list:
                    ordered_list.append(el)
            
            return ordered_list
            
        
        cols = cat_idxs if imputer is None else _compute_features(imputer, cat_idxs, columns)
        
        with open(training_file, 'r') as dfile:
            dfile.seek(block_offset)
            contains = lambda idx: idx in cols
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
 
            dfy_train = pl.read_csv(dfile, has_header = False, n_rows = block_size,  n_threads = IO_workers, 
                                    new_columns = list(feature_num_idx_dict.keys()), columns = list(feature_num_idx_dict.values()))

            
        if imputer is not None:
            custom_cols_mapping = _compute_mapping(columns = dfy_train.columns)\
                                  if len(columns) != len(dfy_train.columns)\
                                  else None
            data_np = dfy_train.to_numpy()
            
            imputer.transform(data_np, custom_cols_mapping)
            dfy_train = pl.DataFrame(data_np, schema = dfy_train.columns)
        
        return dfy_train