import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from time import perf_counter as time
import numpy as np
from typing import List, Dict, Optional, Union
from functools import partial
from Mneme.utils import reduce_onehot_encoders, _copy_attr
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


__all__ = [
    'ParOneHotEncoder'
]

class ParOneHotEncoder(OneHotEncoder):
    '''
    ParOneHotEncoder is a parallelized version of the OneHotEncoder from sklearn.preprocessing. It encodes 
    categorical features as a binary, one-hot encoded array.

    This class is designed to work with large datasets that do not fit into memory, by processing chunks 
    of the input training data in parallel, making use of distributed computing. It implements methods 
    for fitting the encoder incrementally in parallel, reducing the partial fitted encoders into the 
    final encoder and transforming data.
    '''
    
    def __init__(self, data_file: str, cat_idxs: List[str], categories: Union[str, List[str]] = 'auto', 
                 drop: Optional[Union[str, np.ndarray]] = None, sparse_output: Optional[bool] = True, 
                 dtype: Optional[np.dtype] = np.float64, handle_unknown: Optional[str] = 'error') -> None:
        '''
        Initialize the ParOneHotEncoder object.

        This constructor method initializes the ParOneHotEncoder object by calling the constructor of the superclass 
        (OneHotEncoder) with the provided parameters and initializing additional attributes specific to the 
        ParOneHotEncoder class.

        Args:
            data_file (str): The path to the training file containing the dataset.
            
            cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
            
            categories (Union[str, List[str]], optional): The categories to be used for encoding. 
            If 'auto', the categories are determined from the data. Defaults to 'auto'.
            
            drop (Optional[Union[str, np.ndarray]], optional): Specifies a methodology for dropping one of the
            categories per feature. If None, no category is dropped. If 'first', the first category is dropped.
            If 'if_binary', a category is dropped for binary features. If an array, the indices of the categories
            to drop. Defaults to None.
            
            sparse_output (Optional[bool], optional): Whether to return a sparse matrix. Defaults to True.
            
            dtype (Optional[np.dtype], optional): The desired data type of the output. Defaults to np.float64.
            
            handle_unknown (Optional[str], optional): The method to handle unknown categories. 
            If 'error', an error is raised. Defaults to 'error'.
        
        Returns:
            None
        '''
        
        # Call the constructor of the superclass (OneHotEncoder)
        super(ParOneHotEncoder, self).__init__(categories = categories, drop = drop, sparse_output = sparse_output, 
                                               dtype = dtype, handle_unknown = handle_unknown)
        self.data_file = data_file
        self.cat_idxs = cat_idxs
        self.onehot_encoders_ = []


    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Fits a OneHotEncoder to a chunk of data and stores it.

        This method takes a chunk of data, fits a OneHotEncoder to the features specified by cat_idxs 
        and stores the partial fitted encoder to the list of encoders.

        Args:
            chunk_data (pl.DataFrame): A chunk of data to be processed.
        
        Returns:
            None
        '''
        
        encoder = OneHotEncoder(categories = self.categories, sparse_output = self.sparse_output, dtype = self.dtype, 
                                handle_unknown = self.handle_unknown)
        
        # Fit the encoder to the corresponding features in the chunk of data    
        encoder.fit(chunk_data[self.cat_idxs])
        encoder.drop = self.drop
        
        self.onehot_encoders_.append(encoder)
        
            
    def reduce(self) -> None:
        '''
        Combines the partial fitted encoders from all chunks into a final encoder.

        This method uses the reduce_onehot_encoders function to combine the encoders fitted to each chunk of data 
        into a final encoder and copies the attributes of the final encoder to the current ParOneHotEncoder object. 
        
        Returns:
            None
        '''
        
        # Combine the encoders fitted to each chunk of data into a final encoder
        final_encoder = reduce_onehot_encoders(self.onehot_encoders_)
        
        # Copy the attributes of the final encoder to the ParOneHotencoder object
        _copy_attr(self, final_encoder)
        
        # Delete the list of the partial fitted encoders and the (combined) final encoder
        del(self.onehot_encoders_)
        del(final_encoder)
    
    
    def get_partial_fit(self) -> OneHotEncoder:
        '''
        Returns the partial fitted encoder.
        
        It's important to note that this method also removes the returned encoder from the list. This is necessary 
        to ensure that if the same process takes on another task, the `onehot_encoders_` list does not still contain the 
        partially fitted encoders from the previous tasks. If they weren't removed, the list would contain all 
        the partial fitted encoders from tasks that the specific process has completed, resulting in us not knowing 
        which is the partial fitted encoder of the specific task and therefore not returning the correct encoder.

        Returns:
            tmp_onehot_encoders (OneHotEncoder): The partial fitted encoder.
        '''
        tmp_onehot_encoders = self.onehot_encoders_.pop()
        return  tmp_onehot_encoders
    
    
    def set_partial_fits(self, partial_fits: List[OneHotEncoder]) -> None:
        '''
        Gather the partial fitted encoders (from the different chunks of data) into the 'onehot_encoders_' list.
        
        Args:
            partial_fits (List[OneHotEncoder]): A list of partial fitted encoders.
            
        Returns:
            None
        
        '''
        
        self.onehot_encoders_ = list(partial_fits) 
    
    
    def print(self, use_parallel: bool = True) -> None:
        '''
        Print the unique categories of the fitted encoder.
        
        Args:
        use_parallel (bool, optional): If True, the encoder is considered as parallel. Defaults to True.
        
        Returns:
            None
        
        '''
        
        kind_encoder = "pohe" if use_parallel else "ohe"
        
        # Print categories_ attribute of the encoder
        for index in range(len(self.categories_)):
            print(f"{kind_encoder}: Feature = {self.feature_names_in_[index]}, Categories = {self.categories_[index]}")
          
    
    def _fit(self, use_parallel: Optional[bool] = False, block_reader: Union[Optional[BlockReader], None] = None, 
             num_workers: Optional[int] = 2, IO_workers: Optional[int] = 1, num_blocks: Optional[int] = 100, 
             chunk_size: Optional[int] = 5000, imputer: Union[Optional[ParImputer], None] = None) -> None: 
        '''
        Read and fit blocks of data incrementally.
        
        This method reads the data file in chunks and fits the encoder to each chunk. If an imputer is provided, 
        it also applies the imputation to each chunk before fitting the encoder. The method can operate in either 
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
            imputer (ParImputer, optional): An imputer to apply to the data before fitting the encoder. Defaults to None.
        
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
                    # Fit the encoder to the chunk
                    self.process_chunk(chunk_data)
                    
            else:
                # Otherwise, directly process each chunk
                for chunk_data in data:
                    self.process_chunk(chunk_data)
            
            t1 = time()
            
            # Combine the fitted encoders into a final encoder
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Sequential Times of Standalone OneHotEncoder -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
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
            # Create a partial function for fitting the encoders to the corresponding block of data
            partial_func = partial(self._partial_fit, 
                                   args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            self.onehot_encoders_ = pool.map(partial_func, range(num_blocks))
            
            # Close the multiprocessing pool and wait for all processes to finish
            pool.close()
            pool.join()
            
            t1 = time()
            
            # Combine the partial fitted encoders into a final encoder
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Times of Standalone OneHotEncoder -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            

    def _partial_fit(self, index: int, args: tuple) -> OneHotEncoder:
        '''
        Perform fitting on a specific block of data.

        This method loads a block of data, fits a OneHotEncoder on it and returns the fitted encoder. The block of data 
        is determined by the value of the index, which specifies the block offset. The fitting is performed only on the 
        columns that correspond to this specific encoder (cat_idxs), not all columns.

        Args:
            index (int): The index of the block of data to fit the encoder to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer
            and the number of IO workers.

        Returns:
            OneHotEncoder: The encoder fitted to the block of data.
        '''
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        # Load the block of data
        chunk = self._load_chunk(self.data_file, block_size, columns, self.cat_idxs, block_offsets[index], 
                                 feature_idxs_map, imputer, IO_workers)
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [Standalone ParOneHotEncoder (1st Level)]: {t1-t0:.6f}")
        
        # Fit an OneHotEncoder on the loaded chunk and return the encoder
        encoder = OneHotEncoder(categories = self.categories, sparse_output = self.sparse_output, dtype = self.dtype, 
                                handle_unknown = self.handle_unknown)
        encoder.fit(chunk[self.cat_idxs])
        encoder.drop = self.drop
        
        return encoder
    
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        '''
        Transform the data (NOT in-place) using the final fitted encoder.

        This method applies the transformation to the specified columns of the data array in place. It first extracts
        the columns specified by `self.feature_idxs_` and transforms them using the inherited `transform` method
        from the superclass.

        Args:
            data (numpy.ndarray): The input data array to be transformed.

        Returns:
            numpy.ndarray: The transformed data array.
        '''
        
        # Ignore warnings about feature_names_in when fitted with Pandas Dataframe containing named columns
        warnings.filterwarnings("ignore", category = UserWarning)
        
        # Extract the specified columns for transformation and apply the transformation 
        transformed_data = super().transform(data[:, self.feature_idxs_])

        warnings.resetwarnings()
        
        return transformed_data


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
        
        # Define a lambda (helper) function to check if an index is present in self.cat_idxs
        contains = lambda idx: idx in self.cat_idxs
        
        # Filter the feature indices map based on the indices stored in self.cat_idxs
        feature_cat_idx_dict = filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items())
        
        # Store the numerical indices corresponding to the feature indices stored in self.cat_idxs
        self.feature_idxs_ = [feature_pair[1] for feature_pair in feature_cat_idx_dict]
    
    
    def _load_chunk(self, training_file: str, block_size: int, columns: List[str], cat_idxs: List[str], 
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
            cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
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
        
        
        def _compute_features(imputer: ParImputer, cat_idxs: List[str], columns: List[str]) -> List[str]:
            '''
            This function determines the union of features from the ParImputer object and the cat_idxs of the ParOneHotEncoder 
            (the features to be encoded) and returns them in the order they appear.

            Args:
                imputer (ParImputer): The ParImputer object.
                cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
                columns (List[str]): The column names in the training data.

            Returns:
                List[str]: A list containing the union of features from the ParImputer and ParOneHotEncoder in the order they appear.
            '''
            
            features = []
            
            # Iterate over the imputers and extract their columns
            for imputer_idx in imputer.imputers_.keys():
                cols = imputer.imputers_[imputer_idx]["cols"]
                
                # If columns are represented by numerical indices, convert them to column names
                if isinstance(cols[0], int):
                    cols = [key for key, value in feature_idxs_map.items() if value in cols] 
                
                features.extend(cols)
            
            features.extend(cat_idxs)
                
            ordered_list = []
            # Create a list containing the union of features from the ParImputer and ParOneHotEncoder  
            # in the order they appear.
            for el in columns:
                if el in features and el not in ordered_list:
                    ordered_list.append(el)
            
            return ordered_list
            
        
        cols = cat_idxs if imputer is None else _compute_features(imputer, cat_idxs, columns)
        
        with open(training_file, 'r') as dfile:
            # Move the file pointer to the specified block offset
            dfile.seek(block_offset)
            contains = lambda idx: idx in cols
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            # Read the corresponding block of data from the file. The size of the block is determined by the 'block_size' 
            # parameter. We only read the columns that will be used by the pipeline's preprocessors.
            dfy_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, 
                                    n_threads = IO_workers, new_columns = list(feature_num_idx_dict.keys()),
                                    columns = list(feature_num_idx_dict.values()))
            
            
        # Impute potentially missing values in the block of data if an imputer is provided
        if imputer is not None:
            # Compute custom column mapping if the number of columns in dfy_train differs from the columns of the 
            # total training dataset. This is done when the columns intended for fitting the encoder are fewer 
            # than the columns in the entire dataset

            custom_cols_mapping = _compute_mapping(columns = dfy_train.columns)\
                                  if len(columns) != len(dfy_train.columns)\
                                  else None
            data_np = dfy_train.to_numpy()
            
            # Apply the transformation using the imputer and the updated columns mapping
            imputer.transform(data_np, custom_cols_mapping)
            # Convert the transformed numpy array back to a dataframe with the same columns
            dfy_train = pl.DataFrame(data_np, schema = dfy_train.columns)
        
        return dfy_train