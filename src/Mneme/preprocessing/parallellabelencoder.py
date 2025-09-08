import pandas as pd
from sklearn.preprocessing import LabelEncoder
from time import perf_counter as time
import numpy as np
from typing import List, Dict, Optional, Union
from functools import partial
from Mneme.utils import reduce_label_encoders
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer
import warnings
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


__all__ = [
    'ParLabelEncoder'
]

class ParLabelEncoder(LabelEncoder):
    '''
    ParLabelEncoder is a parallelized version of the LabelEncoder from sklearn.preprocessing. 
    It encodes categorical labels with value between 0 and n_classes-1.

    This class is designed to work with large datasets that do not fit into memory, by processing chunks 
    of the input training data in parallel, making use of distributed computing. It implements methods 
    for fitting the encoder incrementally in parallel, reducing the partial fitted encoders into the 
    final encoder and transforming data.
    '''
    
    def __init__(self, data_file: str, cat_idxs: List[str]) -> None:
        '''
        Initialize the ParLabelEncoder object.

        Args:
            data_file (str): The path to the data file that will be processed.
            cat_idxs (List[str]): A list of column names that should be treated as categorical variables.

        Attributes:
            label_encoders_ (list): A list of LabelEncoder objects, one for each categorical variable.
            label_encoders (dict): A dictionary that will hold the final fitted LabelEncoder objects.
        
        Returns:
            None
        '''
        
        # Call the parent class's initializer
        super(ParLabelEncoder, self).__init__()
        
        self.data_file = data_file
        self.cat_idxs = cat_idxs

        self.label_encoders_ = [[] for _ in range(len(self.cat_idxs))]
        self.label_encoders = {}


    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Process a chunk of data, fitting a LabelEncoder for each categorical variable in the chunk.
        
        This method iterates over the categorical indices (cat_idxs), fits a LabelEncoder to the 
        corresponding column in the chunk of data and stores the fitted encoder in the label_encoders_ list.

        Args:
            chunk_data (pl.DataFrame): A chunk of the input data.
        
        Returns:
            None
        '''
        
        # Iterate over the categorical indices
        for index, cat_idx in enumerate(self.cat_idxs):
           
            encoder = LabelEncoder()
            
            # Fit the encoder to the categorical variable in the chunk
            encoder.fit(chunk_data[cat_idx])
            
            # Store the fitted encoder in the corresponding index of the label_encoders_ list
            self.label_encoders_[index] = encoder
    
    
    def reduce(self) -> None:
        '''
        Reduce the partial fitted encoders into the final encoder.

        This method iterates over the categorical indices (cat_idxs), reduces the corresponding list of 
        partial fitted LabelEncoders (for the specific categorical index) in label_encoders_ into a single fitted encoder using the reduce_label_encoders function
        and stores the final fitted encoder in the label_encoders dictionary with the corresponding categorical index 
        as the key.
        
        Returns:
            None
        '''
        
        # Iterate over the categorical indices
        for index, cat_idx in enumerate(self.cat_idxs):
            # Reduce the list of the partial fitted LabelEncoders of this categorical index into a single final fitted encoder
            fitted_encoder = reduce_label_encoders(self.label_encoders_[index])
            
            # Store the final fitted encoder in the label_encoders dictionary with 
            # the corresponding categorical index as the key
            self.label_encoders[cat_idx] = fitted_encoder
    
    
    def get_partial_fit(self) -> List[LabelEncoder]:
        '''
        Returns the partially fitted encoders for each categorical variable.

        It's important to note that this method also removes the returned encoders from the list. This is necessary 
        to ensure that if the same process takes on another task, the `label_encoders_` list does not still contain the 
        partially fitted encoders from the previous tasks. If they weren't removed, the list would contain all 
        the partially fitted encoders from tasks that the specific process has completed, resulting in us not knowing 
        which are the partially fitted encoders of the specific task and therefore not returning the correct encoders.

        Returns:
            tmp_label_encoders_ (List[LabelEncoder]): The partially fitted encoders for each categorical variable.
        '''
        
        tmp_label_encoders_ = self.label_encoders_
        self.label_encoders_ = [[] for _ in range(len(self.cat_idxs))]

        return tmp_label_encoders_
    
    
    def set_partial_fits(self, partial_fits : List[List[LabelEncoder]]) -> None:
        '''
        Gather the partially fitted encoders (from the different chunks of data) into the 'label_encoders_' list.

        This method takes a list of lists of partially fitted encoders (one list for each categorical variable) 
        and sets the label_encoders_ attribute to this list. The zip function is used to transpose the list of lists, 
        so that each inner list contains the partially fitted encoders for a single categorical variable.

        Args:
            partial_fits (List[List[LabelEncoder]]): A list of lists of partially fitted encoders, one list for each categorical variable.

        Returns:
            None
        '''
        
        # Transpose the list of lists using zip, so that each inner list contains the 
        # partially fitted encoders for a single categorical variable
        partial_fit_encs_per_feature = [list(encoder) for encoder in zip(*partial_fits)]
        
        self.label_encoders_ = partial_fit_encs_per_feature
    
    
    def print(self, use_parallel: bool = True) -> None:
        '''
        For every categorical feature, print the unique classes of the corresponding fitted encoder.
        
        Args:
        use_parallel (bool, optional): If True, the encoder is considered as parallel. Defaults to True.
        
        Returns:
            None
        
        '''
        
        kind_encoder = "ple" if use_parallel else "le"
        
        # Iterate over the keys in the label_encoders dictionary, which are the categorical indices
        for cat_index in self.label_encoders.keys():
            # Get the fitted encoder for the current categorical index and print its classes 
            feature_encoder = self.label_encoders[cat_index]
            print(f"{kind_encoder}: Feature = {cat_index}, Classes = {feature_encoder.classes_}")
            

    def fit(self, block_reader: Union[Optional[BlockReader], None] = None, 
             num_workers: Optional[int] = 2, IO_workers: Optional[int] = 1, num_blocks: Optional[int] = 100, 
             chunk_size: Optional[int] = 5000, imputer: Union[Optional[ParImputer], None] = None) -> None: 
        '''
        Read and fit blocks of data incrementally.
        
        This method reads the data file in chunks and, for every categorical index(feature), it fits the 
        encoder to each chunk. If an imputer is provided, it also applies the imputation to each chunk before 
        fitting the encoder. The method can operate in either sequential or parallel mode, 
        depending on the value of the use_parallel parameter.
        
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
            # Set chunk_size to the block_size of block_reader if it exists, otherwise keep the provided chunk_size
            chunk_size = block_reader.block_size if block_reader is not None else chunk_size
            
            # Read data from the provided CSV file in chunks
            data = pd.read_csv(self.data_file, chunksize = chunk_size)
            
            # If a block reader is not provided, create a new one
            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            # Set the feature mapping by identifying the numerical indexes for the given column names (cat_idxs)
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            t0 = time()
            
            # Gather the partially fitted encoders (from the different chunks of data)
            partial_fits = self._seq_fit(data, imputer)
            
            # Transpose the list of lists of partially fitted encoders (one list for each categorical variable)
            # using zip, so that each inner list contains the partially fitted encoders for a single categorical variable
            self.label_encoders_ = [list(encoders) for encoders in list(zip(*partial_fits))]
            
            t1 = time()
            
            # For every feature, combine the partial fitted encoders into a final fitted encoder
            self.reduce()
    
            t2 = time()
            _Mneme_logger.benchmark(f"Sequential Times of Standalone LabelEncoder -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
        else:
            # Parallel fitting 
            
            t0 = time()
            # If a block reader is not provided, create a new one
            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            block_size, columns, block_offsets, num_blocks, feature_idxs_map = \
            block_reader.block_size, block_reader.columns, block_reader.block_offsets,\
            block_reader.num_blocks, block_reader.feature_idxs_map
            
            # Set the feature mapping by identifying the numerical indexes for the given column names (cat_idxs)
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            pool = mp.Pool(processes = num_workers)
            # Create a partial function for fitting the encoder(s) to the corresponding block of data
            partial_func = partial(self._partial_fit, 
                                   args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            results = pool.map(partial_func, range(num_blocks))
            
            # Close the multiprocessing pool and wait for all processes to finish
            pool.close()
            pool.join()
            
            # Transpose the list of lists of partially fitted encoders (one list for each categorical variable)
            # using zip, so that each inner list contains the partially fitted encoders for a single categorical variable
            self.label_encoders_ = [list(encoders) for encoders in list(zip(*results))]
            
            t1 = time()
            
            # For every feature, combine the partial fitted encoders into a final fitted encoder
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Times of Standalone LabelEncoder -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")

    
    def _seq_fit(self, data: pd.DataFrame, imputer: Union[Optional[ParImputer], None]) -> List[List[LabelEncoder]]:
        '''
        Sequentially fit the encoders to each categorical feature of the data.

        This method reads the data in chunks and fits an encoder to each categorical feature in each chunk. 
        If an imputer is provided, it also applies the imputation to each chunk before fitting the encoders. 
        The method operates in sequential mode.

        Args:
            data (pd.DataFrame): The data to fit the encoders to.
            imputer (ParImputer, optional): An imputer to apply to the data before fitting the encoders. Defaults to None.

        Returns:
            List[List[LabelEncoder]]: A list of lists of partial fitted encoders, one list for each categorical feature.
        '''
    
        partial_fits = []
        
        if imputer is not None:
                 for chunk_data in data:
                    # Initialize a list of lists to store the fitted encoders for each categorical feature
                    partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
                    
                    # Convert the chunk of data to a numpy array 
                    # because the transform method of ParImputer expects a numpy array as input
                    chunk_data_np = chunk_data.to_numpy()
                    imputer.transform(chunk_data_np)
                    
                    # Convert the numpy array back to a DataFrame
                    chunk_data = pd.DataFrame(chunk_data_np, columns = chunk_data.columns, copy = False)
                    
                    # Fit an encoder to each categorical feature in the chunk of data
                    for index, cat_idx in enumerate(self.cat_idxs):
                        encoder = LabelEncoder()
                        encoder.fit(chunk_data[cat_idx])
                        
                        # Store the fitted encoder in the list of fitted encoders for the current categorical feature
                        partial_fit_enc[index] = encoder
                    
                    # Add the list of the partial fitted encoders for the current chunk of data 
                    # to the list of all fitted encoders
                    partial_fits.append(partial_fit_enc)
        
        else: 
            for chunk_data in data:
                # Initialize a list of lists to store the fitted encoders for each categorical feature
                partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
                
                # Fit an encoder to each categorical feature in the chunk of data
                for index, cat_idx in enumerate(self.cat_idxs):
                    encoder = LabelEncoder()
                    encoder.fit(chunk_data[cat_idx])
                    
                    # Store the fitted encoder in the list of fitted encoders for the current categorical feature
                    partial_fit_enc[index] = encoder
                
                # Add the list of the partial fitted encoders for the current chunk of data 
                # to the list of all fitted encoders
                partial_fits.append(partial_fit_enc)
    
    
        return partial_fits
    
    
    def _partial_fit(self, index: int, args: tuple) -> List[LabelEncoder]:
        '''
        Perform partial fitting on a specific block of data.

        This method loads a block of data and fits a LabelEncoder to each categorical feature in the block. The block of data 
        is determined by the value of the index, which specifies the block offset. The fitting is performed only on the 
        columns that correspond to the categorical indices (cat_idxs), not all columns. The method also measures the time 
        it takes to load the data of a block and prints this time for the first block.

        Args:
            index (int): The index of the block of data to fit the encoders to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer
            and the number of IO threads.

        Returns:
            List[LabelEncoder]: A list of encoders fitted to the block of data, one for each categorical feature.
        '''
        
        # Initialize a list of lists to store the fitted encoders for each categorical feature
        partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        # Load the block of data
        chunk = self._load_chunk(self.data_file, block_size, columns, self.cat_idxs, block_offsets[index], 
                                 feature_idxs_map, imputer, IO_workers)
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [Standalone ParLabelEncoder (1st Level)]: {t1-t0:.6f}")
        
        # Fit a LabelEncoder to each categorical feature in the block of data
        for index, cat_idx in enumerate(self.cat_idxs):
            encoder = LabelEncoder()
            encoder.fit(chunk[cat_idx])
            
            # Store the fitted encoder in the list of fitted encoders for the current categorical feature
            partial_fit_enc[index] = encoder
      
        return partial_fit_enc

    
    def transform(self, data: np.ndarray) -> None:
        '''
        Transform the categorical features of the data using the fitted encoders.

        This method applies each fitted encoder to the corresponding categorical feature of the data. The transformed features 
        replace the original features in the data. The method modifies the data in-place.

        Args:
            data (np.ndarray): The data to transform.

        Returns:
            None: [The method modifies the data in-place]
        '''
        
        # Initialize a list to store the transformed features, which will replace the original features in the data
        transformed_features = []
        
        # Ignore warnings about feature_names_in when fitted with Pandas Dataframe containing named columns
        warnings.filterwarnings("ignore", category = UserWarning)
        
        # Loop over each categorical feature index
        for index, cat_idx in enumerate(self.cat_idxs): 
            # Get the fitted encoder for the current categorical feature    
            feature_encoder = self.label_encoders[cat_idx]
            
            # Transform the current categorical feature using the encoder and add it to the list of transformed features
            transformed_features.append(feature_encoder.transform(data[:, self.feature_idxs_[index]]))
        
        # Replace the original categorical features in the data with the transformed features
        # Stack the transformed features (which is a sequence of ndarrays) vertically (row wise) to create a 
        # 2D array and then transpose it. This results in an array where each column corresponds to a transformed feature.
        data[:, self.feature_idxs_] =  np.vstack(transformed_features).T

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
            This function determines the union of features from the ParImputer object and the cat_idxs of the ParLabelEncoder 
            (the features to be encoded) and returns them in the order they appear.

            Args:
                imputer (ParImputer): The ParImputer object.
                cat_idxs (List[str]): A list of strings representing the names of the features to be encoded.
                columns (List[str]): The column names in the training data.

            Returns:
                List[str]: A list containing the union of features from the ParImputer and ParLabelEncoder in the order they appear.
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
            # Create a list containing the union of features from the ParImputer and ParLabelEncoder  
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