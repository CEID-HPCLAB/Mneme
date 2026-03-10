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
    
    def __init__(self, cat_idxs: List[str], data_file: str = None) -> None:
        super(ParLabelEncoder, self).__init__()
        
        self.data_file = data_file; self.cat_idxs = cat_idxs

        self.label_encoders_ = [[] for _ in range(len(self.cat_idxs))]; self.label_encoders = {}


    def process_chunk(self, chunk_data: pl.DataFrame) -> None:
        '''
        Process a chunk of data, fitting a LabelEncoder for each categorical variable in the chunk.

        Args:
            chunk_data (pl.DataFrame): A chunk of the input data.
        
        Returns:
            None
        '''
        
        for index, cat_idx in enumerate(self.cat_idxs):
            encoder = LabelEncoder()
            encoder.fit(chunk_data[cat_idx])

            self.label_encoders_[index] = encoder
    
    
    def reduce(self) -> None:
        '''
        Reduce the partial fitted encoders into the final encoder.

        Returns:
            None
        '''
        
        for index, cat_idx in enumerate(self.cat_idxs):
            fitted_encoder = reduce_label_encoders(self.label_encoders_[index])
            
            self.label_encoders[cat_idx] = fitted_encoder
    
    
    def get_partial_fit(self) -> List[LabelEncoder]:
        '''
        Returns the partially fitted encoders for each categorical variable.

        Returns:
            tmp_label_encoders_ (List[LabelEncoder]): The partially fitted encoders for each categorical variable.
        '''
        
        tmp_label_encoders_ = self.label_encoders_
        self.label_encoders_ = [[] for _ in range(len(self.cat_idxs))]

        return tmp_label_encoders_
    
    
    def set_partial_fits(self, partial_fits : List[List[LabelEncoder]]) -> None:
        '''
        Gather the partially fitted encoders (from the different chunks of data) into the 'label_encoders_' list.

        Args:
            partial_fits (List[List[LabelEncoder]]): A list of lists of partially fitted encoders, one list for each categorical variable.

        Returns:
            None
        '''

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
        
        for cat_index in self.label_encoders.keys():
            feature_encoder = self.label_encoders[cat_index]
            print(f"{kind_encoder}: Feature = {cat_index}, Classes = {feature_encoder.classes_}")
            

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
            
            partial_fits = self._seq_fit(data, imputer)
            
            self.label_encoders_ = [list(encoders) for encoders in list(zip(*partial_fits))]
            
            t1 = time()
            
            self.reduce()
    
            t2 = time()
            _Mneme_logger.benchmark(f"Sequential Standalone LabelEncoder Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
            
        else:
            
            t0 = time()

            block_reader = block_reader if block_reader is not None else BlockReader(self.data_file, method = 5, num_blocks = num_blocks)
            
            block_size, columns, block_offsets, num_blocks, feature_idxs_map = \
            block_reader.block_size, block_reader.columns, block_reader.block_offsets,\
            block_reader.num_blocks, block_reader.feature_idxs_map
            
            self._set_feature_mapping(block_reader.feature_idxs_map)
            
            pool = mp.Pool(processes = num_workers)

            partial_func = partial(self._partial_fit, args = (block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers))
            results = pool.map(partial_func, range(num_blocks))
            
            pool.close(); pool.join()
            
            self.label_encoders_ = [list(encoders) for encoders in list(zip(*results))]
            
            t1 = time()
            
            self.reduce()
            
            t2 = time()
            _Mneme_logger.benchmark(f"Parallel Standalone LabelEncoder Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")

    
    def _seq_fit(self, data: pd.DataFrame, imputer: Union[Optional[ParImputer], None]) -> List[List[LabelEncoder]]:
        '''
        Sequentially fit the encoders to each categorical feature of the data.

        Args:
            data (pd.DataFrame): The data to fit the encoders to.
            imputer (ParImputer, optional): An imputer to apply to the data before fitting the encoders. Defaults to None.

        Returns:
            List[List[LabelEncoder]]: A list of lists of partial fitted encoders, one list for each categorical feature.
        '''
    
        partial_fits = []
        
        if imputer is not None:
                 for chunk_data in data:
                    partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
                    
                    chunk_data_np = chunk_data.to_numpy()
                    imputer.transform(chunk_data_np)
                    
                    chunk_data = pd.DataFrame(chunk_data_np, columns = chunk_data.columns, copy = False)
                    
                    for index, cat_idx in enumerate(self.cat_idxs):
                        encoder = LabelEncoder()
                        encoder.fit(chunk_data[cat_idx])
                        
                        partial_fit_enc[index] = encoder
                    
                    partial_fits.append(partial_fit_enc)
        
        else: 
            for chunk_data in data:
                partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
                
                for index, cat_idx in enumerate(self.cat_idxs):
                    encoder = LabelEncoder()
                    encoder.fit(chunk_data[cat_idx])
                    
                    partial_fit_enc[index] = encoder
                
                partial_fits.append(partial_fit_enc)
    
    
        return partial_fits
    
    
    def _partial_fit(self, index: int, args: tuple) -> List[LabelEncoder]:
        '''
        Perform partial fitting on a specific block of data.

        Args:
            index (int): The index of the block of data to fit the encoders to.
            args (tuple): A tuple containing the block size, column names, block offsets, feature index map, imputer
            and the number of IO threads.

        Returns:
            List[LabelEncoder]: A list of encoders fitted to the block of data, one for each categorical feature.
        '''
        
        partial_fit_enc = [[] for _ in range (len(self.cat_idxs))]
        
        block_size, columns, block_offsets, feature_idxs_map, imputer, IO_workers = args
        
        t0 = time()
        
        chunk = self._load_chunk(self.data_file, block_size, columns, self.cat_idxs, block_offsets[index], feature_idxs_map, imputer, IO_workers)
        
        t1 = time()
        if block_offsets[index] == block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [Standalone ParLabelEncoder]: {t1-t0:.6f}")
        
        for index, cat_idx in enumerate(self.cat_idxs):
            encoder = LabelEncoder()
            encoder.fit(chunk[cat_idx])
            
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
        
        transformed_features = []
        
        warnings.filterwarnings("ignore", category = UserWarning)
        
        for index, cat_idx in enumerate(self.cat_idxs):   
            feature_encoder = self.label_encoders[cat_idx]
            transformed_features.append(feature_encoder.transform(data[:, self.feature_idxs_[index]]))
        
        # Replace the original categorical features in the data with the transformed features
        # Stack the transformed features (which is a sequence of ndarrays) vertically (row wise) to create a 
        # 2D array and then transpose it. This results in an array where each column corresponds to a transformed feature.
        data[:, self.feature_idxs_] =  np.vstack(transformed_features).T

        warnings.resetwarnings()
    
    
    def _set_feature_mapping(self, feature_idxs_map: Dict[str, int]) -> None:
        '''
        Set the feature mapping based on the provided feature indices map.

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
            # Move the file pointer to the specified block offset
            dfile.seek(block_offset)
            
            contains = lambda idx: idx in cols
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            dfy_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, n_threads = IO_workers, 
                                    new_columns = list(feature_num_idx_dict.keys()), columns = list(feature_num_idx_dict.values()))
        

        if imputer is not None:
            custom_cols_mapping = _compute_mapping(columns = dfy_train.columns)\
                                  if len(columns) != len(dfy_train.columns)\
                                  else None
            
            data_np = dfy_train.to_numpy()
            
            imputer.transform(data_np, custom_cols_mapping)
            dfy_train = pl.DataFrame(data_np, schema = dfy_train.columns)
        
        return dfy_train