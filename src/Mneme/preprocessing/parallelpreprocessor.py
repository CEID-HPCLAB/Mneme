import pandas as pd
import numpy as np
from time import perf_counter as time
import operator
from functools import reduce, partial
from Mneme import BlockReader
from Mneme.preprocessing import ParImputer, ParOneHotEncoder
import multiprocessing as mp
import polars as pl
from .._mneme_logging import _Mneme_logger


class ParallelPipeline():
    '''
    A class used to perform parallel preprocessing on large datasets.

    This class is designed to handle large datasets that cannot fit into memory. It facilitates parallel preprocessing
    of data blocks using multiple operators (scalers | encoders). It allows for efficient processing of large datasets 
    by dividing the preprocessing tasks into smaller chunks and processing them in parallel using multiple processes. 
    The partial results are then reduced to obtain the final preprocessor (preprocessing pipeline) on the whole dataset.
    '''
    
    def __init__(self, operators: dict,  data_file: str, imputer: ParImputer = None) -> None:
        
        default_operators = {"InputFeatures": [], "TargetVar": []}

        # Merge the default operators with the provided operators. If a preprocessor is provided for a 
        # variable type ("InputFeatures" or "TargetVar"), it will replace the default preprocessor for that variable type.
        self.operators = {**default_operators, **operators}
        
        self.data_file = data_file; self.imputer = imputer
        

    def process_chunk(self, block_offset: int) -> dict:
        '''
        Processes a block (chunk) of data from the training file.

        Args:
            block_offset (int): The offset of the block to be processed. This is used to determine the starting point 
                for reading the data block from the training file.
        
        Returns:
            dict: A dictionary containing the fitted operators for each preprocessor applied to the specific block of data.
        '''
        
        training_file = self.block_reader.training_file; block_size = self.block_reader.block_size
        columns = self.block_reader.columns; target_label_name = self.block_reader.target_label_name
        
        partial_fits = {}
        for index, preprocessor in enumerate(reduce(operator.add, self.operators.values())):
            key_name = type(preprocessor).__name__ + f"_{index}"
            partial_fits[key_name] = list()   

        X, y = self._load_batch_core(training_file, block_size, target_label_name, block_offset)

        for index, in_var_preprocessor in enumerate(self.operators["InputFeatures"]):
            t0 = time(); in_var_preprocessor.process_chunk(X); t1 = time()
            
            if block_offset == self.block_reader.block_offsets[0]:
                _Mneme_logger.benchmark(f"[INFO] Time to fit a block [InputFeatures] [ParallelPipeline -> {type(in_var_preprocessor)}]: {t1-t0:.6f}")
            
            in_var_preprocessor_name = type(in_var_preprocessor).__name__ + f"_{index}"
            partial_fits[in_var_preprocessor_name].append(in_var_preprocessor.get_partial_fit())

        for index, out_var_preprocessor in enumerate(self.operators["TargetVar"], start=len(self.operators["InputFeatures"])):
            t0 = time(); out_var_preprocessor.process_chunk(y); t1 = time()
            
            if block_offset == self.block_reader.block_offsets[0]:
                _Mneme_logger.benchmark(f"[INFO] Time to fit a block [TargetVar] [ParallelPipeline-> {type(out_var_preprocessor)}]: {t1-t0:.6f}")
                
            out_var_preprocessor_name = type(out_var_preprocessor).__name__ + f"_{index}"
            partial_fits[out_var_preprocessor_name].append(out_var_preprocessor.get_partial_fit())

        del X, y

        return partial_fits
    
 
    def reduce(self) -> None:
        '''
        Reduces the partial fits of all operators.

        Returns:
            None
        '''
        
        for preprocessor in reduce(operator.add, self.operators.values()): 
            preprocessor.reduce()

 
    def fit(self, block_reader: BlockReader, num_workers: int = 2, IO_workers: int = 1) -> None:
        '''
        Fits a series of operators in parallel using multiple processes.
        
        Returns:
            None
        '''

        self.block_reader = block_reader; self.num_workers = num_workers
        self.IO_workers = IO_workers; self._features = self._get_union_features()
        
        t0 = time()
        
        gathered_partial_fits = {}
        
        for index, preprocessor in enumerate(reduce(operator.add, self.operators.values())):
            preprocessor._set_feature_mapping(self.block_reader.feature_idxs_map); preprocessor.data_file = self.data_file
            key_name = type(preprocessor).__name__ + f"_{index}"; gathered_partial_fits[key_name] = list()
        
        pool = mp.Pool(processes = self.num_workers)
        partial_fits = pool.map(self.process_chunk, self.block_reader.block_offsets)
        
        pool.close(); pool.join()
    
        t1 = time()
        
        for partial_fit_dict in partial_fits:
            for key in partial_fit_dict.keys():
                gathered_partial_fits[key].extend(partial_fit_dict[key])
        
        for index, preprocessor in enumerate(reduce(operator.add, self.operators.values())):
            key_name = type(preprocessor).__name__ + f"_{index}"
            preprocessor.set_partial_fits(gathered_partial_fits[key_name])
        
        self.reduce()
        
        t2 = time()
        _Mneme_logger.benchmark(f"Parallel Preprocessing Pipeline Time -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
        

    def print(self) -> None:
        '''
        Prints the details (information) of each preprocessor in the pipeline.
        
        Returns:
            None
        '''
        
        for preprocessor in reduce(operator.add, self.operators.values()):
            preprocessor.print()
            print("-*-"*10)
            
    
    def transform(self, data: np.ndarray) -> np.ndarray | tuple:
        '''
        Transforms the input data using the pipeline's operators.

        If an OneHotEncoder is present in the pipeline, the method returns a tuple containing the transformed 
        data and the one-hot encoded data. Otherwise, it returns the transformed data only.

        Args:
            data (numpy.ndarray): The input data to be transformed.

        Returns:
            numpy.ndarray or tuple: The transformed data. If OneHotEncoder was used, a tuple is returned.
        '''
        
        transformed_data = np.copy(data)
        onehot_transformed_data = None
        
        for preprocessor in reduce(operator.add, self.operators.values()):
            if not isinstance(preprocessor, ParOneHotEncoder):
                preprocessor.transform(transformed_data)
            else:
                onehot_transformed_data = preprocessor.transform(transformed_data)
        
        out = transformed_data
        
        if onehot_transformed_data is not None:
            out = (transformed_data, onehot_transformed_data)
       
        return out
    
    
    def _get_union_features(self) -> list:
        '''
        Retrieves the features processed by the imputer and those to be processed by the pipeline's operators.

        Returns:
            list: An ordered list of features, encompassing both the ones that have been processed by the imputer 
            and the ones that will be processed by the pipeline's operators, ensuring each feature is only included once.
        '''
        
        union_features = [] 
        
        if self.imputer is not None:
            for imputer_idx in self.imputer.imputers_.keys():
                cols = self.imputer.imputers_[imputer_idx]["cols"]
                
                if isinstance(cols[0], int):
                    cols = [key for key, value in self.block_reader.feature_idxs_map.items() if value in cols] 
                
                union_features.extend(cols)
        
        for preprocessor in reduce(operator.add, self.operators.values()):
            try:
                num_idxs = getattr(preprocessor, "num_idxs")
                union_features.extend(num_idxs)
            except AttributeError as _:
                cat_idxs = getattr(preprocessor, "cat_idxs")
                union_features.extend(cat_idxs)
    
        ordered_list = []
        for col in self.block_reader.columns:
            if col in union_features and col not in ordered_list:
               ordered_list.append(col)
        
        return ordered_list
      
        
    def _load_batch_core(self, training_file: str, block_size: int, target_label_name: list, block_offset: int) -> tuple:
        '''
        Load a batch (block) of data from a training file.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The number of rows to read from the file.
            target_label_name (list): The name of the target label column(s).
            block_offset (int): The offset from the beginning of the file.

        Returns:
            tuple: A tuple containing the input features (X) and the target labels (y).
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
        
        t0 = time()
        
        with open(training_file, 'r') as dfile:
            # Seek to the specified offset in the file
            dfile.seek(block_offset)
            # Compute required feature indices for the block as a dictionary
            # These indices correspond to the original feature set of the dataset.
            feature_idxs_map = self.block_reader.feature_idxs_map
            contains = lambda idx: idx in self._features
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            dfX_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, n_threads = self.IO_workers, 
                                    new_columns = list(feature_num_idx_dict.keys()), columns = list(feature_num_idx_dict.values()))

        if self.imputer is not None:
            custom_cols_mapping = _compute_mapping(columns = dfX_train.columns) if len(self.block_reader.columns) != len(dfX_train.columns) else None
            self.imputer._transform(dfX_train, custom_cols_mapping)
            
        y = []
        
        if len(self.operators["TargetVar"]) > 0:
            dfy_train = pl.DataFrame(dfX_train[target_label_name]); dfy_train.columns = target_label_name
            dfX_train = dfX_train.drop([*target_label_name]); y = dfy_train
            
        X = dfX_train
        
        t1 = time()
        
        if block_offset == self.block_reader.block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [ParallelPipeline]: {t1-t0:.6f}")

        return X, y