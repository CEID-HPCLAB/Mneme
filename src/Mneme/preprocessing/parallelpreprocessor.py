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


class ParPreprocessor():
    '''
    A class used to perform parallel preprocessing on large datasets.

    This class is designed to handle large datasets that cannot fit into memory. It facilitates parallel preprocessing
    of data blocks using multiple preprocessors (scalers | encoders). It allows for efficient processing of large datasets 
    by dividing the preprocessing tasks into smaller chunks and processing them in parallel using multiple processes. 
    The partial results are then reduced to obtain the final preprocessor (preprocessing pipeline) on the whole dataset.
    '''
    
    def __init__(self, block_reader: BlockReader, 
                 preprocessors: dict = {"InputVar": [], "TargetVar": []},
                 num_workers: int = 2,
                 IO_workers: int = 1,
                 imputer: ParImputer = None) -> None:
        '''
        Initializes a ParPreprocessor object.
        
        This method (initializer) initializes a ParPreprocessor object with the provided parameters. It sets the preprocessors 
        dictionary to the provided preprocessors if any, otherwise it uses the default preprocessors. It also 
        computes the union of features from all preprocessors to be used in subsequent processing.

        Args:
            block_reader (BlockReader): An instance of the BlockReader class. This object is responsible for 
                reading data blocks from a file.
            preprocessors (dict, optional): A dictionary mapping variable types ("InputVar", "TargetVar") to a list 
                of preprocessor instances. These preprocessors are applied to the corresponding variable type. 
                If not provided, defaults to an empty list of preprocessors for both "InputVar" and "TargetVar".
            num_workers (int, optional): The number of worker processes to use for parallel processing. This determines 
                the number of data blocks that can be processed simultaneously. If not provided, defaults to 2.
            IO_workers (int, optional): The number of threads to use for IO operations. If not provided, defaults to 1.
            imputer: An optional imputer object used to handle missing values in the data. This object should provide 
                a transform method. If not provided, defaults to None.
        '''
     
        default_preprocessors = {"InputVar": [], "TargetVar": []}

        # Merge the default preprocessors with the provided preprocessors. If a preprocessor is provided for a 
        # variable type ("InputVar" or "TargetVar"), it will replace the default preprocessor for that variable type.
        self.preprocessors = {**default_preprocessors, **preprocessors}
        
        self.block_reader = block_reader
        self.num_workers = num_workers
        self.IO_workers = IO_workers
        self.imputer = imputer

        self._features = self._get_union_features()
        

    def process_chunk(self, block_offset: int) -> dict:
        '''
        Processes a block (chunk) of data from the training file.

        Args:
            block_offset (int): The offset of the block to be processed. This is used to determine the starting point 
                for reading the data block from the training file.
        
        Returns:
            dict: A dictionary containing the fitted preprocessors for each preprocessor applied to the specific block of data.

        This method reads a block of data from the training file, starting at the specified block offset. It then 
        applies each input variable preprocessor to the input variables (X) and each output variable preprocessor 
        to the corresponding output variable(s) (y). The partial fit of each preprocessor is stored in a dictionary, with the key 
        being the name of the preprocessor and the index of the preprocessor in the list of partial fitted preprocessors for its 
        variable type. 
        
        Note: The input and output data (X and y) are deleted after processing to free up memory.
        '''
        
        # Get necessary attributes from the block reader
        training_file = self.block_reader.training_file
        block_size = self.block_reader.block_size
        columns = self.block_reader.columns
        target_label_name = self.block_reader.target_label_name
        
        # Initialize a dictionary to store the (partial) fit of each preprocessor for the specific block
        partial_fits = {}
        for index, preprocessor in enumerate(reduce(operator.add, self.preprocessors.values())):
            key_name = type(preprocessor).__name__ + f"_{index}"
            partial_fits[key_name] = list()   

        # Load the block of data from the training file that corresponds to the specific offset
        X, y = self._load_batch_core(training_file, block_size, target_label_name, block_offset)

        # Process the input variables with each input variable preprocessor
        for index, in_var_preprocessor in enumerate(self.preprocessors["InputVar"]):
            t0 = time()
            in_var_preprocessor.process_chunk(X)
            t1 = time()
            
            if block_offset == self.block_reader.block_offsets[0]:
                _Mneme_logger.benchmark(f"[INFO] Time to fit a block [InputVar] [ParPreprocessor -> {type(in_var_preprocessor)}]: {t1-t0:.6f}")
            
            # Store the partial fit of the preprocessor
            in_var_preprocessor_name = type(in_var_preprocessor).__name__ + f"_{index}"
            partial_fits[in_var_preprocessor_name].append(in_var_preprocessor.get_partial_fit())

        # Process the target variable with each target variable preprocessor
        for index, out_var_preprocessor in enumerate(self.preprocessors["TargetVar"], start=len(self.preprocessors["InputVar"])):
            t0 = time()
            out_var_preprocessor.process_chunk(y)
            t1 = time()
            
            if block_offset == self.block_reader.block_offsets[0]:
                _Mneme_logger.benchmark(f"[INFO] Time to fit a block [TargetVar] [ParPreprocessor -> {type(out_var_preprocessor)}]: {t1-t0:.6f}")
                
            # Store the partial fit of the preprocessor
            out_var_preprocessor_name = type(out_var_preprocessor).__name__ + f"_{index}"
            partial_fits[out_var_preprocessor_name].append(out_var_preprocessor.get_partial_fit())

        # Delete the input and target data to free up memory
        del X, y

        return partial_fits
    
 
    def reduce(self) -> None:
        '''
        Reduces the partial fits of all preprocessors.
        
        This method iterates over all preprocessors (both for input and output (target) variables) defined in 
        the ParPreprocessor instance and applies the reduce method to each preprocessor. 
        The reduce method is responsible for combining partial fits obtained from different data blocks 
        to produce a final fit on the entire dataset. This is done after all blocks of data have been processed.

        Returns:
            None
        '''
        
        # Iterate over all preprocessors and apply the reduce method to each preprocessor
        for preprocessor in reduce(operator.add, self.preprocessors.values()): 
            preprocessor.reduce()

 
    def parallel_fit(self) -> None:
        '''
        Fits a series of preprocessors in parallel using multiple processes.

        The method uses the multiprocessing library to create a pool of worker processes. Each worker process is 
        responsible for fitting a chunk of data using a specific pipeline of preprocessors. The results of these fits are stored 
        in a list of dictionaries, where each dictionary contains the partial fits produced by a worker process.

        After all the worker processes have completed their tasks, the method collects the partial fits from the list of 
        dictionaries and extends the corresponding lists in the `gathered_partial_fits` dictionary.

        After all partial fits have been gathered, the method sets the partial fits of each preprocessor and 
        calls the reduce method to combine the partial fits into a final fit.
        
        Note: This function uses the 'multiprocessing' library and is designed to work on systems with multiple cores.
        
        Returns:
            None
        '''
        
        t0 = time()
        
        gathered_partial_fits = {}
        
        # For each preprocessor, set the feature mapping and initialize an empty list for its partial fits
        for index, preprocessor in enumerate(reduce(operator.add, self.preprocessors.values())):
            preprocessor._set_feature_mapping(self.block_reader.feature_idxs_map)
            key_name = type(preprocessor).__name__ + f"_{index}"
            gathered_partial_fits[key_name] = list()
        
        pool = mp.Pool(processes = self.num_workers)
        partial_fits = pool.map(self.process_chunk, self.block_reader.block_offsets)
        
        # Close the multiprocessing pool and wait for all processes to finish
        pool.close()
        pool.join()
    
        t1 = time()
        
        # The first solution iterates through each partial_fit_dict in the list of partial_fits and extends the 
        # corresponding lists in the gathered partial fits dictionary.
        for partial_fit_dict in partial_fits:
            for key in partial_fit_dict.keys():
                gathered_partial_fits[key].extend(partial_fit_dict[key])

        # The second solution is equivalent to the first one. It uses a dictionary comprehension and 
        # the sum() function to concatenate lists of partial fits.
        # gathered_partial_fits = {key: sum([partial_fit_dict[key] for partial_fit_dict in partial_fits], []) for key in partial_fits.pop().keys()}
        
        # For each preprocessor, define its partial fits 
        for index, preprocessor in enumerate(reduce(operator.add, self.preprocessors.values())):
            key_name = type(preprocessor).__name__ + f"_{index}"
            preprocessor.set_partial_fits(gathered_partial_fits[key_name])
        
        # Reduce the partial fits of preprocessors to final fits
        self.reduce()
        
        t2 = time()
        _Mneme_logger.benchmark(f"Parallel Times of Preprocessing Pipeline -> Fit: {t1-t0:.6f} | Reduce: {t2-t1:.6f}")
        

    def print(self) -> None:
        '''
        Prints the details (information) of each preprocessor in the pipeline.
        
        Returns:
            None
        '''
        
        # For each preprocessor in the pipeline
        for preprocessor in reduce(operator.add, self.preprocessors.values()):
            # Print the details of the preprocessor
            preprocessor.print()
            print("-*-"*10)
            
    
    def transform(self, data: np.ndarray) -> np.ndarray | tuple:
        '''
        Transforms the input data using the pipeline's preprocessors.

        If an OneHotEncoder is present in the pipeline, the method returns a tuple containing the transformed 
        data and the one-hot encoded data. Otherwise, it returns the transformed data only.

        Args:
            data (numpy.ndarray): The input data to be transformed.

        Returns:
            numpy.ndarray or tuple: The transformed data. If OneHotEncoder was used, a tuple is returned.
        '''
        
        transformed_data = np.copy(data)
        onehot_transformed_data = None
        
        # For each preprocessor in the pipeline
        for preprocessor in reduce(operator.add, self.preprocessors.values()):
            if not isinstance(preprocessor, ParOneHotEncoder):
                # Apply the preprocessor's transform method to the data
                preprocessor.transform(transformed_data)
            else:
                # If the preprocessor is a one-hot encoder, apply its transform method and store the result separately
                onehot_transformed_data = preprocessor.transform(transformed_data)
        
        out = transformed_data
        
        if onehot_transformed_data is not None:
            # Set the output as a tuple containing the transformed data and the one-hot encoded data
            out = (transformed_data, onehot_transformed_data)
       
        return out
    
    
    def _get_union_features(self) -> list:
        '''
        Retrieves the features processed by the imputer and those to be processed by the pipeline's preprocessors.

        The method compiles a list of features, which includes those already processed by the imputer and those 
        that will be processed by the pipeline's preprocessors. It then returns an ordered list of these features, 
        ensuring each feature is only included once.

        Returns:
            list: An ordered list of features, encompassing both the ones that have been processed by the imputer 
            and the ones that will be processed by the pipeline's preprocessors, ensuring each feature is only included once.
        '''
        
        union_features = [] 
        
        if self.imputer is not None:
            for imputer_idx in self.imputer.imputers_.keys():
                # Get the columns processed by the imputer
                cols = self.imputer.imputers_[imputer_idx]["cols"]
                
                # If the columns are represented as integers, map them back to their original names
                if isinstance(cols[0], int):
                    cols = [key for key, value in self.block_reader.feature_idxs_map.items() if value in cols] 
                
                # Add the columns to the list of union features
                union_features.extend(cols)
        
        # For each preprocessor in the pipeline
        for preprocessor in reduce(operator.add, self.preprocessors.values()):
            try:
                # Try to get the numerical indices processed by the preprocessor 
                num_idxs = getattr(preprocessor, "num_idxs")
                union_features.extend(num_idxs)
            except AttributeError as _:
                # If the preprocessor does not process numerical indices, 
                # it's not a scaler and therefore it's an encoder, so it has cat_idxs
                cat_idxs = getattr(preprocessor, "cat_idxs")
                union_features.extend(cat_idxs)
    
        ordered_list = []
        # For each column in the (input) training file
        for col in self.block_reader.columns:
            if col in union_features and col not in ordered_list:
               ordered_list.append(col)
        
        return ordered_list
      
        
    def _load_batch_core(self, training_file: str, block_size: int, 
                        target_label_name: list, block_offset: int) -> tuple:
        '''
         Load a batch (block) of data from a training file.

        This method opens the training file, seeks to the specified offset and reads a block of data. 
        It then applies any necessary transformations using the imputer (if imputer is provided). 
        If there are preprocessors for the target variable, it separates the target labels from the input features. 
        Finally, it returns the input features and the target labels.

        Args:
            training_file (str): The path to the training file.
            block_size (int): The number of rows to read from the file.
            target_label_name (list): The name of the target label column(s).
            block_offset (int): The offset from the beginning of the file.

        Returns:
            tuple: A tuple containing the input features (X) and the target labels (y).
        '''     
        
        # Define a helper function to compute a mapping from column names to their numeric indices
        # This mapping is useful because we need to refer to columns by their index rather than 
        # their name (e.g. when we have a numpy array)
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
        
        t0 = time()
        
        with open(training_file, 'r') as dfile:
            # Seek to the specified offset in the file
            dfile.seek(block_offset)
            # Compute required feature indices for the block as a dictionary
            # These indices correspond to the original feature set of the dataset.
            feature_idxs_map = self.block_reader.feature_idxs_map
            contains = lambda idx: idx in self._features
            feature_num_idx_dict = dict(filter(lambda feature_pair: contains(feature_pair[0]), feature_idxs_map.items()))
           
            # Read the corresponding block of data from the file. The size of the block is determined by the 'block_size' 
            # parameter. We only read the columns that will be used by the pipeline's preprocessors.
            dfX_train = pl.read_csv(dfile, has_header = False, n_rows = block_size, 
                                    n_threads = self.IO_workers, new_columns = list(feature_num_idx_dict.keys()),
                                    columns = list(feature_num_idx_dict.values()), schema_overrides = {"column_10": pl.Float64})

            # print("----------------")
            # print(dfX_train.columns)
        
        
        # If an imputer is provided, apply it to the data
        if self.imputer is not None:
            # Compute a new column mapping if the number of columns in the block reader and the loaded dataframe are different
            custom_cols_mapping = _compute_mapping(columns = dfX_train.columns) if len(self.block_reader.columns) != len(dfX_train.columns) else None
            data_np = dfX_train.to_numpy()
            # Transform the data using the provided imputer and the updated column mapping
            self.imputer.transform(data_np, custom_cols_mapping)
            dfX_train = pl.DataFrame(data_np, schema = dfX_train.columns)
            
        y = []
        
        # If there are preprocessors for the target variable(s), separate the target labels from the input features
        if len(self.preprocessors["TargetVar"]) > 0:
            dfy_train = pl.DataFrame(dfX_train[target_label_name])
            dfy_train.columns = target_label_name
            # Remove the target labels from the input features
            #polars
            dfX_train = dfX_train.drop([*target_label_name])
           
    
            y = dfy_train
            
      
        X = dfX_train
        t1 = time()
        
        if block_offset == self.block_reader.block_offsets[0]:
            _Mneme_logger.benchmark(f"[INFO] Indicative time for loading the data of a block [ParPreprocessor]: {t1-t0:.6f}")

        return X, y