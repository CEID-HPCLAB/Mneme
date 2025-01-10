from Mneme.preprocessing import ParImputer
from Mneme.utils import reduce_imputers
from sklearn.impute import SimpleImputer
from Mneme import BlockReader
from copy import deepcopy
from typing import List, Any, Union, Tuple
import pytest
import os
import pandas as pd
import numpy as np

'''
This file contains tests for comparing results between parallel and sequential imputers.

The tests compare the results of parallel imputers with their sequential counterparts to ensure that parallel 
processing does not affect the outcomes.

This verification ensures the reliability of parallel processing compared to sequential processing.
'''

@pytest.fixture(scope = "module")
def datafile_path() -> str:
    '''
    Pytest fixture to provide the path to the data file.
    
    This fixture constructs the path to a data file located in the "data" directory 
    relative to the directory of the current file. The fixture has module scope, so the path is constructed only once 
    per test module.

    Returns:
        str: The path to the data file.
    '''
    
    # Construct the path to the data file
    datafile_name = "sample_data.csv"
    
    datafile_path = os.path.join(os.path.dirname(__file__), "data", datafile_name)
    
    # Return the path to the data file    
    return datafile_path


@pytest.fixture(scope = "module")
def generate_BlockReader(datafile_path) -> BlockReader:
    '''
    Pytest fixture to provide a BlockReader instance.
    
    This fixture creates a BlockReader instance with the provided data file path. 
    The fixture has module scope, so the BlockReader instance is created only once per test module.

    Args:
        datafile_path (str): The path to the data file.

    Returns:
        BlockReader: The created BlockReader instance.
    '''
    
    # Create a BlockReader instance with the provided data file path 
    block_reader = BlockReader(training_file = datafile_path, num_blocks = 100)
   
    return block_reader 


def generate_imputers_params() -> List[Tuple[SimpleImputer, List[str]]]:
    '''
    Generate a list of tuples, each containing an instance of SimpleImputer and a list of feature indices.

    Returns:
        List[Tuple[SimpleImputer, List[str]]]: A list of tuples. Each tuple contains a SimpleImputer 
                                               instance and a list of feature indices.
    '''

    imputers = []
    
    num_idxs = [f"x{i}" for i in range(3)]
    
    imputers.append((SimpleImputer(strategy = "mean"), num_idxs))
    
    imputers.append((SimpleImputer(strategy = "most_frequent"), num_idxs[:2]))
    
    return imputers


@pytest.fixture(scope = "module", params = generate_imputers_params())
def pipeline_imputers(request: pytest.FixtureRequest, 
                      generate_BlockReader: BlockReader) -> Tuple[ParImputer, SimpleImputer]:
    '''
    Pytest fixture that generates a tuple of ParImputer and SimpleImputer instances.

    Args:
        request (pytest.FixtureRequest): The fixture request object.
        generate_BlockReader (BlockReader): The block reader to use.

    Returns:
        Tuple[ParImputer, SimpleImputer]: A tuple of ParImputer and SimpleImputer instances.
    '''
    
    par_imputers = ParImputer(custom_block_reader = generate_BlockReader, imputers_map = {request.param[0]: request.param[1]})
    
    # Create a deep copy of the provided imputer
    seq_imputer = deepcopy(request.param[0])
    
    return par_imputers, seq_imputer


def fit_seq_imputer(seq_imputer: SimpleImputer, imputer_features: List[Union[int, str]], data_file: str, 
                    block_reader: BlockReader) -> SimpleImputer:
    
        '''
        Fit a SimpleImputer to chunks of data from a CSV file.

        The data is read in chunks from the CSV file. Each chunk is used to fit a deep copy of the provided SimpleImputer.
        The partial fitted imputers are then reduced to a single final fitted imputer using the provided strategy.

        Args:
            seq_imputer (SimpleImputer): The imputer to fit to the data.
            imputer_features (List[Union[int, str]]): The features to use for imputation.
            data_file (str): The path to the CSV file containing the data.
            block_reader (BlockReader): The block reader to get the chunk size from.

        Returns:
            SimpleImputer: The final fitted imputer.
        '''
        
        # Get the appropriate block size from the block reader 
        chunk_size = block_reader.block_size
        
        data = pd.read_csv(data_file, chunksize = chunk_size, usecols=imputer_features)
        
        partial_imputer_fits = []
        
        for chunk in data:
            seq_imputer = deepcopy(seq_imputer)
            seq_imputer.fit(chunk)
            
            seq_imputer.n_samples_seen_ = chunk.shape[0]
            
            # If the imputer's strategy is 'mean', calculate the number of NaN values in the chunk
            if seq_imputer.strategy == 'mean':
                seq_imputer.nan_vals = np.array(np.isnan(chunk).sum())
            partial_imputer_fits.append(seq_imputer)
        
        
        # Reduce the fitted imputers to a single imputer using the provided strategy
        fitted_imputer = reduce_imputers(partial_imputer_fits, strategy = seq_imputer.strategy)
            
        return fitted_imputer


def test_imputers_results(pipeline_imputers: Tuple[ParImputer, SimpleImputer], 
                          datafile_path: str, generate_BlockReader: BlockReader) -> None:
    '''
    Test the results of two imputers.

    This function fits two imputers to the data and then compares their attributes. 

    Args:
        pipeline_imputers (Tuple[ParImputer, SimpleImputer]): The imputers to test.
        datafile_path (str): The path to the data file.
        generate_BlockReader (BlockReader): The block reader to use.

    Returns:
        None
    '''
  
    # Unpack the imputers
    par_imputers, seq_imputer = pipeline_imputers
    
    # Fit the parallel imputer
    par_imputers.parallel_fit()
    
    # Get the fitted imputer from the parallel imputer
    par_imputer = par_imputers.reduced_imputers.pop(0)
    
    # Get the features used by the first imputer in the dictionary of imputers
    imputer_features = list(par_imputers.imputers_.values())[0]['cols']
    
    # Fit the sequential imputer
    seq_imputer = fit_seq_imputer(seq_imputer, imputer_features, datafile_path, generate_BlockReader)
    
    eps = 1e-10
    
    # Compare the attributes of the two imputers
    for par_attr, seq_attr in zip(vars(par_imputer), vars(seq_imputer)):
        
        par_attr_val = getattr(par_imputer, par_attr)
        seq_attr_val = getattr(seq_imputer, seq_attr)
        
        if par_attr == 'statistics_':
            # Remove NaN values from the attribute values
            par_attr_val = par_attr_val[~np.isnan(par_attr_val)]
            seq_attr_val = seq_attr_val[~np.isnan(seq_attr_val)]
            assert (np.abs(par_attr_val - seq_attr_val) <= eps).all()
        
        # If the attribute values are numpy arrays, check if all elements are equal
        elif isinstance(par_attr_val, np.ndarray) and isinstance(seq_attr_val, np.ndarray):        
            assert (par_attr_val == seq_attr_val).all()
        
        else:
            # If the attribute values are floats and the par_attr_val is NaN, check if seq_attr_val is also NaN
            if isinstance(par_attr_val, float) and isinstance(seq_attr_val, float): 
                    if np.isnan(par_attr_val):
                        assert np.isnan(seq_attr_val)
                    continue 
                
            assert  par_attr_val == seq_attr_val