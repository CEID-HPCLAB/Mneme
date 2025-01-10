from Mneme.preprocessing import (ParStandardScaler,
                                  ParMinMaxScaler,
                                  ParRobustScaler,
                                  ParMaxAbsScaler,
                                  ParMinMaxScaler,
                                  ParPreprocessor,
                                  ParLabelEncoder,
                                  ParOneHotEncoder,
                                  ParOrdinalEncoder,
                                  )
from Mneme import BlockReader
from copy import deepcopy
from typing import List, Any, Union, Tuple
import pytest
import os
import numpy as np


'''
This file contains tests for comparing results between parallel and sequential preprocessors.

The tests compare the results of parallel preprocessors with their sequential counterparts to ensure that parallel 
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


def generate_scalers_params(file_path: str) -> list:
    '''
    Function to generate a list of scaler instances.
    
    This function creates instances of four different types of scalers (ParStandardScaler, 
    ParMinMaxScaler, ParMaxAbsScaler, ParRobustScale), each with the provided data file 
    path and a list of numerical feature indices. The function returns a list of these scaler instances.

    Args:
        file_path (str): The path to the data file.

    Returns:
        List[ParStandardScaler, ParMinMaxScaler, ParMaxAbsScaler, ParRobustScaler]: 
        The list of created scaler instances.
    '''
    
    scalers = []
    
    # Create instances of four different types of scalers and append them to the list
    scalers.append(ParStandardScaler(data_file = file_path, num_idxs = [f"x{i}" for i in range(3)]))
    scalers.append(ParMinMaxScaler(data_file = file_path, num_idxs = [f"x{i}" for i in range(3)]))
    scalers.append(ParMaxAbsScaler(data_file = file_path, num_idxs = [f"x{i}" for i in range(3)]))
    scalers.append(ParRobustScaler(data_file = file_path, num_idxs = [f"x{i}" for i in range(3)]))
    
    # Return the list of created scaler instances
    return scalers


def generate_encoders_params(file_path: str) -> list:
    '''
    Function to generate a list of encoder instances.
    
    This function creates instances of three different types of encoders (ParOneHotEncoder, 
    ParLabelEncoder, ParOrdinalEncoder), each with the provided data file 
    path and a list of categorical feature indices. The function returns a list of these encoder instances.

    Args:
        file_path (str): The path to the data file.

    Returns:
        List[ParOneHotEncoder, ParLabelEncoder, ParOrdinalEncoder]: 
        The list of created encoder instances.
    '''
    
    encoders = []
    
    # Create instances of three different types of encoders and append them to the list
    encoders.append(ParOneHotEncoder(data_file = file_path, cat_idxs = [f"y{i}" for i in range(1,2)]))
    encoders.append(ParLabelEncoder(data_file = file_path, cat_idxs = [f"y{i}" for i in range(1,2)]))
    encoders.append(ParOrdinalEncoder(data_file = file_path, cat_idxs = [f"y{i}" for i in range(1,2)]))
    
    # Return the list of created encoder instances
    return encoders


def get_scalers_statistic_attributes(par_scaler) -> List[str]:
    '''
    Function to get the statistic attributes of a scaler instance.
    
    This function checks the type of the provided scaler instance and returns a list of its statistic attributes.

    Args:
        par_scaler (Scaler): The scaler instance.

    Returns:
        List[str]: The list of the scaler's statistic attributes.
    '''
    
    # Check the type of the scaler and return the corresponding statistic attributes
    if isinstance(par_scaler, ParStandardScaler):
        return  ["mean_", "var_", "scale_"]
    
    if isinstance(par_scaler, ParMinMaxScaler):
        return ["min_", "scale_", "data_min_", "data_max_", "data_range_"]
    
    if isinstance(par_scaler, ParMaxAbsScaler):
        return ["scale_", "max_abs_"]
    
    # If the scaler type is not one of the above, it is a robust scaler, return its statistic attributes
    return ["center_", "scale_", "data_min_", "data_max_", "data_range_"]  


def check_label_encoder(par_encoders: ParLabelEncoder, 
                        seq_encoders: ParLabelEncoder) -> None:
    '''
    Function to check (for every provided categorical feature) the equality of two label encoders.
    
    This function iterates over the attributes of the provided encoders and checks if they are equal. 
    If the attribute is "classes_" or if both attribute values are numpy arrays, it checks if all elements are equal.

    Args:
        par_encoders (ParLabelEncoder): The first encoder to compare.
        seq_encoders (ParLabelEncoder): The second encoder to compare.

    Returns:
        None
    '''
    
    # Iterate over the fitted label encoders
    for encoder in zip(par_encoders.label_encoders.values(), seq_encoders.label_encoders.values()): 
        
        par_encoder, seq_encoder = encoder
        
        # Iterate over the attributes of the encoders   
        for par_attr, seq_attr in zip(vars(par_encoder), vars(seq_encoder)):
                
            par_attr_val = getattr(par_encoder, par_attr)
            seq_attr_val = getattr(seq_encoder, seq_attr)
        
            # If the attribute is "classes_" or if both attribute values are numpy arrays, check if all elements are equal
            if par_attr == "classes_" or (isinstance(par_attr_val, np.ndarray) and isinstance(seq_attr_val, np.ndarray)):       
                assert (par_attr_val == seq_attr_val).all()
        
            else:
                assert  par_attr_val == seq_attr_val
                
                
@pytest.fixture(scope = "module", params = generate_scalers_params(f"./data/sample_data.csv"))
def preprocessor_scalers(request: pytest.FixtureRequest, generate_BlockReader: BlockReader) -> Tuple[ParPreprocessor, Any]:
    '''
    Pytest fixture to create a pair of preprocessors.
    
    This fixture creates a ParPreprocessor Pipeline wtih one scaler and a copy of the provided scaler. 
    The ParPreprocessor is initialized with a block reader and a dictionary of preprocessors. 
    The fixture returns a tuple of the created preprocessors.

    Args:
        request (pytest.FixtureRequest): An object that has a param attribute which is set to the value 
                                         returned by the params option of the pytest.fixture decorator.
        generate_BlockReader (BlockReader): A function to generate a BlockReader instance.

    Returns:
        Tuple[ParPreprocessor, Any]: The created preprocessors.
    '''
    
    par_preprocessor = ParPreprocessor(block_reader = generate_BlockReader, preprocessors = {"InputVar": [request.param]})
    
    # Create a copy of the provided scaler
    seq_preprocessor = deepcopy(request.param)
    
    return par_preprocessor, seq_preprocessor


@pytest.fixture(scope = "module", params = generate_encoders_params(f"./data/sample_data.csv"))
def preprocessor_encoders(request: pytest.FixtureRequest, generate_BlockReader: BlockReader) -> Tuple[ParPreprocessor, Any]:
    '''
    Pytest fixture to create a pair of preprocessors for encoding.
    
    This fixture creates a ParPreprocessor Pipeline wtih one encoder and a copy of the provided encoder. 
    The ParPreprocessor is initialized with a block reader and a dictionary of preprocessors. 
    The fixture returns a tuple of the created preprocessors.

    Args:
        request (pytest.FixtureRequest): An object that has a param attribute which is set to the value 
                                         returned by the params option of the pytest.fixture decorator.
        generate_BlockReader (BlockReader): A function to generate a BlockReader instance.

    Returns:
        Tuple[ParPreprocessor, Any]: The created preprocessors.
    '''
    
    par_preprocessor = ParPreprocessor(block_reader = generate_BlockReader, preprocessors = {"TargetVar": [request.param]})
    
    # Create a copy of the provided encoder
    preprocessor = deepcopy(request.param)
    
    return par_preprocessor, preprocessor


def test_scalers_results(preprocessor_scalers: Tuple[ParPreprocessor, Any], generate_BlockReader: BlockReader) -> None:
    '''
    Test function to compare the results of two scalers.
    
    This function fits the preprocessors, retrieves the scalers and compares their attributes. 
    
    Args:
        preprocessor_scalers (Tuple[ParPreprocessor, Any]): A tuple of preprocessors to compare.
        generate_BlockReader (BlockReader): The block reader to use.

    Returns:
        None
    '''
    
    # Unpack the preprocessors
    par_preprocessor, seq_scaler = preprocessor_scalers
    
    # Fit the preprocessors
    par_preprocessor.parallel_fit()
    seq_scaler._fit(use_parallel = False, block_reader = generate_BlockReader)
    
    # Retrieve the fitted scaler from the ParPreprocessor pipeline
    par_scaler = par_preprocessor.preprocessors["InputVar"].pop(0)
    
    # Get the statistical attributes of the scaler
    stats_attributes = get_scalers_statistic_attributes(par_scaler)
    
    eps = 1e-10
    
    # Compare the attributes of the two scalers
    for par_attr, seq_attr in zip(vars(par_scaler), vars(seq_scaler)):
       
        par_attr_val = getattr(par_scaler, par_attr)
        seq_attr_val = getattr(seq_scaler, seq_attr)
        
        # If the attribute is a statistical attribute and is not None, check if the absolute difference is less than or equal to epsilon
        if par_attr in stats_attributes and par_attr_val is not None:
            assert (np.abs(par_attr_val - seq_attr_val) <= eps).all()
        
        # If the attribute values are numpy arrays, check if all elements are equal
        elif isinstance(par_attr_val, np.ndarray) and isinstance(seq_attr_val, np.ndarray):           
            assert (par_attr_val == seq_attr_val).all()
        
        else:
            assert  par_attr_val == seq_attr_val
    
    
def test_encoder_results(preprocessor_encoders: Tuple[ParPreprocessor, Any], generate_BlockReader: BlockReader) -> None:
    '''
    Test function to compare the results of two encoders.
    
    This function fits the preprocessors, retrieves the encoders and compares their attributes. 

    Args:
        preprocessor_encoders (Tuple[ParPreprocessor, Any]): A tuple of preprocessors to compare.
        generate_BlockReader (BlockReader): The block reader to use.

    Returns:
        None
    '''
    
    # Unpack the preprocessors
    par_preprocessor, seq_encoder = preprocessor_encoders
    
    # Fit the preprocessors
    par_preprocessor.parallel_fit()
    seq_encoder._fit(use_parallel = False, block_reader = generate_BlockReader)
    
    # Retrieve the fitted encoder from the ParPreprocessor pipeline
    par_encoder = par_preprocessor.preprocessors["TargetVar"].pop(0)
    
    if isinstance(par_encoder, ParLabelEncoder):
        check_label_encoder(par_encoder, seq_encoder)
    
    else:
        for par_attr, seq_attr in zip(vars(par_encoder), vars(seq_encoder)):
        
            par_attr_val = getattr(par_encoder, par_attr)
            seq_attr_val = getattr(seq_encoder, seq_attr)

            # If the attribute is "categories_", check if all elements are equal
            if par_attr == "categories_":   
                print(par_attr_val)        
                print(seq_attr_val)
                for index in range(len(par_attr_val)):
                    assert (par_attr_val[index] == seq_attr_val[index]).all()
            
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