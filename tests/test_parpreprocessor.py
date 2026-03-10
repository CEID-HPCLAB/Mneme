from Mneme.preprocessing import (ParStandardScaler,
                                  ParMinMaxScaler,
                                  ParRobustScaler,
                                  ParMaxAbsScaler,
                                  ParMinMaxScaler,
                                  ParallelPipeline,
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
                
                
@pytest.fixture(scope="module", params=[ParStandardScaler, ParMinMaxScaler, ParMaxAbsScaler, ParRobustScaler])
def preprocessor_scalers(request, datafile_path) -> Tuple[ParallelPipeline, Any]:

    scaler = request.param(data_file=datafile_path, num_idxs=[f"x{i}" for i in range(3)])

    par_preprocessor = ParallelPipeline({"InputFeatures": [scaler]}, datafile_path)

    seq_preprocessor = deepcopy(scaler)

    return par_preprocessor, seq_preprocessor


@pytest.fixture(scope="module", params=[ParOneHotEncoder, ParLabelEncoder, ParOrdinalEncoder])
def preprocessor_encoders(request: pytest.FixtureRequest, datafile_path) -> Tuple[ParallelPipeline, Any]:
    
    encoder = request.param(data_file=datafile_path, cat_idxs=[f"y{i}" for i in range(1,2)])

    par_preprocessor = ParallelPipeline({"TargetVar": [encoder]}, datafile_path)

    seq_encoder = deepcopy(encoder)

    return par_preprocessor, seq_encoder


def test_scalers_results(preprocessor_scalers: Tuple[ParallelPipeline, Any], generate_BlockReader: BlockReader) -> None:
    '''
    Test function to compare the results of two scalers.
    
    This function fits the preprocessors, retrieves the scalers and compares their attributes. 
    '''
    
    # Unpack the preprocessors
    par_preprocessor, seq_scaler = preprocessor_scalers
    
    # Fit the preprocessors
    par_preprocessor.fit(block_reader = generate_BlockReader, num_workers = 4, IO_workers = 2)
    seq_scaler.fit(block_reader = generate_BlockReader, num_workers = 1, IO_workers = 1)
    
    # Retrieve the fitted scaler from the ParallelPipeline 
    par_scaler = par_preprocessor.operators["InputFeatures"].pop(0)
    
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
    
    
def test_encoder_results(preprocessor_encoders: Tuple[ParallelPipeline, Any], generate_BlockReader: BlockReader) -> None:
    '''
    Test function to compare the results of two encoders.
    
    This function fits the preprocessors, retrieves the encoders and compares their attributes. 
    '''
    
    # Unpack the preprocessors
    par_preprocessor, seq_encoder = preprocessor_encoders
    
    # Fit the preprocessors
    par_preprocessor.fit(block_reader = generate_BlockReader, num_workers = 4, IO_workers = 2)
    seq_encoder.fit(block_reader = generate_BlockReader, num_workers = 1, IO_workers = 1)
    
    # Retrieve the fitted encoder from the ParallelPipeline
    par_encoder = par_preprocessor.operators["TargetVar"].pop(0)
    
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