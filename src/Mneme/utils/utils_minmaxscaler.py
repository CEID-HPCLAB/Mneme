import numpy as np
import copy
from typing import List
from sklearn.preprocessing import MinMaxScaler


'''
This file contains utility functions for handling MinMaxScaler objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted MinMaxScalers into a single MinMaxScaler. 
These functions are designed to streamline the process of working with MinMaxScaler instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def _copy_attr(target_obj, source_obj: MinMaxScaler) -> None:
    '''
    Copies attributes from a source MinMaxScaler object to a target object.

    This function iterates over all attributes of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (MinMaxScaler): The MinMaxScaler object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))


def reduce_minmax_scalers(scalers: List[MinMaxScaler]) -> MinMaxScaler:
    '''
    Reduces a list of partial fitted MinMaxScalers into a single final fitted 
    MinMaxScaler.

    Args:
        scalers (List[MinMaxScaler]): A list of MinMaxScalers to be reduced.

    Returns:
        MinMaxScaler: The final fitted standard scaler.
    '''
    
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    
    # Loop through the remaining scalers in the list
    for sc in scalers:
        # Merge the scaler 'sc' with all the previous scalers in the list
        scaler = _merge_scalers(scaler, sc)
    
    # Update the statistics of the final fitted scaler
    min, max = scaler.feature_range
    scaler.scale_ = (max - min) / (scaler.data_max_ - scaler.data_min_)
    scaler.min_ = min - scaler.data_min_ * scaler.scale_
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    
    # Return the final minmax scaler after merging all partial fitted scalers
    return scaler


def _merge_scalers(scaler1: MinMaxScaler, scaler2: MinMaxScaler) -> MinMaxScaler:
    ''' 
    Merges two partial fitted MinMaxScalers into a single MinMaxScaler 
    by updating the statistics of the first scaler.

    Args:
        scaler1 (MinMaxScaler): The first MinMaxScaler to be merged. This scaler will be updated with the merged statistics.
        scaler2 (MinMaxScaler): The second MinMaxScaler to be merged.

    Returns:
        MinMaxScaler: The updated MinMaxScaler with the merged statistics.
    '''
    
    # Extract the statistics from the first scaler
    last_min = scaler1.data_min_
    last_max = scaler1.data_max_
    last_sample_count = scaler1.n_samples_seen_
    
    # Extract the statistics from the second scaler
    new_min = scaler2.data_min_
    new_max = scaler2.data_max_
    new_sample_count = scaler2.n_samples_seen_

    # Compute the updated statistics
    updated_min, updated_max, updated_sample_count = _incremental_min_and_max(new_min, new_max, new_sample_count, 
                                                                              last_min, last_max, last_sample_count)
    
    # Update the statistics of the first scaler
    scaler1.data_min_ = updated_min
    scaler1.data_max_ = updated_max
    scaler1.n_samples_seen_ = updated_sample_count
    
    return scaler1

        
def _incremental_min_and_max(new_min: np.ndarray, new_max: np.ndarray, new_sample_count: int, last_min: np.ndarray, 
                             last_max: np.ndarray, last_sample_count: int) -> tuple[np.ndarray, np.ndarray, int]:    
    '''
    Incrementally update (for every feature) the minimum and maximum value and the sample count by combining the statistics 
    of two partial fitted MinMaxScalers.

    This function is used to incrementally update the minimum and maximum value of every feature of a dataset by combining 
    every time the statistics of two partial fitted MinMaxScalers. The updated minimum value is the 
    minimum of the new and last updated minimum values, the updated maximum value is the 
    maximum of the new and last updated maximum values and the updated sample count is the sum of the new and 
    last updated sample counts.

    Args:
        new_min (numpy.ndarray shape (n_features,)): The minimum value of the new data.
        (new data -> chunk data of the one MinMaxScaler).
        
        new_max (numpy.ndarray shape (n_features,)): The maximum value of the new data.
        
        new_sample_count (int): The number of samples in the new data.
        
        last_min (numpy.ndarray shape (n_features,)): The minimum value of the data up to the last update.
        (last update -> the previous already reduced partial fitted MinMaxScalers)
            
        last_max (numpy.ndarray shape (n_features,)): The maximum value of the data up to the last update.
    
        last_sample_count (int): The number of samples in the data up to the last update.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: A tuple containing the updated maximum absolute values and the updated sample count.
    
    '''
    
    # Compute (for every feature) the updated minimum value
    updated_min = np.minimum(new_min, last_min)
    
    # Compute (for every feature) the updated maximum value
    updated_max = np.maximum(new_max, last_max)
    
    # Compute the updated sample count
    updated_sample_count = last_sample_count + new_sample_count
    
    return updated_min, updated_max, updated_sample_count