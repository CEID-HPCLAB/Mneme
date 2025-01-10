import numpy as np
import copy
from typing import List
from sklearn.preprocessing import MaxAbsScaler


'''
This file contains utility functions for handling MaxAbsScaler objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted MaxAbsScalers into a single MaxAbsScaler. 
These functions are designed to streamline the process of working with MaxAbsScaler instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''



def reduce_maxabs_scalers(scalers: List[MaxAbsScaler]) -> MaxAbsScaler:
    '''
    Reduces a list of partial fitted MaxAbsScalers into a single final fitted 
    MaxAbsScaler.

    Args:
        scalers (List[MaxAbsScaler]): A list of MaxAbsScalers to be reduced.

    Returns:
        MaxAbsScaler: The final fitted standard scaler.
    '''
    
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    
    # Loop through the remaining scalers in the list
    for sc in scalers:
        # Merge the scaler 'sc' with all the previous scalers in the list
        scaler = _merge_scalers(scaler, sc)
        
    scaler.scale_ = scaler.max_abs_
    
    # Return the final maxabs scaler after merging all partial fitted scalers
    return scaler


def _merge_scalers(scaler1: MaxAbsScaler, scaler2: MaxAbsScaler) -> MaxAbsScaler:
    '''
    Merges two partial fitted MaxAbsScalers into a single MaxAbsScaler 
    by updating the statistics of the first scaler.

    Args:
        scaler1 (MaxAbsScaler): The first MaxAbsScaler to be merged. This scaler will be updated with the merged statistics.
        scaler2 (MaxAbsScaler): The second MaxAbsScaler to be merged.

    Returns:
        MaxAbsScaler: The updated MaxAbsScaler with the merged statistics.
    '''
    
    # Extract the statistics from the first scaler
    last_abs_max = scaler1.max_abs_
    last_sample_count = scaler1.n_samples_seen_
    
    # Extract the statistics from the second scaler
    new_abs_max = scaler2.max_abs_
    new_sample_count = scaler2.n_samples_seen_

    # Compute the updated statistics
    updated_max_abs, updated_sample_count = _incremental_max_abs(new_abs_max, new_sample_count, last_abs_max, 
                                                                 last_sample_count)
    
    # Update the statistics of the first scaler
    scaler1.max_abs_ = updated_max_abs
    scaler1.n_samples_seen_ = updated_sample_count
    
    return scaler1

        
def _incremental_max_abs(new_abs_max: np.ndarray, new_sample_count: int, last_abs_max: np.ndarray, 
                         last_sample_count: int) -> tuple[np.ndarray, int]:    
    '''
    Incrementally update (for every feature) the maximum absolute value and the sample count by combining the statistics 
    of two partial fitted MaxAbsScalers.

    This function is used to incrementally update the maximum absolute value of every feature of a dataset by combining 
    every time the statistics of two partial fitted MaxAbsScalers. The updated maximum absolute value is the 
    maximum of the new and last updated absolute values and the updated sample count is the sum of the new and 
    last updated sample counts.

    Args:
        new_abs_max (numpy.ndarray shape (n_features,)): The maximum absolute value of the new data.
        (new data -> chunk data of the one MaxAbsScaler).
        
        new_sample_count (int): The number of samples in the new data.
            
        last_abs_max (numpy.ndarray shape (n_features,)): The maximum absolute value of the data up to the last update.
        (last update -> the previous already reduced partial fitted MaxAbsScalers)
        
        last_sample_count (int): The number of samples in the data up to the last update.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the updated maximum absolute values and the updated sample count.
    '''
    
    # Compute (for every feature) the updated maximum absolute value
    updated_abs_max = np.maximum(new_abs_max, last_abs_max)
    
    # Compute the updated sample count
    updated_sample_count = last_sample_count + new_sample_count
    
    return updated_abs_max, updated_sample_count