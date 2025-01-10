import numpy as np
import copy
from typing import List
from sklearn.preprocessing import StandardScaler


'''
This file contains utility functions for handling StandardScaler objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted StandardScalers into a single StandardScaler. 
These functions are designed to streamline the process of working with StandardScaler instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def _copy_attr(target_obj, source_obj: StandardScaler) -> None:
    '''
    Copies attributes from a source StandardScaler object to a target object.

    This function iterates over all attributes of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (StandardScaler): The StandardScaler object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))
        
        
# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def reduce_std_scalers(scalers: List[StandardScaler]) -> StandardScaler:
    '''
    Reduces a list of partial fitted StandardScalers into a single final fitted 
    StandardScaler.

    Args:
        scalers (List[StandardScaler]): A list of StandardScalers to be reduced.

    Returns:
        StandardScaler: The final fitted standard scaler.
    '''
    
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    
    # Loop through the remaining scalers in the list
    for sc in scalers:
        # Merge the scaler 'sc' with all the previous scalers in the list
        scaler = _merge_scalers(scaler, sc)
    
    # The reason we set the mean_, var_ and scale_ attributes to None after the final reduction is due to 
    # the way these attributes are used in the computation of the StandardScaler.
    # The issue arises specifically when with_mean is set to False and with_std is set to True. 
    # In this case, if we were to set mean_ to None before the final reduction, it would not be available for 
    # the computation of var_ and scale_, leading to incorrect results. This is because the computation of 
    # the variance (var_) and scale (scale_) relies on the mean (mean_). Therefore, even if with_mean is False, 
    # we still need to compute the mean and use it in these computations. 
    
    if scaler.with_mean == False:
        scaler.mean_ = None
    
    if scaler.with_std == False:
        scaler.var_ = None
        scaler.scale_ = None

    # Return the final standard scaler after merging all partial fitted scalers
    return scaler


def _merge_scalers(scaler1: StandardScaler, scaler2: StandardScaler) -> StandardScaler:
    '''
    Merges two partial fitted StandardScalers into a single StandardScaler 
    by updating the statistics of the first scaler.

    Args:
        scaler1 (StandardScaler): The first StandardScaler to be merged. This scaler will be updated with the merged statistics.
        scaler2 (StandardScaler): The second StandardScaler to be merged.

    Returns:
        StandardScaler: The updated StandardScaler with the merged statistics.
    '''
    
    # Extract the statistics from the first scaler
    last_mean = scaler1.mean_
    last_variance = scaler1.var_
    last_sample_count = scaler1.n_samples_seen_

    # Extract the statistics from the second scaler
    new_mean = scaler2.mean_
    new_variance = scaler2.var_
    new_sample_count = scaler2.n_samples_seen_

    # Compute the updated statistics
    updated_mean, updated_variance, updated_sample_count = _incremental_mean_and_var(new_mean, new_variance, 
                                                                                     new_sample_count, last_mean, 
                                                                                     last_variance, last_sample_count)
    
    # Compute the updated scale
    updated_scale = np.sqrt(updated_variance)

    # Update the statistics of the first scaler
    scaler1.mean_ = updated_mean
    scaler1.var_ = updated_variance
    scaler1.n_samples_seen_ = updated_sample_count
    scaler1.scale_ = updated_scale
    
    return scaler1


def _incremental_mean_and_var(new_mean: np.ndarray, new_variance: np.ndarray, new_sample_count: int, 
                              last_mean: np.ndarray, last_variance: np.ndarray, last_sample_count: int) -> tuple[np.ndarray, np.ndarray, int]:
    '''
    Calculate the mean update, the Youngs and Cramer variance and the sample count by combining the statistics 
    of two partial fitted StandardScalers.
    
    This function is used to incrementally update the mean and variance of every feature of a dataset
    by combining every time the statistics of two partial fitted StandardScalers using an algorithm described 
    in the paper "Algorithms for computing the sample variance: analysis and recommendations" by Chan, Golub and LeVeque.

    Args:
        new_mean (numpy.ndarray shape (n_features,)): The mean of the new data. 
        (new data -> chunk data of the one StandardScaler).

        new_variance (numpy.ndarray shape (n_features,)): The variance of the new data. 
       
        new_sample_count (int): The number of samples in the new data.
        
        last_mean(numpy.ndarray shape (n_features,)): The mean of the data up to the last update.
        (last update -> the previous already reduced partial fitted StandardScalers)

        last_variance(numpy.ndarray shape (n_features,)): The variance of the data up to the last update.
    
        last_sample_count (int): The number of samples  of the data up to the last update.

    Returns:
        tuple: A tuple containing the updated mean values, the updated variance values and the updated sample count.

    Notes:
        NaNs are ignored during the algorithm.

    References:
        T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    '''
    
    # Compute the total sum of the old and new data
    last_sum = last_mean * last_sample_count
    # new_sum = _safe_accumulator_op(np.nansum, X, axis=0)
    new_sum = new_mean * new_sample_count

    # new_sample_count = np.sum(~np.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        # new_unnormalized_variance = (
        #    _safe_accumulator_op(np.nanvar, X, axis=0) * new_sample_count)
        
        # Compute the unnormalized variance for the new samples
        new_unnormalized_variance = new_variance * new_sample_count

        # Compute the unnormalized variance for the last updated samples
        last_unnormalized_variance = last_variance * last_sample_count

        # Compute the updated unnormalized variance
        with np.errstate(divide='ignore', invalid='ignore'):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance + new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        
        # Handle cases where the last sample count is zero
        zeros = last_sample_count == 0
        
        # Use boolean indexing to select elements from updated_unnormalized_variance
        # where the corresponding element in zeros is True and update them with the
        # corresponding elements from new_unnormalized_variance. 
        # If zeros is True, it selects all elements, if zeros is False, it selects none.
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        
        # Compute the updated variance
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count