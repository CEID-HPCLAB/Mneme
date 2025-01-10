import numpy as np
import copy
from typing import List
from sklearn.preprocessing import RobustScaler


'''
This file contains utility functions for handling RobustScaler objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted RobustScalers into a single RobustScaler. 
These functions are designed to streamline the process of working with RobustScaler instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def _copy_attr(target_obj, source_obj: RobustScaler) -> None:
    '''
    Copies attributes from a source RobustScaler object to a target object.

    This function iterates over all attributes of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (RobustScaler): The RobustScaler object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))


def reduce_robust_scalers(scalers: List[RobustScaler]) -> RobustScaler:
    '''
    Reduces a list of partial fitted RobustScalers into a single final fitted 
    RobustScaler.
    
    Note: This function does not work exactly like the  RobustScaler provided by sklearn.preprocessing. 
    Because it was not possible (as yet) to parallelize the median calculation algorithm, the median 
    element is defined as the median of the individual median elements of the data chunks (blocks).
    An alternative implementation considers the median to be the mean value of the individual 
    medians of the data chunks (blocks). Also, the interquartile range is defined as the mean of the 
    individual interquartile ranges of the data chunks (blocks), not the interquartile range of the entire dataset. 
    Improving these specific algorithms is a future goal.

    Args:
        scalers (List[RobustScaler]): A list of RobustScalers to be reduced.

    Returns:
        RobustScaler: The final fitted robust scaler.
    '''
    
    head = scalers.pop(0)
    scaler = copy.deepcopy(head)
    
    if scaler.with_centering and scaler.with_scaling:
        # If both are True, then both centering and scaling operations will be performed on the data
    
        # Initialize 'centers' list with median values (median value per feature) from first partial
        # fitted robust scaler statistics
        centers = [[median] for median in scaler.center_]
        
        # Iterate over the remaining scalers in the list
        for sc in scalers:
            # Accumulate interquartile ranges from the current scaler 'sc' and update 'scaler'
            scaler = _accumulate_interquartile_ranges(scaler, sc)
            
            # For every feature, gather the median value from the statistics of the current robust scaler 
            # and store it in 'centers' list with the previous robust scalers median values for this feature
            centers = _gather_medians(sc, centers)
            
        
        # Compute the median of each feature's medians    
        medians = [np.median(feature_med) for feature_med in centers]
        
        # Note: An alternative implementation could consider the median as the mean value of the individual 
        # medians of the data chunks (blocks)
        # medians = [np.sum(feature_med)/(len(scalers) + 1) for feature_med in centers]
        
        # Update the center attribute of the final fitted robust scaler with the computed medians
        scaler.center_ = np.array(medians)
        
        # Update the scale attribute of the final fitted robust scaler with the mean of the 
        # individual interquartile ranges of the data chunks (blocks)
        scaler.scale_ = scaler.scale_ / (len(scalers) + 1)
        
        # Return the final robust scaler after merging all partial fitted scalers
        return scaler

    elif scaler.with_centering:
        # If 'with_centering' is True and 'with_scaling' is False, only centering operation will be performed on the data
        
        # Initialize 'centers' list with median values (median value per feature) from first partial
        # fitted robust scaler statistics
        centers = [[median] for median in scaler.center_]
        
        # Iterate over the remaining scalers in the list
        for sc in scalers:
            # For every feature, gather the median value from the statistics of the current robust scaler 
            # and store it in 'centers' list with the previous robust scalers median values for this feature
            centers = _gather_medians(sc, centers)
            
        
        # Compute the median of each feature's medians    
        medians = [np.median(feature_med) for feature_med in centers]
        
        # Note: An alternative implementation could consider the median as the mean value of the individual 
        # medians of the data chunks (blocks)
        # medians = [np.sum(feature_med)/(len(scalers) + 1) for feature_med in centers]
        
        # Update the center attribute of the final fitted robust scaler with the computed medians
        scaler.center_ = medians

        # Return the final robust scaler after merging all partial fitted scalers
        return scaler
        
    elif scaler.with_scaling:
        # If 'with_scaling' is True and 'with_centering' is False, only scaling operation will be performed on the data
        
        # Iterate over the remaining scalers in the list
        for sc in scalers:
            # Accumulate interquartile ranges from the current scaler 'sc' and update 'scaler'
            scaler = _accumulate_interquartile_ranges(scaler, sc)
        
        # Update the scale attribute of the final fitted robust scaler with the mean of the 
        # individual interquartile ranges of the data chunks (blocks)
        scaler.scale_ = scaler.scale_ / (len(scalers) + 1)
        
        # Return the final robust scaler after merging all partial fitted scalers
        return scaler
    
    # If neither 'with_centering' nor 'with_scaling' is True, no operation 
    # will be performed and the scaler will be returned as is. In this case, 
    # there is no need for reduction.
    return scaler
        
        
        
def _gather_medians(scaler: RobustScaler, centers: List[List[list]]) -> List[List[list]]:
    '''
    Gather the median values from the statistics of a RobustScaler and append them to the existing list 
    of median values from all the previous 'merged' scalers.
    
    This function gathers the median values from the statistics of a RobustScaler and appends them 
    to the existing list of medians. It iterates over each element in the 'center_' attribute of the RobustScaler, 
    which contains the median values for each feature. For each feature, it appends the 
    corresponding median values to the sublist in 'centers' representing that feature.
    
    Args:
        scaler (RobustScaler): The RobustScaler instance from which to gather the median values.
        centers (List[list]): The list containing median values for each feature. 
        Each sublist represents the median values for a single feature.

    Returns:
        List[list]: The updated list of median values after appending the median values 
        from the statistics of the RobustScaler.
    '''
    
    for index, median in enumerate(scaler.center_):
        # Append the median value for the current feature 
        # to the corresponding sublist in 'centers'
        centers[index].append(median)
        
    return centers


def _accumulate_interquartile_ranges(scaler1: RobustScaler, scaler2: RobustScaler) -> RobustScaler:
    '''
    Accumulates the interquartile ranges of two RobustScalers.

    This function takes two RobustScalers as input and adds the scale (interquartile range) of the second scaler to the first one.
    The updated first scaler is then returned.

    Args:
        scaler1 (RobustScaler): The first RobustScaler whose scale will be updated.
        scaler2 (RobustScaler): The second RobustScaler whose scale will be added to the first scaler.

    Returns:
        RobustScaler: The updated first RobustScaler with the accumulated scale.
    '''
    
     # Add the scale of the second scaler to the first scaler
    scaler1.scale_ += scaler2.scale_
    
    return scaler1