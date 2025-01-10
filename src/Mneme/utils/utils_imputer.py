import numpy as np
from copy import deepcopy
from typing import List
from sklearn.impute import SimpleImputer


'''
This file provides utility functions for handling imputers in a parallel computing context. 

It includes functions to reduce multiple imputers into a single one by merging their statistics, 
depending on the imputation strategy (mean, median, most frequent, constant).
'''


def reduce_imputers(imputers: List[SimpleImputer], strategy: str) -> SimpleImputer:
    '''
    Reduces a list of partial fitted imputers into a single imputer by merging their statistics, 
    depending on the specified imputation strategy.

    Args:
        imputers (list): A list of imputers to be reduced.
        strategy (str): The imputation strategy to be used. It can be "mean", "median", "most_frequent", or "constant".

    Returns:
        reduced_imputer (SimpleImputer): The final fitted imputer.
    '''
    
    # Mapping of imputation strategies to corresponding reduction functions
    merge_method = {
        "mean": _reduce_mean_imputers,
        "median": _reduce_median_imputers,
        "most_frequent": _reduce_most_frequent_imputers,
        "constant": _reduce_constant_imputers
    }
    
    # Call the appropriate reduction function based on the strategy    
    reduced_imputer = merge_method[strategy](imputers)

    return reduced_imputer


def _reduce_mean_imputers(imputers: List[SimpleImputer]) -> SimpleImputer:
    '''
    Reduces a list of partial fitted SimpleImputers (with strategy 'mean') into a single final fitted 
    SimpleImputer.

    Args:
        imputers (List[SimpleImputer]): A list of SimpleImputers (with strategy 'mean') to be reduced.

    Returns:
        SimpleImputer: The final fitted imputer.
    '''
    
    head = imputers.pop(0)
    imputer = deepcopy(head)
    
    # Loop through the remaining imputers in the list
    for imp in imputers:
        # Merge the imputer 'imp' with all the previous imputers in the list
        imputer = _merge_mean_imputers(imputer, imp)

    # Return the final imputer after merging all imputers
    return imputer


def _merge_mean_imputers(imputer1: SimpleImputer, imputer2: SimpleImputer) -> SimpleImputer:
    '''
    Merges two SimpleImputers (with strategy 'mean') by updating the statistics of the first one based on the second one.

    This function merges two SimpleImputer instances by updating the mean statistics and sample counts
    using an incremental mean calculation.

    The 'statistics_' attribute of imputer1 is updated with the combined mean of imputer1 and imputer2.
    The 'n_samples_seen_' attribute of imputer1 is updated with the combined sample counts of imputer1 and imputer2.
    The 'nan_vals' attribute of imputer1 is updated with the combined count of missing values from imputer1 and imputer2.
    
    Args:
        imputer1 (SimpleImputer): The first SimpleImputer to be merged.
        imputer2 (SimpleImputer): The second SimpleImputer to be merged.

    Returns:
        imputer1 (SimpleImputer): The merged SimpleImputer instance.
    '''
    
    # Extract statistics and sample counts from the first imputer
    last_mean = imputer1.statistics_
    last_sample_count = imputer1.n_samples_seen_
    last_nan_count = imputer1.nan_vals
    
    # Extract statistics and sample counts from the second imputer
    new_mean = imputer2.statistics_
    new_sample_count = imputer2.n_samples_seen_
    new_nan_count = imputer2.nan_vals
    
    # Calculate the updated mean, sample count and nan count using incremental mean and 
    # variance algorithm (T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
    # variance: recommendations, The American Statistician, Vol. 37, No. 3, pp. 242-247)
    updated_mean, updated_sample_count, updated_nan_count = _incremental_mean(new_mean, last_mean, new_sample_count, 
                                                                              last_sample_count, new_nan_count, last_nan_count)

    # Update the attributes of imputer1 with the updated values
    imputer1.statistics_ = np.array(updated_mean)
    imputer1.n_samples_seen_ = updated_sample_count
    imputer1.nan_vals = np.array(updated_nan_count)
   
    return imputer1


def _incremental_mean(new_mean: float, last_mean: float, new_sample_count: int, last_sample_count: int, 
                      new_nan_count: int, last_nan_count: int) -> tuple[float, int, int]:
    '''
    Calculate the incremental mean, sample count and NaN count by combining the statistics 
    of two partial fitted SimpleImputers.
    
    This function calculates the incremental mean, sample count and NaN count by combining the statistics
    of two partial fitted SimpleImputers using an algorithm described in the paper "Algorithms for computing 
    the sample variance: analysis and recommendations" by Chan, Golub and LeVeque.
    
    Args:
        new_mean (float): The mean value for the samples seen by SimpleImputer 2.
        last_mean (float): The mean value for the samples seen by SimpleImputer 1.
        new_sample_count (int): The number of samples seen by SimpleImputer 2.
        last_sample_count (int): The number of samples seen by SimpleImputer 1.
        new_nan_count (int): The number of NaN values in the samples seen by SimpleImputer 2.
        last_nan_count (int): The number of NaN values in the samples seen by SimpleImputer 1.

    Returns:
        tuple: The updated mean, sample count and NaN count.
    
    References:    
        T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    '''
    
    updated_sample_count = last_sample_count + new_sample_count
    last_sum = last_mean * (last_sample_count - last_nan_count)
    new_sum = new_mean * (new_sample_count - new_nan_count)
    updated_nan_count = last_nan_count + new_nan_count
    
    # Calculate the updated mean using the formula proposed by Chan, Golub and LeVeque
    updated_mean = (last_sum + new_sum) / (updated_sample_count - updated_nan_count)
    
    return updated_mean, updated_sample_count, updated_nan_count


def _reduce_most_frequent_imputers(imputers: List[SimpleImputer]) -> SimpleImputer:
    '''
    Reduces a list of partial fitted SimpleImputers (with strategy 'most_frequent') into a single final fitted 
    SimpleImputer.
    
    Note: This function does not work exactly like the SimpleImputer with 'most_frequent' strategy provided by sklearn.impute. 
    Because it was not possible (as yet) to parallelize the most_frequent calculation algorithm, the most_frequent 
    element is defined as the most_frequent of the individual most_frequent elements of the data chunks (blocks).

    Args:
        imputers (List[SimpleImputer]): A list of SimpleImputers (with strategy 'most_frequent') to be reduced.

    Returns:
        SimpleImputer: The final fitted imputer.
    '''
    
    head = imputers.pop(0)
    imputer = deepcopy(head)
    
    # Initialize 'centers' list with most frequent values (most_frequent value per feature) from first partial
    # fitted imputer statistics
    centers = [[most_frequent] for most_frequent in imputer.statistics_]
    
    # Iterate over the remaining imputers in the list
    for imp in imputers:
        # For every feature, gather the most frequent value from the statistics of the current imputer 
        # and store it in 'centers' list with the previous imputers most frequent values for this feature
        centers = _gather_centers(imp, centers)
        # Update the sample count of final fitted imputer by adding the sample count of the current imputer
        imputer.n_samples_seen_+= imp.n_samples_seen_
    
    # Iterate over the features and calculate the most frequent value for each feature
    for idx, partial_most_frequent in enumerate(centers):
        
        # Convert the list of partial most frequent values of this feature into a NumPy array
        np_partial = np.array(partial_most_frequent)
        # Find the unique elements and their counts 
        unique_elements, counts = np.unique(np_partial, return_counts = True)
        
        # Find the index of the most frequent element and get the most frequent element
        most_frequent_index = np.argmax(counts)
        most_frequent_element = unique_elements[most_frequent_index]
        
        # Update the statistics of the final fitted imputer with the most frequent element of the current feature 
        imputer.statistics_[idx] = most_frequent_element
        
    return imputer


def _gather_centers(imputer: SimpleImputer, centers: List[list]) -> List[list]:
    '''
    Gather the center values (e.g. median, most frequent) from the statistics of a SimpleImputer and append them to the existing list 
    of center values from all the previous 'merged' imputers.
    
    This function gathers the center values from the statistics of a SimpleImputer and appends them 
    to the existing list of centers. It iterates over each element in the 'statistics_' attribute of the SimpleImputer, 
    which contains the center values for each feature. For each feature, it appends the 
    corresponding center value to the sublist in 'centers' representing that feature.
    
    Args:
        imputer (SimpleImputer): The SimpleImputer instance from which to gather the center values.
        centers (List[list]): The list containing center values for each feature. 
        Each sublist represents the center values for a single feature.

    Returns:
        List[list]: The updated list of center values after appending the center values 
        from the statistics of the SimpleImputer.
    '''
    
    for index, center_val in enumerate(imputer.statistics_):
        # Append the center value (e.g., median, most frequent) for the current feature 
        # to the corresponding sublist in 'centers'
        centers[index].append(center_val)
        
    return centers


def _reduce_constant_imputers(imputers: List[SimpleImputer]) -> SimpleImputer:
    '''
    Reduce a list of partial fitted SimpleImputers by returning the first imputer, 
    following the constant imputation logic.
    
    Since the partial fitted SimpleImputers in the list use constant imputation strategy, 
    the first SimpleImputer instance is considered the reduced imputer.

    Args:
        imputers (List[SimpleImputer]): A list of SimpleImputers (with strategy 'constant') to be reduced.

    Returns:
        SimpleImputer: The final fitted imputer.
    '''
    imputer = imputers.pop(0)
   
    return imputer


def _reduce_median_imputers(imputers: List[SimpleImputer]) -> SimpleImputer:
    '''
    Reduces a list of partial fitted SimpleImputers (with strategy 'median') into a single final fitted 
    SimpleImputer.
    
    Note: This function does not work exactly like the SimpleImputer with 'median' strategy provided by sklearn.impute. 
    Because it was not possible (as yet) to parallelize the median calculation algorithm, the median 
    element is defined as the median of the individual median elements of the data chunks (blocks).
    An alternative implementation considers the median to be the mean value of the individual 
    medians of the data chunks (blocks).

    Args:
        imputers (List[SimpleImputer]): A list of SimpleImputers (with strategy 'median') to be reduced.

    Returns:
        SimpleImputer: The final fitted imputer.
    '''
    
    head = imputers.pop(0)
    imputer = deepcopy(head)
    
    # Initialize 'centers' list with median values (median value per feature) from first partial
    # fitted imputer statistics
    centers = [[median] for median in imputer.statistics_]
    
    # Iterate over the remaining imputers in the list
    for imp in imputers:
        # For every feature, gather the median value from the statistics of the current imputer 
        # and store it in 'centers' list with the previous imputers median values for this feature
         centers = _gather_centers(imp, centers)
          # Update the sample count of final fitted imputer by adding the sample count of the current imputer
         imputer.n_samples_seen_+= imp.n_samples_seen_
    
    # Calculate the final median values for each feature by taking the median of the median values 
    # from each imputer for that feature
    medians = [np.median(feature_med) for feature_med in centers]
    
    # Note: An alternative implementation could consider the median as the mean value of the individual 
    # medians of the data chunks (blocks)
    # medians = [np.sum(feature_med)/(len(imputers) + 1) for feature_med in centers]
    
    # Set the final fitted imputer statistics to the calculated median values
    imputer.statistics_ = np.array(medians)
    
    return imputer