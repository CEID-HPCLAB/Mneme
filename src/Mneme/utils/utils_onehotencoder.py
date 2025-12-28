import numpy as np
import copy
from typing import List
from sklearn.preprocessing import OneHotEncoder


'''
This file contains utility functions for handling OneHotEncoder objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted OneHotEncoders into a single OneHotEncoder. 
These functions are designed to streamline the process of working with OneHotEncoder instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def _copy_attr(target_obj, source_obj: OneHotEncoder) -> None:
    '''
    Copies attributes from a source OneHotEncoder object to a target object.

    This function iterates over all attributes of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (OneHotEncoder): The OneHotEncoder object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))
        

def reduce_onehot_encoders(onehotencoders: List[OneHotEncoder]) -> OneHotEncoder:
    '''
    Reduces a list of partial fitted OneHotEncoders into a single final fitted 
    OneHotEncoder.

    Args:
        encoders (List[OneHotEncoder]): A list of OneHotEncoders to be reduced.

    Returns:
        OneHotEncoder: The final fitted one-hot encoder.
    '''
    
    head = onehotencoders.pop(0)
    onehotencoder = copy.deepcopy(head)
    
    # Loop through the remaining encoders in the list
    for enc in onehotencoders:
        # Merge the encoder 'enc' with all the previous encoders in the list
        onehotencoder = _merge_onehot_encoders(onehotencoder, enc)
    
    onehotencoder._n_features_outs = [len(cats) for cats in onehotencoder.categories_]
    
    # Compute the indices of the features to be dropped based on the 'drop' parameter of the OneHotEncoder
    # This is necessary because during the merging of the encoders, new categories might have been added
    compute_drop_idxs(onehotencoder)
    
    
    # Return the final one-hot encoder after merging all partial fitted encoders
    return onehotencoder

  
def _merge_onehot_encoders(onehotencoder1: OneHotEncoder, onehotencoder2: OneHotEncoder) -> OneHotEncoder:
    '''
    Merges two partial fitted OneHotEncoders into a single OneHotEncoder 
    by updating the categories of the first encoder.

    Args:
        onehotencoder1 (OneHotEncoder): The first OneHotEncoder to be merged. 
        This encoder will be updated with the unified categories.
        onehotencoder2 (OneHotEncoder): The second OneHotEncoder to be merged.

    Returns:
        OneHotEncoder: The updated OneHotEncoder with the unified categories.
    '''
    
    merged_categories = []
    
    # Extract the categories from the first encoder
    cat1 = onehotencoder1.categories_
    
    # Extract the categories from the second encoder
    cat2 = onehotencoder2.categories_
    
    # Compute the updated categories of each feature by unifying the categories from both encoders
    for last_categories, new_categories in zip(cat1, cat2):
        updated_categories = np.union1d(new_categories, last_categories)
        merged_categories.append(updated_categories)
    
    # Update the categories of the first encoder
    onehotencoder1.categories_ = merged_categories
    
    return onehotencoder1


def compute_drop_idxs(onehotencoder: OneHotEncoder) -> None:
    '''
    Computes the indices of the features to be dropped based on the 'drop' parameter of the OneHotEncoder.

    Args:
        onehotencoder (OneHotEncoder): The OneHotEncoder for which the drop indices are to be computed.

    Returns:
        None: The function modifies the 'drop_idx_' attribute of the OneHotEncoder in-place.
    '''
    
    # If 'drop' is None, no feature is to be dropped
    if onehotencoder.drop is None:
        pass
    
    # If 'drop' is 'first', the first category of each feature is to be dropped    
    elif onehotencoder.drop == 'first':
        onehotencoder.drop_idx_ = np.zeros(len(onehotencoder.categories_), dtype = object)
        onehotencoder._drop_idx_after_grouping = onehotencoder.drop_idx_
        
        # Update the number of output features after dropping the first category
        onehotencoder._n_features_outs = [num_cats - 1 for num_cats in onehotencoder._n_features_outs]
        
    # If 'drop' is 'if_binary', the first category is dropped if the corresponding
    # feature has exactly two categories
    elif onehotencoder.drop == 'if_binary':
        onehotencoder.drop_idx_ = np.array([0 if len(feature_categories) == 2 else None for feature_categories 
                                            in onehotencoder.categories_], 
                                           dtype = object)
        onehotencoder._drop_idx_after_grouping = onehotencoder.drop_idx_
        
        # Find the indices of the features where a category is to be dropped
        not_none_idxs = np.where(onehotencoder.drop_idx_ != None)[0]
        
        # Update the number of output features for the features where a category is to be dropped
        for idx in not_none_idxs:
            onehotencoder._n_features_outs[idx] -= 1
    
    # If 'drop' is neither None, 'first' nor 'if_binary', it is assumed to be an array-like of shape (n_features,)
    # specifying the categories to be dropped for each feature
    else:
        def get_cat_idx(val, cats):
            try:
                return np.where(cats == val)[0][0]
            except IndexError:
                raise ValueError(f"{val} not found in {cats}")

        onehotencoder.drop_idx_ = np.array([None if d is None else get_cat_idx(d, cats)
                                            for d, cats in zip(onehotencoder.drop, onehotencoder.categories_)], dtype = object)
        onehotencoder._drop_idx_after_grouping = onehotencoder.drop_idx_
        
        # Find the indices of the features where a category is to be dropped
        not_none_idxs = np.where(onehotencoder.drop_idx_ != None)[0]
        
        # Update the number of output features for the features where a category is to be dropped
        for idx in not_none_idxs:
            onehotencoder._n_features_outs[idx] -= 1