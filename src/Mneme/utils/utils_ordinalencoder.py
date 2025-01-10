import numpy as np
import copy
from typing import List
from sklearn.preprocessing import OrdinalEncoder


'''
This file contains utility functions for handling OrdinalEncoder objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted OrdinalEncoders into a single OrdinalEncoder. 
These functions are designed to streamline the process of working with OrdinalEncoder instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def _copy_attr(target_obj, source_obj: OrdinalEncoder) -> None:
    '''
    Copies attributes from a source OrdinalEncoder object to a target object.

    This function iterates over all attributes of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (OrdinalEncoder): The OrdinalEncoder object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))


def reduce_ordinal_encoders(ordinalencoders: List[OrdinalEncoder]) -> OrdinalEncoder:
    '''
    Reduces a list of partial fitted OrdinalEncoders into a single final fitted 
    OrdinalEncoder.

    Args:
        encoders (List[OrdinalEncoder]): A list of OrdinalEncoders to be reduced.

    Returns:
        OrdinalEncoder: The final fitted ordinal encoder.
    '''
    
    head = ordinalencoders.pop(0)
    ordinalencoder = copy.deepcopy(head)
    
    # Loop through the remaining encoders in the list
    for enc in ordinalencoders:
        # Merge the encoder 'enc' with all the previous encoders in the list
        ordinalencoder = _merge_ordinal_encoders(ordinalencoder, enc)
        
    # Return the final ordinal encoder after merging all partial fitted encoders
    return ordinalencoder


def _merge_ordinal_encoders(ordinalencoder1: OrdinalEncoder, ordinalencoder2: OrdinalEncoder) -> OrdinalEncoder:
    '''
    Merges two partial fitted OrdinalEncoders into a single OrdinalEncoder 
    by updating the categories of the first encoder.

    Args:
        ordinalencoder1 (OrdinalEncoder): The first OrdinalEncoder to be merged. 
        This encoder will be updated with the unified categories.
        ordinalencoder2 (OrdinalEncoder): The second OrdinalEncoder to be merged.

    Returns:
        OrdinalEncoder: The updated OrdinalEncoder with the unified categories.
    '''
    
    merged_categories = []

    # Extract the categories from the first encoder
    cat1 = ordinalencoder1.categories_
    
    # Extract the categories from the second encoder
    cat2 = ordinalencoder2.categories_
   
    # Compute the updated categories of each feature by unifying the categories from both encoders
    for last_categories, new_categories in zip(cat1, cat2):
        updated_categories = np.union1d(new_categories, last_categories)
        merged_categories.append(updated_categories)
    
    # Update the categories of the first encoder
    ordinalencoder1.categories_ = merged_categories
    
    return ordinalencoder1