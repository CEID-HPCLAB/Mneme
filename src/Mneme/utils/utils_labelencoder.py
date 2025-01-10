import numpy as np
import copy
from typing import List
from sklearn.preprocessing import LabelEncoder


'''
This file contains utility functions for handling LabelEncoder objects in a machine learning pipeline. 
It includes functions for reducing a list of partially fitted LabelEncoders into a single LabelEncoder. 
These functions are designed to streamline the process of working with LabelEncoder instances, 
particularly in scenarios where data preprocessing steps are distributed or parallelized.
'''

def reduce_label_encoders(labelencoders: List[LabelEncoder]) -> LabelEncoder:
    '''
    Reduces a list of partial fitted LabelEncoders into a single final fitted 
    LabelEncoder.

    Args:
        encoders (List[LabelEncoder]): A list of LabelEncoders to be reduced.

    Returns:
        LabelEncoder: The final fitted ordinal encoder.
    '''
    
    head = labelencoders.pop(0)
    labelencoder = copy.deepcopy(head)
    
    # Loop through the remaining encoders in the list
    for enc in labelencoders:
        # Merge the encoder 'enc' with all the previous encoders in the list
        labelencoder = _merge_label_encoders(labelencoder, enc)

    # Return the final label encoder after merging all partial fitted encoders
    return labelencoder


def _merge_label_encoders(labelencoder1: LabelEncoder, labelencoder2: LabelEncoder) -> LabelEncoder:
    '''
    Merges two partial fitted LabelEncoders into a single LabelEncoder 
    by updating the classes of the first encoder.

    Args:
        labelencoder1 (LabelEncoder): The first LabelEncoder to be merged. 
        This encoder will be updated with the unified classes.
        labelencoder2 (LabelEncoder): The second LabelEncoder to be merged.

    Returns:
        LabelEncoder: The updated LabelEncoder with the unified classes.
    '''
    
    # Extract the classes from the first encoder
    last_classes = labelencoder1.classes_
    
    # Extract the classes from the second encoder
    new_classes = labelencoder2.classes_

    # Compute the updated classes by unifying the classes from both encoders
    updated_classes = np.union1d(new_classes, last_classes)
    
    # Update the classes of the first encoder
    labelencoder1.classes_ = updated_classes

    return labelencoder1