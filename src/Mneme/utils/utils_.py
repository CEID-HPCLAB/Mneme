from typing import Union
import logging
from enum import Enum



def _copy_attr(target_obj, source_obj) -> None:
    '''
    Copies attributes from a source object to a target object.

    This function iterates over all public attributes
    of the source object and sets the same attribute for the target object.

    Args:
        target_obj (object): The object to which attributes will be copied.
        source_obj (preprocessorType): The object from which attributes will be copied.

    Returns:
        None
    '''
    
    # Iterate over all attributes of the source object
    for attr in vars(source_obj):
        # Set the same attribute for the target object
        setattr(target_obj, attr, getattr(source_obj, attr))


def set_logging_level(logger: logging.Logger, level: Enum) -> None:
    '''
    Sets the logging level for the Mneme logger.

   Args:
        level (Enum): The logging level to set. Default logging level: LogLevel.INFO.
        e.g. LogLevel.BENCHMARK

    Returns:
        None
    '''
    
    logger.setLevel(level.value)

    