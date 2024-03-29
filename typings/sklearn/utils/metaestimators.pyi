"""
This type stub file was generated by pyright.
"""

from typing import Any, List
from abc import ABCMeta, abstractmethod
from ..base import BaseEstimator

"""Utilities for meta-estimators"""
__all__ = ['if_delegate_has_method']
class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for classifiers composed of named estimators.
    """
    steps: List[Any]
    @abstractmethod
    def __init__(self) -> None:
        ...
    


class _IffHasAttrDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if none of the delegates (specified in ``delegate_names``) is an attribute
    of the base object or the first found delegate does not have an attribute
    ``attribute_name``.

    This allows ducktyping of the decorated method based on
    ``delegate.attribute_name``. Here ``delegate`` is the first item in
    ``delegate_names`` for which ``hasattr(object, delegate) is True``.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """
    def __init__(self, fn, delegate_names, attribute_name) -> None:
        ...
    
    def __get__(self, obj, type=...):
        ...
    


def if_delegate_has_method(delegate): # -> (fn: Unknown) -> _IffHasAttrDescriptor:
    """Create a decorator for methods that are delegated to a sub-estimator

    This enables ducktyping by hasattr returning True according to the
    sub-estimator.

    Parameters
    ----------
    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.

    """
    ...

