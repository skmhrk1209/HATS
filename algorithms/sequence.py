from operator import *
from functools import *


def map_innermost(function, sequence, **kwargs):
    '''
    apply function to innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda element: map_innermost(function, element, **kwargs), sequence))
            if isinstance(sequence, kwargs.get("classes", list)) else function(sequence))


def enumerate_innermost(sequence, indices=(), **kwargs):
    '''
    make tuple of innermost element and index.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda index_element: enumerate_innermost(index_element[1], indices + (index_element[0],), **kwargs), enumerate(sequence)))
            if isinstance(sequence, kwargs.get("classes", list)) else (indices, sequence))


def zip_innermost(*sequences, **kwargs):
    '''
    make tuple of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost(*elements, **kwargs), zip(*sequences)))
            if all(map(lambda sequence: isinstance(sequence, kwargs.get("classes", list)), sequences)) else sequences)


def all_innermost(sequence, **kwargs):
    '''
    return whether all innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (all(map(lambda element: all_innermost(element, **kwargs), sequence))
            if isinstance(sequence, kwargs.get("classes", list)) else bool(sequence))


def any_innermost(sequence, **kwargs):
    '''
    return whether any innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (any(map(lambda element: any_innermost(element, **kwargs), sequence))
            if isinstance(sequence, kwargs.get("classes", list)) else bool(sequence))


def flatten_innermost(sequence, **kwargs):
    '''
    return flattened list of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost(element, **kwargs), sequence))
            if isinstance(sequence, kwargs.get("classes", list)) else [sequence])
