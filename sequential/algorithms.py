from operator import *
from functools import *


def map_innermost(function, sequence, classes=(list,)):
    '''
    apply function to innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda element: map_innermost(function, element, classes), sequence))
            if isinstance(sequence, classes) else function(sequence))


def enumerate_innermost(sequence, classes=(list,), indices=()):
    '''
    make tuple of innermost element and index.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda index_element: enumerate_innermost(index_element[1], classes, indices + (index_element[0],)), enumerate(sequence)))
            if isinstance(sequence, classes) else (indices, sequence))


def zip_innermost(*sequences, classes=(list,)):
    '''
    make tuple of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost(*elements, classes), zip(*sequences)))
            if all(map(lambda sequence: isinstance(sequence, classes), sequences)) else sequences)


def all_innermost(sequence, classes=(list,)):
    '''
    return whether all innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (all(map(lambda element: all_innermost(element, classes), sequence))
            if isinstance(sequence, classes) else bool(sequence))


def any_innermost(sequence, classes=(list,)):
    '''
    return whether any innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (any(map(lambda element: any_innermost(element, classes), sequence))
            if isinstance(sequence, classes) else bool(sequence))


def flatten_innermost(sequence, classes=(list,)):
    '''
    return flattened list of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost(element, classes), sequence))
            if isinstance(sequence, classes) else [sequence])
