from operator import *
from functools import *


def compose(func, *funcs):
    '''
    conpose functions from left to right
    '''

    return lambda *args: compose(*funcs)(func(*args)) if funcs else func(*args)


def map_innermost_element(func, seq, classes=(list,)):
    '''
    apply function to innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(seq)(map(lambda element: map_innermost_element(func, element, classes=classes), seq))
            if isinstance(seq, classes) else func(seq))


def map_innermost_list(func, seq, classes=(list,)):
    '''
    apply function to innermost lists.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (type(seq)(map(lambda element: map_innermost_list(func, element, classes=classes), seq))
            if isinstance(seq, classes) and any(map(lambda element: isinstance(element, classes), seq)) else func(seq))


def enumerate_innermost_element(seq, classes=(list,), indices=()):
    '''
    make tuple of innermost element and index.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(seq)(map(lambda index_element: enumerate_innermost_element(index_element[1], classes=classes, indices=indices + (index_element[0],)), enumerate(seq)))
            if isinstance(seq, classes) else (indices, seq))


def enumerate_innermost_list(seq, classes=(list,), indices=()):
    '''
    make tuple of innermost element and index.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (type(seq)(map(lambda index_element: enumerate_innermost_list(index_element[1], classes=classes, indices=indices + (index_element[0],)), enumerate(seq)))
            if isinstance(seq, classes) and any(map(lambda element: isinstance(element, classes), seq)) else (indices, seq))


def zip_innermost_element(*seqs, classes=(list,)):
    '''
    make tuple of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost_element(*elements, classes=classes), zip(*seqs)))
            if all(map(lambda sequence: isinstance(sequence, classes), seqs)) else seqs)


def zip_innermost_list(*seqs, classes=(list,)):
    '''
    make tuple of innermost elements.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost_list(*elements, classes=classes), zip(*seqs)))
            if all(map(lambda sequence: isinstance(sequence, classes) and any(map(lambda element: isinstance(element, classes), sequence)), seqs)) else seqs)


def flatten_innermost_element(seq, classes=(list,)):
    '''
    return flattened list of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost_element(element, classes=classes), seq), [])
            if isinstance(seq, classes) else [seq])


def flatten_innermost_list(seq, classes=(list,)):
    '''
    return flattened list of innermost elements.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost_list(element, classes=classes), seq))
            if isinstance(seq, classes) and any(map(lambda element: isinstance(element, classes), seq)) else [seq])
