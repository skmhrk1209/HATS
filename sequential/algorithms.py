from operator import *
from functools import *


def map_innermost(function, sequence, predicate):

    return function(sequence) if predicate(sequence) else type(sequence)(map(lambda element: map_innermost(function, element, predicate), sequence))


def enumerate_innermost(sequence, indices=(), predicate):

    return (indices, sequence) if predicate(sequence) else type(sequence)(map(lambda index_element: enumerate_innermost(index_element[1], indices + (index_element[0],), predicate), enumerate(sequence)))


def zip_innermost(sequences, predicate):

    return sequences if predicate(sequences) else [zip_innermost(elements, predicate) for elements in zip(*sequences)]


def all_innermost(sequence, predicate):

    return bool(sequence) if predicate(sequence) else all(map(lambda element: all_innermost(element, predicate), sequence))


def any_innermost(sequence, predicate):

    return bool(sequence) if predicate(sequence) else any(map(lambda element: any_innermost(element, predicate), sequence))


def flatten_innermost(sequence, predicate):

    return [sequence] if predicate(sequence) else reduce(add, map(lambda element: flatten_innermost(element, predicate), sequence))
