def map_innermost(function, sequence, **kwargs):
    ''' 
    apply function to innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return ([map_innermost(function, element, **kwargs) for element in sequence]
            if isinstance(sequence, kwargs.get("classes", list)) else function(sequence))


def enumerate_map_innermost(function, sequence, indices=[], **kwargs):
    ''' 
    apply function to innermost elements with indices.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return ([enumerate_map_innermost(function, element, indices + [index], **kwargs) for index, element in enumerate(sequence)]
            if isinstance(sequence, kwargs.get("classes", list)) else function(indices, sequence))


def zip_innermost(*sequences, **kwargs):
    ''' 
    make tuple of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return ([zip_innermost(*elements, **kwargs) for elements in zip(*sequences)]
            if all([isinstance(sequence, kwargs.get("classes", list)) for sequence in sequences]) else sequences)


def all_innermost(sequence, **kwargs):
    ''' 
    return whether all innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (all([all_innermost(element, **kwargs) for element in sequence])
            if isinstance(sequence, kwargs.get("classes", list)) else bool(sequence))


def any_innermost(sequence, **kwargs):
    ''' 
    return whether any innermost elements are evaluated as True.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (any([any_innermost(element, **kwargs) for element in sequence])
            if isinstance(sequence, kwargs.get("classes", list)) else bool(sequence))
