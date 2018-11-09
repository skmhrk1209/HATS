def map_innermost(function, sequence):

    return ([map_innermost(function, element) for element in sequence]
            if isinstance(sequence, list) else function(sequence))

def enumerate_map_innermost(function, sequence, indices=[]):

    return ([enumerate_map_innermost(function, element, indices + [index]) for index, element in enumerate(sequence)]
            if isinstance(sequence, list) else function(indices, sequence))

def zip_innermost(*sequences):

    return ([zip_innermost(*elements) for elements in zip(*sequences)]
            if all([isinstance(sequence, list) for sequence in sequences]) else sequences)


def nest_depth(sequence):

    return (max([nest_depth(element) for element in sequence]) + 1
            if isinstance(sequence, list) else 0)