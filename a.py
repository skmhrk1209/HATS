def enumerate_map_innermost(function, sequence, indices=[]):

    return ([enumerate_map_innermost(function, element, indices + [index]) for index, element in enumerate(sequence)]
            if isinstance(sequence, list) else function(indices, sequence))

print(enumerate_map_innermost(lambda indices, x: (indices, x), [[1, 2], [4, [3, 1]]]))