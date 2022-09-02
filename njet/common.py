def check_zero(value):
    # check if a value is zero; value may be an iterable
    check = value == 0
    if hasattr(check, '__iter__'):
        return all(check)
    else:
        return check
    
def factorials(n: int):
    k = 1
    facts = [1]
    for j in range(1, n + 1):
        k *= j
        facts.append(k)
    return facts

def n_over_ks(n: int):
    facts = factorials(n)
    return [[facts[j]//(facts[k]*facts[j - k]) for k in range(j + 1)] for j in range(len(facts))]

def convert_indices(list_of_tuples):
    '''
    Convert a list of tuples denoting the indices in a multivariate Taylor expansion into a list of indices of
    the corresponding multilinear map.

    Parameters
    ----------
    list_of_tuples: list
        List of (self.n_args)-tuples denoting the indices in the multivariate Taylor expansion.

    Returns
    -------
    list
        List of tuples denoting the indices of a multilinear map.

    Example
    -------
    list_of_tuples = [(0, 2, 1, 9), (1, 0, 3, 0)]
    In this example we are looking at indices in the Taylor expansion of a function in 4 variables.
    The first member, (0, 2, 1, 9), corresponds to x0**0*x1**2*x2*x3**9 which belongs to the multilinear map
    of order 0 + 2 + 1 + 9 = 12, the second member to x0*x2**3, belonging to the multilinear map of order
    1 + 3 = 4.
    Hence, these indices will be transformed to tuples of length 12 and 4, respectively:
    (1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3) (variable 1 has power 2, variable 2 has power 1, variable 3 has power 9) 
    and
    (0, 2, 2, 2) (variable 0 has power 1, variable 2 has power 3).
    '''
    l1g = [[tpl[j]*[j] for j in range(len(tpl))] for tpl in list_of_tuples]
    return [tuple(e for subl in l1 for e in subl) for l1 in l1g] # flatten l1 and return list of converted tuples
