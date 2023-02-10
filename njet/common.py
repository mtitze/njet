def check_zero(value):
    # check if a value is zero; value may be an iterable
    check = value == 0
    if hasattr(check, '__iter__'):
        try:
            return check.all()
        except:
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
    
def _nCr(n, r=[]):
    # Generator idea for n_over_k's,
    # taken from https://stackoverflow.com/questions/24093387/pascals-triangle-for-python
    for x in range(n + 1):
        l = len(r)
        r = [1 if i == 0 or i == l else r[i - 1] + r[i] for i in range(l + 1)]
        yield r
        
def nCr(n: int):
    return list(_nCr(n))

def convert_indices(list_of_tuples):
    '''
    Convert a list of tuples denoting the indices in a multivariate Taylor expansion 
    into a list of indices of the corresponding multilinear map.

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

def buildTruncatedJetFunction(*func, truncate=float('inf'), n_args: int=1):
    '''
    Modify a given chain of functions so that the output
    will be truncated between two function calls -- and at the end.
    
    Parameters
    ----------
    *func: callable(s)
        Functions which should be truncated. Note that these functions must support
        jets as input parameters.
    
    truncate: int, optional
        The power beyond which powers should be dropped.
        
    n_args: int, optional
        The number of input parameters of the series of functions.
        a) If n_args == 1, then it is assumed that those functions return
        individual jets/values. 
        b) If n_args > 1, it is assumed that *all* functions
        return iterables (vectors; their lengths may vary depending on the functions). 
        In case b) the user has to ensure that even for functions
        which take one argument, those functions return iterables of length 1.
        
    Returns
    -------
    callable
        A function taking n_args jet objects. 
    '''
    if n_args == 1: # we assume the output is not iterable here
        def tchain(z, **kwargs):
            for f in func:
                z = f(z, **kwargs)
                z = z.truncate(truncate)
            return z
    else:
        def tchain(*z, **kwargs):
            for f in func:
                z = f(*z, **kwargs)
                z = (*[ev.truncate(truncate) for ev in z],)
            return z
    return tchain