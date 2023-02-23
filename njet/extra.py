from more_itertools import distinct_permutations

from . import jet
from .common import factorials

def accel_asc(n: int):
    '''
    Compute the partitions of a given integer.
    
    Parameters
    ----------
    n: int
    
    Returns
    -------
    generator
        A generator, producing lists of integers which sum up to n. Every
        list hereby represents a unique set of integers.
    '''
    # taken from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def symtensor_call(z, terms: dict):
    '''
    Let 'terms' represent the terms of a *homogeneous* jetpoly object 
    P = sum_{j1, ..., jr} P_{j1, ..., jr} * x_1^{n_1} * ... * x_r^{n_r}
    of degree k (i.e. k = j1 + ... + jr).
    
    Then this routine will compute, for a given set of k vectors z = [z1, ..., zk]
    the result
    P[z] = sum_{j1, ..., jr} P_{j1, ..., jr} * z1_1^{n_1} * ... * zk_r^{n_r},
    where zj_w denotes the w-th component of the vector zj.
    
    Parameters
    ----------
    z: list
        A list of iterables, where each element of z represents a vector of the same dimension.
        
    terms: dict
        A dictionary representing the terms of a homogeneous jetpoly object. 
        Attention: No check will be made if the jetpoly object is actually homogeneous.
        
    Returns
    -------
    jetpoly
        A jetpoly object, which is the result of inserting the given vectors into 
        the polynomial, as described above.
    '''
    # Further information can be found here:
    # https://mathoverflow.net/questions/106323/faa-di-brunos-formula-for-vector-valued-functions
    dim = len(z)
    result = 0
    for fs, v in terms.items():
        f = 1
        k = 0
        for index, power in fs:
            for p in range(power):
                f *= z[k][index]
                k += 1
        result += f*v
    return result

def general_faa_di_bruno(f, g):
    '''
    Faa di Bruno for vector-valued functions.
    
    Let G: K^l -> K^n and F: K^n -> K^m be two vector-valued functions. The
    goal is to compute the higher-order derivatives of the composition F o G.
    
    Assume that f = [f1, ..., fn] and g = [g1, ..., gn] represent the 
    n-jet-collections of F and G, i.e.
    fk = n-jet(fk_1, ..., fk_n)
    where fk_j represents the j-th derivative of the k-th component of F etc.
    
    Parameters
    ----------
    f: list
        A list of n-jet objects, representing the derivatives of the function F
        at position G(z).
        
    g: list
        A list of n-jet objects, representing the derivatives of the function G at
        a given position z.
    
    Returns
    -------
    list
        A list of jet objects, representing the higher-orders of the components
        of the compositional map F o G.
    '''
    # Further information can be found here:
    # https://mathoverflow.net/questions/106323/faa-di-brunos-formula-for-vector-valued-functions
    n_dim = len(g) # (== dimension of f)
    max_order = f[0].order # number of requested derivatives
    assert all([jf.order == max_order for jf in f] + [jg.order == max_order for jg in g])
    
    facts = factorials(max_order)
    
    out = [[fk.array(0) for fk in f]]
    for order in range(1, max_order + 1):
        out_order = [0]*n_dim
        nfac = facts[order]
        for j in accel_asc(order):
            r = len(j)
            for e in distinct_permutations(j):
                for k in range(n_dim):
                    jfk = f[k]
                    if jfk.array(r) == 0:
                        # skip in case that the r-th derivative of the k-th component of jfk does not exist
                        continue
                    out_order[k] += symtensor_call([[jg.array(nj)/facts[nj] for jg in g] for nj in e], jfk.array(r).terms)/facts[r]*nfac
                
        out.append(out_order)
    return [jet(*[out[k][j] for k in range(max_order + 1)], n=max_order) for j in range(n_dim)]