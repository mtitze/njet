import numpy as np
from more_itertools import distinct_permutations
from tqdm import tqdm

from . import jet, derive, get_taylor_coefficients
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

def _make_run_params(max_order: int):
    '''
    Compute the summation indices for a given order and load them into memory.
    This routine is intended to be used in order to avoid re-calculating 
    these indices every time the general Faa di Bruno routine is called (for a specific
    max_order).
    
    Parameters
    ----------
    max_order: int
        The requested maximal order up to which the derivatives should be computed.
        
    Returns
    -------
    list
        A list of tuples, where each tuple T = (order, r, [n1, ..., nr]) represents the
        order number, the length r of the partition [n1, ..., nr] of order and the partition itself.
    '''
    facts = factorials(max_order)
    summation_indices = [(order, len(j), dp) for order in range(1, max_order + 1) for j in accel_asc(order) for dp in distinct_permutations(j)]
    return facts, summation_indices

def general_faa_di_bruno(f, g, run_params=()):
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
    n_dim = len(g) # (== domain dimension of f)
    max_order = f[0].order # number of requested derivatives
    assert all([jf.order == max_order for jf in f] + [jg.order == max_order for jg in g])
    
    if len(run_params) > 0:
        facts, indices = run_params
    else:
        facts, indices = _make_run_params(max_order)
    
    out = [[fk.array(0) for fk in f]] + [[0 for k in range(n_dim)] for l in range(max_order)]
    for order, r, e in indices:
        for k in range(n_dim):
            jfk = f[k]
            if jfk.array(r) == 0: 
                # skip in case that the r-th derivative of the k-th component of jfk does not exist
                continue
            out[order][k] += symtensor_call([[jg.array(nj)/facts[nj] for jg in g] for nj in e], jfk.array(r).terms)/facts[r]*facts[order]
    return [jet(*[out[k][j] for k in range(max_order + 1)], n=max_order) for j in range(n_dim)]

class derive_chain:
    '''
    Derive a chain of functions with repetitions, utilizing numpy. 
    The given functions should be unique, while their repetition in the chain
    will be given by an optional ordering. This might have better
    performance than deriving the entire chain of functions
    with the default 'derive' method.
    '''
    def __init__(self, functions, order: int, ordering=[], **kwargs):
        '''
        Parameters
        ----------
        functions: callable(s)
            The unique functions in the chain to be derived.
            
        order: int
            The maximal order of the derivatives to be computed.
            
        ordering: list
            The order defining how the unique functions are arranged in the chain.
            
        **kwargs
            Optional keyworded arguments passed to njet.ad.derive init.
        '''

        if len(ordering) == 0:
            ordering = range(len(functions))

        self.functions = functions
        self.dfunctions = [derive(f, order=order, **kwargs) for f in functions]
        self.n_functions = len(functions)
        self.ordering = ordering
        self.n_chain = len(ordering)
        self.order = order
        self.factorials, self.run_indices = _make_run_params(order)
            
    def probe(self, *point, **kwargs):
        '''
        Run a point through the chain once, to determine the point(s) at which
        the derivative(s) should be calculated.
        '''
        out = [point]
        for k in range(self.n_chain):
            f = self.functions[self.ordering[k]]
            point = f(*point, **kwargs)
            out.append(point)
            
        result = [[out[l] for l in range(self.n_chain) if self.ordering[l] == k] for k in range(self.n_functions)]
        # let Q = points_per_function[j], so Q is a list of points which needs to be computed for function j
        # Then the first element in Q is the one which needs to be applied first, etc. (for element j) by this construction.
        return out, result
    
    def eval(self, *point, **kwargs):
        # 1) Probe the chain: Determine the number of points at which to take the derivative
        self._out, self._probe = self.probe(*point, **kwargs)
        
        # 2) Take the derivative for every unique function in the chain, using numpy arrays for each function.
        deval = []
        for k in tqdm(range(self.n_functions), disable=kwargs.get('disable_tqdm', False)):
            function = self.functions[k]
            n_args_function = self.dfunctions[k].n_args
            points_at_function = self._probe[k]
            components = [np.array([points_at_function[j][l] for j in range(len(points_at_function))], dtype=np.complex128) for l in range(n_args_function)]
            deval.append(self.dfunctions[k].eval(*components, order=self.order, n_args=n_args_function, **kwargs))

        # 3) Compose the derivatives for the entire chain.
        ev0 = [e[0] for e in deval[self.ordering[0]]] # the start is the derivative of the first element at the point of interest
        evr = ev0
        ele_indices = {k: 0 for k in range(self.n_functions)} # keep track of the current passage number for every element
        ele_indices[self.ordering[0]] = 1 # we already tooked the first one.
        self._evaluation_function = [ev0]
        for k in tqdm(range(1, self.n_chain), disable=kwargs.get('disable_tqdm', False)):
            pos = self.ordering[k]
            index_passage = ele_indices[pos]
            ev = [j[index_passage] for j in deval[pos]]
            evr = general_faa_di_bruno(ev, evr, run_params=(self.factorials, self.run_indices))
            ele_indices[pos] += 1
            self._evaluation_function.append(evr)
        self._evaluation = evr
        return evr
    
    def __call__(self, *z, **kwargs):
        # These two keywords are reserved for the get_taylor_coefficients routine and will be removed from the input:
        mult_prm = kwargs.pop('mult_prm', True)
        mult_drv = kwargs.pop('mult_drv', True)
        # perform the computation, based on the input vector
        return get_taylor_coefficients(self.eval(*z, **kwargs), n_args=self.dfunctions[0].n_args, mult_prm=mult_prm, mult_drv=mult_drv)
