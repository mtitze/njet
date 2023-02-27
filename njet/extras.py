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
    Derive a chain of vector-valued functions with repetitions. 
    The given functions should be unique, while their repetition in the chain
    will be given by an optional ordering. This might have better
    performance than deriving the entire chain of functions
    with the default 'derive' method.
    '''
    def __init__(self, functions, order: int, ordering=[], **kwargs):
        '''
        Parameters
        ----------
        functions: callable(s) or 'derive' classes
            The unique functions in the chain to be derived.
            Alternatively, one can pass a list of 'derive' objects. 
            In this second case it will be assumed (by default, see 'probed' parameter below)
            that the 'derived' classes already contain the jet evaluations
            for the run. This can be used to avoid re-calculating the derivatives
            in circumstances where it is not required.
            
        order: int
            The maximal order of the derivatives to be computed.
            
        ordering: list
            The order defining how the unique functions are arranged in the chain.
            Hereby the index j must refer to the function at position j in the list.
            
        **kwargs
            Optional keyworded arguments passed to njet.ad.derive init.
        '''

        if len(ordering) == 0:
            ordering = list(range(len(functions)))

        # Determine user input for the 'functions' parameter
        supported_objects = ['njet.ad.derive', 'njet.extras.derive_chain'] # objects of these kinds will not be instantiated with 'derive'
        
        self.dfunctions = [f if any([s in f.__repr__() for s in supported_objects]) else derive(f, order=order, **kwargs) for f in functions]
            
        self.n_functions = len(functions)
        self.ordering = ordering
        self.chain_length = len(ordering)
        self.order = order
        self.factorials, self.run_indices = _make_run_params(order)
        
        # For every element in the chain, note a number at which a point
        # passing through the given chain will pass through the element.
        self.path = {k: [] for k in range(self.n_functions)}
        for j in range(self.chain_length):
            self.path[self.ordering[j]].append(j)
            
    def probe(self, *point, **kwargs):
        '''
        Run a point through the chain once, to determine the point(s) at which
        the derivative(s) should be calculated.
        '''
        out = [point]
        for k in range(self.chain_length):
            point = self.dfunctions[self.ordering[k]].jetfunc(*point, **kwargs)
            out.append(point)
        self._probe_out = out
        return out
    
    def _probe_check(self, *point, **kwargs):
        '''
        Check if the points in the probe agree with the one stored in the input functions.
        This check requires that the input functions have been evaluated and a probe run has been performed.
        '''
        if not all([hasattr(df, '_input') for df in self.dfunctions]):
            # Nothing can be compared, so the check will fail
            return False
        elif not hasattr(self, '_probe_out'): 
            # Probe the current chain
            self.probe(*point, **kwargs)
            
        if not np.array([point[k] == self.dfunctions[self.ordering[0]]._input[k][0].array(0) for k in range(len(point))]).all():
            # If the input point disagrees with the stored input point, then return false.
            return False
        else:
            # Check the remaining points along the chain
            return all([np.array([self.dfunctions[self.ordering[k]]._input[component_index][self.path[self.ordering[k]].index(k)].array(0) == self._probe_out[k][component_index] for component_index in range(len(self._probe_out[k]))]).all() for k in range(1, self.chain_length)])
    
    def eval(self, *point, **kwargs):
        '''
        Evaluate the individual (unique) functions in the chain at the requested point.
        '''
        out = self.probe(*point, **kwargs)
        points_at_functions = [[out[l] for l in range(self.chain_length) if self.ordering[l] == k] for k in range(self.n_functions)]
        # let Q = points_per_function[j], so Q is a list of points which needs to be computed for function j
        # Then the first element in Q is the one which needs to be applied first, etc. (for element j) by this construction.
        for k in tqdm(range(self.n_functions), disable=kwargs.get('disable_tqdm', False)):
            function = self.dfunctions[k].jetfunc
            n_args_function = self.dfunctions[k].n_args
            points_at_function = points_at_functions[k]
            components = [np.array([points_at_function[j][l] for j in range(len(points_at_function))], dtype=np.complex128) for l in range(n_args_function)]
            _ = self.dfunctions[k].eval(*components, order=self.order, n_args=n_args_function, **kwargs)
        return self.compose(**kwargs)
    
    def compose(self, **kwargs):
        '''
        Compose the given derivatives for the entire chain.      
        '''
        evr = [e[0] for e in self.dfunctions[self.ordering[0]]._evaluation] # the start is the derivative of the first element at the point of interest
        self._evaluation_chain = [evr]
        for k in tqdm(range(1, self.chain_length), disable=kwargs.get('disable_tqdm', False)):
            pos = self.ordering[k]
            index_passage = self.path[pos].index(k)
            ev = [j[index_passage] for j in self.dfunctions[pos]._evaluation]
            evr = general_faa_di_bruno(ev, evr, run_params=(self.factorials, self.run_indices))
            self._evaluation_chain.append(evr)
        self._evaluation = evr
        return evr
    
    def __call__(self, *z, **kwargs):
        '''
        Compute the derivatives of the chain of functions at a given point.
        '''
        # These two keywords are reserved for the get_taylor_coefficients routine and will be removed from the input:
        mult_prm = kwargs.pop('mult_prm', True)
        mult_drv = kwargs.pop('mult_drv', True)
        
        # Determine if a (re-)evaluation is required
        eval_required = False
        if hasattr(self, '_call_kwargs'):
            eval_required = self._call_kwargs != kwargs
        if not all([hasattr(df, '_evaluation') for df in self.dfunctions]):
            eval_required = True
        elif not self._probe_check(*z, **kwargs):
            eval_required = True
        self._call_kwargs = kwargs
        
        # Perform the composition, if necessary
        if eval_required:
            _ = self.eval(*z, **kwargs) # evaluation includes 'self.compose'
        else:
            _ = self.compose(**kwargs)
            
        return get_taylor_coefficients(self._evaluation, n_args=self.dfunctions[0].n_args, mult_prm=mult_prm, mult_drv=mult_drv)