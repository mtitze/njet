import numpy as np
from more_itertools import distinct_permutations, windowed
from tqdm import tqdm
from copy import copy
import warnings
import gc

from . import jet, jetpoly, derive, taylor_coefficients
from .common import factorials, check_zero

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
    return {'factorials': facts, 'indices': summation_indices}

def general_faa_di_bruno(f, g, run_params={}):
    '''
    Faa di Bruno for vector-valued functions.
    
    Let G: K^l -> K^m and F: K^m -> K^w be two vector-valued functions. The
    goal is to compute the higher-order derivatives of the composition F o G.
    
    Assume that f = [f1, ..., fw] and g = [g1, ..., gm] represent the 
    n-jet-collections of F and G, i.e. fk = n-jet(fk_1, ..., fk_n)
    where fk_j represents the j-th derivative of the k-th component of F etc.
    
    Note that this routine requires that both f and g are represented by jetpoly
    objects in their higher-order array entries (they store the various partial derivatives
    for the given orders).
    In particular, the ordinary derive.eval routine will produce such objects. 
    This is the major difference to the one-dimensional Faa di Bruno formula 
    in njet.jet.faa_di_bruno, which has no such requirement.
    
    Parameters
    ----------
    f: list
        A list of n-jet objects, representing the derivatives of the function F
        at position G(z).
        
    g: list
        A list of n-jet objects, representing the derivatives of the function G at
        a given position z.
        
    run_params: dict, optional
        A dictionary containing the output of njet.extras._make_run_params. These parameters
        can be send to the routine to avoid internal re-calculation when the routine
        it is called repeatedly.
    
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
    
    if len(run_params) == 0:
        run_params = _make_run_params(max_order)
    facts, indices = run_params['factorials'], run_params['indices']
    
    out = [[fk.array(0) for fk in f]] + [[0 for k in range(n_dim)] for l in range(max_order)]
    for order, r, e in indices:
        for k in range(n_dim):
            jfk_r = f[k].array(r)
            if check_zero(jfk_r): 
                # skip in case that the r-th derivative of the k-th component of jfk does not exist
                continue
            out[order][k] += symtensor_call([[jg.array(nj)/facts[nj] for jg in g] for nj in e], jfk_r.terms)/facts[r]*facts[order]
    return [jet(*[out[k][j] for k in range(max_order + 1)], n=max_order) for j in range(n_dim)]

def compose(*evals, run_params={}, **kwargs):
    '''
    Compose derivatives of a chain of vector-valued jet-evaluations.

    Parameters
    ----------
    evals: 
        A series of (multi-dimensional) jet-evaluations. Hereby every jet-evaluation 
        is represented by a list of n-jets, one for each individual component.

    **kwargs: dict, optional
        One can pass the boolean parameter disable_tqdm to enable (if False) a progress bar.

    Returns
    -------
    list
        A list of jet evaluations for each vector component of the chain, representing
        the Taylor-expansion of the chain.
    '''
    if len(run_params) == 0:
        max_order = max([e.order for f in evals for e in f])
        run_params = _make_run_params(max_order)
    
    evr = evals[0]
    out = [evr]
    for k in tqdm(range(1, len(evals)), disable=kwargs.get('disable_tqdm', True)):
        evr = general_faa_di_bruno(evals[k], evr, run_params=run_params)
        out.append(evr)
    return out

#################################
#   Jet manipulation routines   #
#################################

def tile(jev, ncopies: int):
    '''
    Construct a new jet in which its entries contain copies of itself.
    
    Intended to work if the jet entries are numpy arrays. A jet with
    scalar values can be converted to a jet containing numpy arrays
    of shape (1,) by using ncopies=1.
    
    Parameters
    ----------
    jev: jet
    
    ncopies: int
        Number of desired copies.
    
    Returns
    -------
    jet
        A jet having arrays of length ncopies + 1
    '''
    assert ncopies >= 0
    if ncopies == 0:
        return jev
    new_jet_array = []
    for k in range(jev.order + 1):
        jk = jev.array(k)
        if hasattr(jk, 'terms'): # Assume the entry is of type jetpoly
            entry_k = jk.terms
            new_terms_k = {}
            for key, value in entry_k.items():
                new_terms_k[key] = np.tile(value, [ncopies] + [1]*len(value.shape))
            new_jet_array.append(jetpoly(terms=new_terms_k))
        else:
            new_jet_array.append(np.tile(jk, [ncopies] + [1]*len(jk.shape)))
    return jet(*new_jet_array)

def _jetp1(jev, index: int, to_concat0=None, ncopies: int=1, location=-1):
    '''
    Add copies of a unity transformations to the give jet entries -- either
    in the end or at the beginning.
    
    Intended to work if all jet entries are numpy arrays.

    Parameters
    ----------
    jev: jet
    
    index: int
        The component of the unity transformation.
        
    to_concat0: array-like, optional
        The value to be concatenated, representing the 0-entry of the jet array.
        If nothing specified, the last (or first) value of jetv.array(0) will be used.
        
    ncopies: int, optional
        The number of copies attached.
        
    location: int, optional
        -1: Attach to end. Everything else: attach to start.
    
    Returns
    -------
    jet
    '''
    assert ncopies >= 0
    if ncopies == 0:
        return jev
    
    a0 = jev.array(0)
    kj = frozenset({(index, 1)})
    if to_concat0 is None:
        if location == -1:
            to_concat0 = np.array([a0[-1]]*ncopies) # The index '-1' here is related to the use of _jetp1 in the cderive.cycle routine: In case there are several passages through the specific function, we will take a copy of the last passage. This also means that _jetp1 is intended to be used on jets whose entries carry numpy arrays.
        else:
            to_contat0 = np.array([a0[0]]*ncopies)
    else:
        to_concat0 = np.array(to_concat0)
            
    zero = np.zeros(to_concat0.shape)
    one = np.ones(to_concat0.shape)
    concat_ones = np.array([one[0]]*ncopies)
    concat_zeros = np.array([zero[0]]*ncopies)
        
    # the unity transformation will reproduce the value of the jet in the 0-array.
    shape = [a0.shape[0] + to_concat0.shape[0]] + list(a0.shape[1:])
    a0_new = np.empty(tuple(shape), dtype=np.complex128)
    if location == -1:
        a0_new[:a0.shape[0]] = a0
        a0_new[a0.shape[0]:] = to_concat0
    else:
        a0_new[:to_concat0.shape[0]] = to_concat0
        a0_new[to_concat0.shape[0]:] = a0
    
    new_jet_array = [a0_new]
    for k in range(1, jev.order + 1):
        jk = jev.array(k)
        if not hasattr(jk, 'terms'):
            entry_k = {}
        else:
            entry_k = jk.terms
        if k == 1:
            # ensure that kj appears in the new jetpoly in any case:
            _ = entry_k.setdefault(kj, zero)
        new_terms_k = {}
        for key, value in entry_k.items():
            if key == kj: # may happen if k = 1
                to_concatj = concat_ones
            else:
                to_concatj = concat_zeros
                
            
            shape = [value.shape[0] + to_concatj.shape[0]] + list(value.shape[1:])
            nt = np.empty(tuple(shape), dtype=np.complex128)
            if location == -1:
                nt[:value.shape[0]] = value
                nt[value.shape[0]:] = to_concatj
            else:
                nt[:to_concatj.shape[0]] = to_concatj
                nt[to_concatj.shape[0]:] = value
            new_terms_k[key] = nt
            
            del nt
        new_jet_array.append(jetpoly(terms=new_terms_k))
    return jet(*new_jet_array)

def _jbuild(jets):
    '''
    Combine the arrays of the given jets, to return a new jet.
    Internal routine; all jetpoly objects need to have identical keys.

    Parameters
    ----------
    jets: list
        A list of njet.jet objects.

    Returns
    -------
    jet
        A jet in which the entries of each original jet are included in a large numpy array.
    '''
    max_order = max(j.order for j in jets)
    new_array = [np.array([j.array(0) for j in jets])]
    for k in range(1, max_order + 1):
        new_terms = {}
        all_keys = set().union(*[getattr(j.array(k), 'terms', {}).keys() for j in jets]) # if there are no terms in an entry for order >= 1, then the jetpoly term was zero.
        for key in all_keys:
            new_terms[key] = np.array([j.array(k).terms.get(key, 0) for j in jets])
        new_array.append(jetpoly(terms=new_terms))
    return jet(*new_array, n=max_order)

##########################
#   Chain-derive class   #
##########################

class cderive:
    '''
    Class to handle the derivatives of a chain of vector-valued functions with repetitions. 
    The given functions should be unique, while their repetition in the chain
    will be given by an optional ordering.

    Parameters
    ----------
    functions: list
        A list of callable(s) or 'derive' classes related to the
        unique vector-valued*) functions in the chain to be derived.

        Alternatively, one can pass a list of 'derive' objects, containing
        vector-valued jet evaluation output. 
        This second case can be used to avoid re-calculating the derivatives
        in circumstances where it is not required.

        Attention:
        By default, the first function is assumed to be executed first, in contrast
        to the mathematical notation. This can be changed by passing
        an 'ordering' argument (see below).

        *) This means that the functions must return iterables in any case.

    order: int, optional
        The maximal order of the derivatives to be computed. If nothing specified,
        the first derivative(s) will be computed.

    ordering: list, optional
        The order defining how the unique functions are arranged in the chain.
        Hereby the index j must refer to the function at position j in the sequence
        of functions.

    run_params: dict, optional
        A dictionary containing the output of njet.extras._make_run_params. These parameters
        can be send to the routine to avoid internal re-calculation when the routine
        it is called repeatedly.
        
    reset: boolean, optional
        Parameter given to self.set_ordering. If 'True' (default), erase any jet-evaluation data 
        coming along with the input (the ._evaluation fields). To prevent this behavior, 
        set this parameter to 'False'. However, in this case one should ensure that correct 
        data has been stored -- in particular concerning the requested ordering of the chain.

    **kwargs
        Optional keyworded arguments passed to njet.ad.derive init.
    '''

    def __init__(self, *functions, order: int=1, ordering=None, run_params={}, reset=True, **kwargs):

        # Determine user input for the 'functions' parameter
        supported_objects = ['njet.ad.derive', 'njet.extras.cderive'] # objects of these kinds will not be instantiated with 'derive'
        self.dfunctions = [copy(f) if any([s in f.__repr__() for s in supported_objects]) else derive(copy(f), order=order, **kwargs) for f in functions]
        
        self.n_functions = len(self.dfunctions)
        self.order = order
        
        if len(run_params) == 0:
            run_params = _make_run_params(order)
        self.run_params = run_params
        
        self.set_ordering(ordering=ordering, reset=reset)
                
    def set_ordering(self, ordering=None, reset=True):
        '''
        Set the ordering of the current chain. Also compute the number of passages expected through
        each element according to the ordering.
        
        Parameters
        ----------
        ordering: list, optional
            The order defining how the unique functions are arranged in the chain.
            Hereby the index j must refer to the function at position j in the sequence
            of functions.
            
        reset: boolean, optional
            Reset the jet-evaluations as well. This may become necessary because these evaluations
            depend strongly on the current ordering.
        '''
        if ordering is None:
            ordering = list(range(self.n_functions))
        assert len(ordering) > 0, 'Ordering list empty.'
            
        # Check input consistency
        uord = np.unique(ordering).tolist()
        assert len(uord) == self.n_functions and uord[0] == 0, 'Number of functions not consistent with the unique items in the ordering.'

        self.ordering = [k for k in ordering] # copy to prevent modification of input somewhere else
        # For every element in the chain, note a number at which a point
        # passing through the given chain will pass through the element.
        # So for example, self.path[k] = [j1, j2, j3, ...] means that
        # the first passage through element k occurs at global position j1,
        # the second passage through element k occurs at global position j2 etc.
        self.path = {k: [] for k in uord}
        for j in range(len(self)):
            self.path[self.ordering[j]].append(j)

        if not reset:
            # check at least if the (new) path is consistent with the given data:
            _ = self._check_path_consistency()
        else:
            self.reset()
        
    def _check_path_consistency(self):
        '''
        Check if the number of entries for each evaluation (if they exist) is consistent with the internal path enumeration.
        '''
        result = True
        for k in range(len(self)):
            f = self.dfunctions[self.ordering[k]]
            if not hasattr(f, '_evaluation'):
                continue
            index = self.path[self.ordering[k]].index(k)
            if index == 0: # Single passages or possible scalar values will be ignored
                continue
            if not all([index < len(e.array(0)) for e in f._evaluation]): # More data than required will also be ignored.
                warnings.warn(f'Insufficient jet-evaluation entries at position >= {k}. Reset or re-evaluation required.')
                result = False
                break
        return result
                
    def reset(self):
        '''
        Remove all jet evaluations (if they exist).
        '''
        for f in self.dfunctions:
            if hasattr(f, '_evaluation'):
                delattr(f, '_evaluation')
                            
    def jetfunc(self, *point, **kwargs):
        '''
        Run a point through the chain.
        
        Parameters
        ----------
        *point: single value or array-like
            The point at which the chain should be evaluated.
            
        **kwargs: dict, optional
            Keyworded arguments passed to the underlying jet-functions.
            
        Returns
        -------
        single value or array-like
            The final value after the chain of functions has been traversed.
        '''
        self._input = point
        out = [point]
        for k in range(len(self)):
            point = self.dfunctions[self.ordering[k]].jetfunc(*point, **kwargs)
            out.append(point)
        self._output = out
        return point
    
    def _probe(self, *point, **kwargs):
        '''
        Check if the points in the current output are in agreement with the one stored in the input functions.
        This check requires that the input functions have been evaluated and a probe run has been performed.
        
        Parameters
        ----------
        *point: single value or array-like
            The point at which the chain should be probed.
            
        **kwargs: dict, optional
            Keyworded arguments passed to the underlying jet-functions.
            
        Returns
        -------
        boolean
            If True, the currently stored data along the chain will be produced
            by collecting the transverse points starting with the given point through the chain.
        '''
        if hasattr(self, '_input'):
            check = all([check_zero(point[k] - self._input[k]) for k in range(len(point))])
            if check == False:
                return False
            
        if not all([hasattr(df, '_input') for df in self.dfunctions]):
            # Nothing can be compared, so the check will fail
            return False
            
        if not hasattr(self, '_output'): 
            # Probe the current chain
            _ = self.jetfunc(*point, **kwargs)
            
        # Check the points along the chain
        try:
            return all([np.array([self.dfunctions[self.ordering[k]]._input[component_index][self.path[self.ordering[k]].index(k)].array(0) == self._output[k][component_index] for component_index in range(len(self._output[k]))]).all() for k in range(len(self))])
        except:
            warnings.warn('Probe check aborted with error(s).')
            return False
                    
    def eval(self, *point, compose=True, **kwargs):
        '''
        Evaluate the functions in the chain at the requested point.
        
        Parameters
        ----------
        *point: single value or array-like
            The point at which the evaluation should be performed.
            
        compose: boolean, optional
            If true (default), compose the evaluations after they have been computed and 
            return the result, resembling the behaviour of the original njet.derive.eval routine.
            If false, only generate the jet-evaluation data (e.g. in preparation for a succeeding merge command).
            
        **kwargs: dict, optional
            Keyworded arguments passed to the underlying jet-functions.
            
            One can pass the boolean parameter disable_tqdm to enable (if False) a progress bar.
            
        Returns
        -------
        list
            The outcome of self.compose routine (containing the jet-evaluations for each component).
        '''        
        _ = self.jetfunc(*point, **kwargs)
        points_at_functions = [[self._output[l] for l in range(len(self)) if self.ordering[l] == k] for k in range(self.n_functions)]
        # let Q = points_per_function[j], so Q is a list of points which needs to be computed for function j
        # Then the first element in Q is the one which needs to be applied first, etc. (for element j) by this construction.
        
        for k in tqdm(range(self.n_functions), disable=kwargs.get('disable_tqdm', True)):
            function = self.dfunctions[k].jetfunc
            n_args_function = self.dfunctions[k].n_args
            points_at_function = points_at_functions[k]
            components = [np.array([points_at_function[j][l] for j in range(len(points_at_function))], dtype=np.complex128) for l in range(n_args_function)]
            _ = self.dfunctions[k].eval(*components, **kwargs)
            
        if compose:
            return self.compose(**kwargs)
    
    def jev(self, pos: int):
        '''
        Convenience function to obtain jet-evaluation data at a specific position
        (such data can be produced by the 'eval' command).
        
        Parameters
        ----------
        pos: integer
            The position within the chain of functions.
            
        Returns
        -------
        list
            A list of jet-evaluations.
        '''
        return [e[self.path[self.ordering[pos]].index(pos)] for e in self[pos]._evaluation]
    
    def compose(self, **kwargs):
        '''
        Compose the current jet-evaluations in the chain (requires function evaluations in advance).
        
        See njet.extras.compose for details; its output is stored in self._evaluation._chain, while the last element is returned.
        '''
        assert all(hasattr(f, '_evaluation') for f in self.dfunctions), 'Composition requires function evaluations in advance.'
        # the path through the chain will require the selection of the point through which the trajectory passes at the respective position:
        evals = [[e[self.path[self.ordering[pos]].index(pos)] for e in self[pos]._evaluation] for pos in range(len(self))]
        self._evaluation_chain = compose(*evals, run_params=self.run_params, **kwargs)
        self._evaluation = self._evaluation_chain[-1]
        return self._evaluation
    
    def _eval_memcheck(self, *z, **kwargs):
        '''
        Check internal memory if evaluation is required.
        
        Parameters
        ----------
        *z
            The point at which the evaluation should be calculated.
            
        **kwargs
            Additional keyworded parameters intended for the functions in the chain.
        
        Returns
        -------
        eval_required: boolean
            True, if a (re-)evaluation is required.
            
        kwargs_changed: boolean
            True, if the keyworded arguments have been changed in comparison to a previous run.
        '''
        # Determine if the keyworded arguments have been changed
        kwargs_changed = False
        if hasattr(self, '_call_kwargs'):
            kwargs_changed = self._call_kwargs != kwargs
        elif len(kwargs) > 0:
            kwargs_changed = True
            
        # Determine if a (re-)evaluation is required
        eval_required = False
        if len(z) > 0:
            if not all([hasattr(df, '_evaluation') for df in self.dfunctions]):
                eval_required = True
            elif not self._probe(*z, **kwargs):
                eval_required = True
        return eval_required, kwargs_changed

    def __call__(self, *z, **kwargs):
        '''
        Compute the derivatives of the chain of functions at a given point.
        
        Parameters
        ----------
        *z: single value or array-like
            The point at which the derivative(s) should be evaluated.
            
        **kwargs: dict, optional
            Keyworded arguments passed to the underlying jet functions.
            
        Returns
        -------
        dict
            The Taylor-coefficients of the chain at the point of interest.
        '''
        # These two keywords are reserved for the taylor_coefficients routine and will be removed from the input:
        mult_prm = kwargs.pop('mult_prm', True)
        mult_drv = kwargs.pop('mult_drv', True)
               
        # Perform the composition, if necessary
        eval_required, kwargs_changed = self._eval_memcheck(*z, **kwargs)
        if eval_required:
            _ = self.eval(*z, **kwargs)
        if not hasattr(self, '_evaluation'):
            _ = self.compose(**kwargs)

        return taylor_coefficients(self._evaluation, n_args=self.dfunctions[0].n_args, mult_prm=mult_prm, mult_drv=mult_drv)

    def merge(self, pattern=(), positions=None, **kwargs):
        '''
        Merge one or more sections in the current chain simultaneously.

        If a pattern may occur on several places, an
        additional parameter 'positions' has to be provided, so that
        all patterns are non-overlapping.
        
        Parameters
        ----------
        pattern: tuple, optional
            Tuple of integers which defines a subsequence in self.ordering.
            If nothings specified, the entire sequence will be used, and so
            this routine becomes very similar to self.compose (with the difference that
            a cderive object will be returned here).
            
        positions: list, optional
            List of integers which defines the start indices of the above pattern in self.ordering.
            If nothing specified, every occurence of 'pattern' in self.ordering will be used.
            
            In any case, only the members of 'pattern' are merged internally, so the number
            of occurences of that pattern should not affect the computational cost.
            
        **kwargs
            Optional keyworded arguments passed to the individual derive class(es).
            
        Returns
        -------
        cderive
            A cderive object which contains a sequence of 'derive' classes, representing a new chain
            of functions in which the selected pattern(s) have been merged.
        '''
        # Input handling and consistency checks
        #######################################
        assert type(pattern) == tuple
        if len(pattern) == 0:
            pattern = tuple(self.ordering)
        size = len(pattern)
        assert 2 <= size and size <= len(self)
        if positions is None:
            positions = _get_pattern_positions(self.ordering, pattern)
            if len(positions) == 0:
                raise RuntimeError('Pattern not found in the sequence.')
        # Sections must not overlap & can be found in the ordering. Evaluation(s) must exist.
        n_patterns = len(positions)
        assert 0 <= min(positions) and max(positions) < len(self) - size + 1, 'Pattern positions out of bounds.'
        for k in range(n_patterns - 1):
            assert positions[k + 1] - positions[k] >= size, 'Overlapping pattern.'
        assert all(tuple(self.ordering[pos: pos + size]) == pattern for pos in positions), 'Not all patterns found in sequence at requested positions.'
        assert all(hasattr(self.dfunctions[k], '_evaluation') for k in pattern), 'Merging requires function evaluations in advance.'
        self._merge_pattern = pattern
        self._merge_positions = positions

        # Merge the members of the pattern
        ##################################
        passage_indices = [[self.ordering[:pos + k].count(pattern[k]) for k in range(size)] for pos in positions] # The number of passages the functions already had in the chain, before the respective pattern(s).
        evr = [e[[passage_indices[k][0] for k in range(n_patterns)]] for e in self.dfunctions[pattern[0]]._evaluation] # The start values of the merged element consists of all start values at the various positions of the pattern
        for k in tqdm(range(1, size), disable=kwargs.get('disable_tqdm', True)):
            ev = [e[[passage_indices[j][k] for j in range(n_patterns)]] for e in self.dfunctions[pattern[k]]._evaluation] # (**)
            evr = general_faa_di_bruno(ev, evr, run_params=self.run_params)

        # Determine the new ordering & path
        ###################################
        # We also need to modify the evaluation results of the other elements in the chain (at (+) below).
        # This is required, because having merged some of its elements, some evaluations are not valid/required anymore.
        # For example, if we merge element Z=XYX in a chain of the form AXBXYX, then the element X originally occured 3 times. So any point will pass 3 times through that
        # chain. However, after merging, that element X will be present only once in the new lattice AXBZ, so only one entry remains.

        # Define a composition function representing the pattern.
        def pattern_function(*z, **pkwargs):
            for k in range(size):
                f = self.dfunctions[pattern[k]].jetfunc
                z = f(*z, **pkwargs)
            return z
        dp = derive(pattern_function, order=self.order, n_args=getattr(self.dfunctions[self.ordering[k]], 'n_args', kwargs.get('n_args', 0)))            
        dp._evaluation = evr # Note that 'evr' contains only those points which are passing through the pattern, which is guaranteed at step (**) above.
                
        pattern_positions = [r for pos in positions for r in range(pos, pos + size)]
        placeholder = -1
        ordering_w_placeholder = [self.ordering[j] if j not in pattern_positions else placeholder for j in range(len(self))]
        k = 0
        new_functions = []
        unique_function_indices = [] # to ensure that we pic only the unique functions in the chain
        while k < len(self):
            no = ordering_w_placeholder[k]
            if no == placeholder:
                if k == positions[0]: # first occurence of the pattern
                    new_functions.append(dp)
                    unique_function_indices.append(placeholder)
                k += size
            else:
                func_index = ordering_w_placeholder[k]
                
                if func_index in unique_function_indices:
                    k += 1
                    continue

                func = self.dfunctions[func_index]
                if func_index in pattern:
                    # (+) The function 'original_func' is in the pattern, but has not been merged (because it is not covered by a pattern within the sequence).
                    # In this case we have to remove those points in the function evaluation(s) which belong to passages through the (now merged) pattern(s).
                    func_path = self.path[func_index]
                    evaluation = [e[[j not in pattern_positions for j in func_path]] for e in func._evaluation]
                    new_func = func.__class__(func.jetfunc, n_args=func.n_args, order=func.order) # initiate (copy) the original function in order to not overwrite its "._evaluation" field.
                    new_func._evaluation = evaluation
                else:
                    new_func = func

                new_functions.append(new_func)
                unique_function_indices.append(func_index)
                k += 1
                
        # remove the chain of placeholders in the new_ordering & recalculate the ordering
        new_ordering = []
        j = 0
        while j < len(self):
            if ordering_w_placeholder[j] == placeholder:
                new_ordering.append(placeholder)
                j += size
            else:
                new_ordering.append(ordering_w_placeholder[j])
                j += 1
        new_ordering = _get_ordering(new_ordering)
        
        return self.__class__(*new_functions, order=self.order, ordering=new_ordering, run_params=self.run_params, reset=False)
    
    def __len__(self):
        return len(self.ordering)
    
    def __getitem__(self, key):
        '''
        If a list is provided, return an object of type(self).
        Otherwise, return the individual derive/cderive object.
        
        Attention: There is only support for strictly monotonous increasing lists
                   without gaps, as the individual jet-evaluations may become invalid
                   otherwise.
        '''
        if type(key) == list:
            requested_func_indices = [self.ordering[e] for e in key]
            requested_positions = key
            if len(key) > 1:
                # If the order has been changed, this may lead to wrong evaluations (they do not 'commute')
                # in this case we shall raise a warning. We shall check this here:
                if not all(i + 1 == j for i, j in zip(key, key[1:])):
                    raise RuntimeError('Requested slice needs to be strictly increasing without any gaps.')        
        else:
            requested_func_indices = self.ordering[key]
            requested_positions = range(len(self))[key]
            
        if type(requested_func_indices) != list:
            return self.dfunctions[requested_func_indices]
        else:
            if len(requested_func_indices) == 0:
                return []
            
            requested_unique_func_indices = list(np.unique(requested_func_indices))
            requested_funcs = [copy(self.dfunctions[e]) for e in requested_unique_func_indices]
            
            # Since we are going to return a cderive object, we remove any evaluation
            # points which belong to elements which are not present in the new chain
            for k in range(len(requested_funcs)):
                rf = requested_funcs[k]
                if not hasattr(rf, '_evaluation'):
                    continue
                findex = requested_unique_func_indices[k]
                func_path = self.path[findex]
                evaluation = [e[[j in requested_positions for j in func_path]] for e in rf._evaluation]
                rf._evaluation = evaluation # n.b. object was copied, so no danger to overwrite one of its fields
            
            new_ordering = [requested_unique_func_indices.index(e) for e in requested_func_indices] # starts from zero up to length of the unique (new) elements
            return self.__class__(*requested_funcs, ordering=new_ordering, order=self.order, run_params=self.run_params, reset=False)
            
    def __iter__(self):
        self._iterpointer = 0
        return self
            
    def __next__(self):
        if self._iterpointer < len(self):
            self._iterpointer += 1
            return self.dfunctions[self.ordering[self._iterpointer - 1]]
        else:
            raise StopIteration
            
    def index(self, value):
        '''
        Return the element number at a given position in the beamline chain (similar as .index for lists)
        '''
        return self.dfunctions.index(value)
    
    def _prepare_cycle(self, *point, periodic=False, warn=False, **kwargs):
        '''
        Internal routine to process user input, check if the current chain admits data 
        suitable for cycling and prepare a function to obtain the jet-evaluations depending on 
        the position in the chain.
                
        Parameters
        ----------
        *point: float or iterable, optional
            The point at which to evaluate the derivative(s). It has to be provided if the chain has
            not yet stored any jet-evaluation data.
        
        periodic: boolean, optional
            If True, assume that the final point equals the given point. In particular, the
            data taken in the calculation will be taken from the current jet-evaluations (if
            it exists).
            
        warn: boolean, optional
            Show warnings.
            
        **kwargs
            Optional keyworded arguments passed to the underlying jet-functions in case an evaluation
            is required. If they differ from self._call_kwargs (if it exists), a warning will be issued.
        
        Returns
        -------
        callable
            A function taking an integer (pos) and returning the jet-evaluation at this position. It will
            have a similar functionality than self.jev, with the difference that it will take values from
            0 to 2*len(self) instead.
        '''
        # Check if given input is stored in the database. If not, re-evaluate.
        eval_required, kwargs_changed = self._eval_memcheck(*point, **kwargs)
        if eval_required:
            if warn:
                warnings.warn('Input parameter(s) changed; re-evaluating ...')
            self.eval(*point, compose=False, **kwargs)

        # In the non-periodic case the evaluation data might have to be extended.
        # Check if end-point agrees with the start.
        inp = self.dfunctions[self.ordering[0]]._input
        point = [inp[k].array(0)[0] for k in range(len(inp))] # the index [0] means we are looking at the first occurence 
        outp = self.dfunctions[self.ordering[-1]]._evaluation
        final_point = [outp[k].array(0)[-1] for k in range(len(outp))] # the index [-1] means we are looking at the last occurence
        periodic = periodic or all([check_zero(point[k] - final_point[k]) for k in range(len(point))]) # Use existing data in case that periodic == False, but start and end point agree nonetheless.

        # Define the function which will provide the jet-evaluation data
        L = len(self)
        if periodic:
            self._cycle = self
            jev = lambda k: self.jev(k%L)
        else:
            # In this scenario, data exists but the point is not fixed. Evaluation has to be done.
            if warn:
                warnings.warn(f"Evaluation data exists, but periodic: {periodic}. Extending & evaluating at\n {final_point}\n ...")
            self._cycle = self.__class__(*[copy(f) for f in self.dfunctions], order=self.order, ordering=self.ordering, run_params=self.run_params, reset=True)
            self._cycle.eval(*final_point, compose=False, **kwargs) # the final point will be the start point of the extension
            def jev(k):
                if k < L:
                    return self.jev(k)
                else:
                    return self._cycle.jev(k%L)
        self._cycle_periodic = periodic
        return jev
    
    @staticmethod
    def _cycle_accumulator(data, L):
        '''
        Internal routine to accumulate data required for cycling.
        
        Code moved out of main routine to better work with the garbage collector.
        '''
        jev0 = data[0]
        cycling_data = [[_jetp1(tile(jev0[component_index], ncopies=1), index=component_index, ncopies=L - 1) for component_index in range(len(jev0))]]
        
        # successively add copies and identity transformations to the given jet-evaluations, defining the traces to be tracked.
        for k in range(1, L):
            jevk = data[k]
            cycling_data.append([_jetp1(tile(jevk[component_index], ncopies=k + 1), 
                                       index=component_index, 
                                       ncopies=L - k - 1) for component_index in range(len(jevk))])

        concat0 = [[jevk[ic].array(0) for ic in range(len(jevk))]]
        for k in range(L - 1):
            jevkpL = data[k + L]
            cycling_data.append([_jetp1(tile(jevkpL[component_index], ncopies=L - k - 1), 
                                        to_concat0=[c[component_index] for c in concat0],
                                        index=component_index, 
                                        ncopies=k + 1,
                                        location=0) for component_index in range(len(jevkpL))])
            concat0.append([jevkpL[ic].array(0) for ic in range(len(jevkpL))])
            
        return cycling_data
    
    def cycle(self, *args, outf='default', noreturn=False, **kwargs):
        r'''
        Cycle through the given chain: Compute the derivatives at each point, assuming
        a periodic structure of the entire chain.

        The basic idea goes as follows: Let c1 --> c2 --> c3 --> c4 --> c5 denote the current
        chain (in this example consisting of 5 elements). Then this routine will extend the internal
        jet-evaluation data to numpy arrays of length 5 by suitable clones and identity operators,
        so that we can compute (compose) in parallel the following chain of length 2*5 - 1:

        |c1 ---> c2 ---> c3 ---> c4 ---> c5*
        |        c2 ---> c3 ---> c4 ---> c5 ---> c1*
        |                c3 ---> c4 ---> c5 ---> c1 ---> c2*
        |                        c4 ---> c5 ---> c1 ---> c2 ---> c3*
        |                                c5 ---> c1 ---> c2 ---> c3 ---> c4*

        The final results will then be returned as the starred values in the above diagram.

        Parameters
        ----------
        *point: float or iterable, optional
            The point at which to evaluate the derivative(s). It has to be provided if the chain has
            not yet stored any jet-evaluation data.

        periodic: boolean, optional
            If true, assume that the final point equals the given point. In particular, the
            data taken in the calculation will be taken from the current jet-evaluations (if
            it exists).

        outf: str, optional
            Output format; if 'default', compose the extended jet-evaluations and return a list of jet-evaluations,
            representing the derivatives along the chain.
            Otherwise, return a cderive object of length len(self)*2 - 1 for further processing. The
            object can be composed to yield the jet-evaluations along the current chain of len(self) in form of numpy
            arrays. This second option is intended to be used for performance improvements.
            
        noreturn: boolean, optional
            If True, the routine will not return any object. One still has access to the results by
            the internal variable self._cycle_result. This option is intended to prevent a (possible) memory overflow
            problem which emerges at the 'return' statement: For some reason the private heap in Python gets approx. doubled 
            if the result(s) are returned after one repeated call of 'cycle'. This may become problematic for large data 
            sets and/or limited memory, where it is therefore recommended to run the routine with noreturn=True option.

        **kwargs
            Optional keyworded arguments passed to a (possible) chain evaluation.

        Returns
        -------
        list or cderive
            Depending on the 'outf' parameter the output is either
            1) A list of jet-evaluations, where the k-th entry corresponds to the derivative of the
            chain from position k to position k + L, where L denotes the length of the chain (using
            a periodic chain structure).
            2) A cderive class with the property mentioned above.
        '''
        if hasattr(self, '_cycle_result'):
            # clean up any previous result to prevent memory overflow, if called repeatedly
            del self._cycle_result
            
        L = len(self)
        self._cycle_jev = self._prepare_cycle(*args, **kwargs)
        data = [self._cycle_jev(k) for k in range(2*L - 1)]
        cycling_data = self._cycle_accumulator(data=data, L=L)
        del data
        
        if outf == 'default':
            # Compose the extended jet-evaluations and return the resulting list along the chain:
            compose_result = compose(*cycling_data, run_params=self.run_params)
            self._cycle_result = [[jcmp[k] for jcmp in compose_result[k + L - 1]] for k in range(L)]
            del cycling_data
            del compose_result
            gc.collect() # cleanup to prevent cluttering of memory
        else:
            # Construct a cderive class of length len(self)*2 - 1, having self._cycle_data_inp as jet-evaluations, and 
            # so that its compose routine will provide the same results as above.
            new_functions = [copy(f) for f in self.dfunctions] # shallow copy to prevent ._evaluation results to be overwritten in original in the code below
            ordering = (self.ordering*2)[:-1]
            for l in range(self.n_functions):
                n_args = new_functions[l].n_args
                jets_l = ([cycling_data[k][icmp] for k in range(len(ordering)) if ordering[k] == l] for icmp in range(n_args))
                new_functions[l]._evaluation = [_jbuild(jc) for jc in jets_l]
            self._cycle_result = self.__class__(*new_functions, ordering=ordering, order=self.order, run_params=self.run_params, reset=False)
            del cycling_data # without this, the memory demand will increase
            gc.collect() # cleanup to prevent cluttering of memory
            
        if noreturn == False:
            # If we return self._cycle_result, then we might get a memory doubling if we repeat this procedure; the amount of data is doubled at the first repetition -- and afterwards mildly keeps increasing. For the time being the only known option to prevent this is to let the results be attached to the current class.
            return self._cycle_result
    
def _get_ordering(sequence, start=0):
    '''
    For a given sequence of integers, find a unique ordering. E.g.
    ordering = [3, 4, 4, 7, 1]
    result = [0, 1, 1, 2, 3]
    
    Parameters
    ----------
    ordering: list
        The list of whose ordering we seek.
        
    start: int, optional
        An integer to control the start of the ordering.
        
    Returns
    -------
    list
        A list of integers describing the ordering of the given sequence.
    '''
    assert len(sequence) > 0
    new_ordering = []
    stash = []
    k = start
    for e in sequence:
        if e not in stash:
            new_ordering.append(k)
            k += 1
            stash.append(e)
        else:
            e_index = stash.index(e)
            new_ordering.append(e_index + start)
    return new_ordering

def _get_pattern_positions(sequence, pattern):
    '''
    For a given sequence and a given pattern, determine the positions at
    which the pattern occurs in the sequence -- non-overlapping -- starting
    from the first occurence in the sequence.
    
    Parameters
    ----------
    sequence: list
        A list of integers, representing the sequence.
        
    pattern: tuple
        A tuple of integers, representing the pattern.
    '''
    size = len(pattern)
    assert 1 <= size and size <= len(sequence)
    positions = []
    last_pos = -size
    k = 0
    for window in windowed(sequence, size):
        if window == pattern and last_pos + size <= k: # second condition ensures non-overlapping
            positions.append(k)
            last_pos = k
        k += 1
    return positions
