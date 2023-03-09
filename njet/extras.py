import numpy as np
from more_itertools import distinct_permutations, windowed
from tqdm import tqdm
from copy import copy
import warnings

from njet import jet, jetpoly, derive, get_taylor_coefficients
from njet.common import check_zero
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
            jfk = f[k]
            if jfk.array(r) == 0: 
                # skip in case that the r-th derivative of the k-th component of jfk does not exist
                continue
            out[order][k] += symtensor_call([[jg.array(nj)/facts[nj] for jg in g] for nj in e], jfk.array(r).terms)/facts[r]*facts[order]
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

def tile(jev, ncopies: int):
    '''
    Construct a new jet in which its entries contain copies of itself.
    
    Note: Intention to work if all jet entries are numpy arrays. A jet with
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
    a0 = jev.array(0)
    a0_new = np.tile(a0, [ncopies] + [1]*len(a0.shape))
    new_jet_array = [a0_new]
    for k in range(1, jev.order + 1):
        entry_k = jev.array(k).terms
        new_terms_k = {}
        for key, value in entry_k.items():
            new_terms_k[key] = np.tile(value, [ncopies] + [1]*len(value.shape))
        new_jet_array.append(jetpoly(terms=new_terms_k))
    return jet(*new_jet_array)

def _jetp1(jev, index: int, idvalue=None, ncopies: int=1, location=-1):
    '''
    Add copies of a unity transformations to the give jet entries -- either
    in the end or at the beginning.
    
    Intended to work if all jet entries are numpy arrays.

    Parameters
    ----------
    jev: jet
    
    index: int
        The component of the unity transformation.
        
    idvalue: float, optional
        The value of the identity. If nothing specified, the last value of jetv.array(0) will be used.
        
    ncopies: int, optional
        The number of copies attached.
        
    location: int, optional
        -1: Attach to end. Everything else: attach to start.
    
    Returns
    -------
    jet
    '''
    a0 = jev.array(0)
    zero = a0*0
    one = zero + 1
    kj = frozenset({(index, 1)})
    if idvalue is None:
        to_concat0 = [a0[-1]]*ncopies
    else:
        to_concat0 = [idvalue]*ncopies
        
    # the unity transformation will reproduce the value of the jet in the 0-array.
    if location == -1:
        #a0_new = np.r_[a0, a0[-1]]
        a0_new = np.concatenate([a0, to_concat0])
    else:
        a0_new = np.concatenate([to_concat0, a0])
    
    new_jet_array = [a0_new]
    for k in range(1, jev.order + 1):
        entry_k = jev.array(k).terms
        if k == 1:
            # ensure that kj appears in the new jetpoly in any case:
            _ = entry_k.setdefault(kj, zero)
        new_terms_k = {}
        for key, value in entry_k.items():
            if key == kj: # may happen if k = 1
                to_concatj = [one[0]]*ncopies
            else:
                to_concatj = [zero[0]]*ncopies
                
            if location == -1:
                #new_terms_k[key] = np.r_[value, 1]
                new_terms_k[key] = np.concatenate([value, to_concatj])
            else:
                #new_terms_k[key] = np.r_[value, 0]
                new_terms_k[key] = np.concatenate([to_concatj, value])
        new_jet_array.append(jetpoly(terms=new_terms_k))
    return jet(*new_jet_array)


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

    **kwargs
        Optional keyworded arguments passed to njet.ad.derive init.
    '''

    def __init__(self, *functions, order: int=1, ordering=None, run_params={}, **kwargs):

        # Determine user input for the 'functions' parameter
        supported_objects = ['njet.ad.derive', 'njet.extras.cderive'] # objects of these kinds will not be instantiated with 'derive'
        self.dfunctions = [copy(f) if any([s in f.__repr__() for s in supported_objects]) else derive(copy(f), order=order, **kwargs) for f in functions]
        
        self.n_functions = len(self.dfunctions)
        self.order = order
        
        if len(run_params) == 0:
            run_params = _make_run_params(order)
        self.run_params = run_params
        
        self.set_ordering(ordering=ordering, **kwargs)
                
    def set_ordering(self, ordering=None, reset=True, **kwargs):
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
        Run a point through the chain once, to determine the point(s) at which
        the derivative(s) should be calculated.
        
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
        if not all([hasattr(df, '_input') for df in self.dfunctions]):
            # Nothing can be compared, so the check will fail
            return False
        elif not hasattr(self, '_output'): 
            # Probe the current chain
            _ = self.jetfunc(*point, **kwargs)
            
        # Check the points along the chain
        try:
            return all([np.array([self.dfunctions[self.ordering[k]]._input[component_index][self.path[self.ordering[k]].index(k)].array(0) == self._output[k][component_index] for component_index in range(len(self._output[k]))]).all() for k in range(len(self))])
        except:
            warnings.warn('Probe check aborted with error(s).')
            return False
            
    def eval(self, *point, **kwargs):
        '''
        Evaluate the individual (unique) functions in the chain at the requested point.
        
        Parameters
        ----------
        *point: single value or array-like
            The point at which the evaluation should be performed.
            
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
        return self.compose(**kwargs)
    
    def jev(self, pos: int):
        '''
        Convenience function to obtain jet-evaluation data at a specific position (This
        data can be produced by the 'eval' command).
        
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
        assert all(hasattr(f, '_evaluation') for f in self.dfunctions), 'Composition requires function evaluations in advance.'
        # the path through the chain will require the selection of the point through which the trajectory passes at the respective position:
        evals = [[e[self.path[self.ordering[pos]].index(pos)] for e in self[pos]._evaluation] for pos in range(len(self))]
        self._evaluation_chain = compose(*evals, run_params=self.run_params, **kwargs)
        self._evaluation = self._evaluation_chain[-1]
        return self._evaluation
    
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
        # These two keywords are reserved for the get_taylor_coefficients routine and will be removed from the input:
        mult_prm = kwargs.pop('mult_prm', True)
        mult_drv = kwargs.pop('mult_drv', True)
        
        # Determine if the keyworded arguments have been changed
        kwargs_changed = True
        if hasattr(self, '_call_kwargs'):
            kwargs_changed = not all(self._call_kwargs.get(key, None) == val for key, val in kwargs.items())
        if kwargs_changed or not hasattr(self, '_call_kwargs'):
            self._call_kwargs = kwargs
                        
        # Determine if a (re-)evaluation is required
        eval_required = False
        if len(z) > 0:
            if kwargs_changed:
                eval_required = True
            if not all([hasattr(df, '_evaluation') for df in self.dfunctions]):
                eval_required = True
            elif not self._probe(*z, **kwargs):
                eval_required = True
        
        # Perform the composition, if necessary
        if eval_required:
            _ = self.eval(*z, **kwargs) # evaluation includes 'self.compose'
        elif not hasattr(self, '_evaluation') or kwargs_changed:
            _ = self.compose(**kwargs)

        return get_taylor_coefficients(self._evaluation, n_args=self.dfunctions[0].n_args, mult_prm=mult_prm, mult_drv=mult_drv)

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
        assert all(tuple(self.ordering[pos: pos + size]) == pattern for pos in positions), 'Not all patterns found in sequence.'
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
    
    def cycle(self, *point, periodic='auto', **kwargs):
        '''
        Cycle through the given chain: Compute the derivatives at each point, assuming
        a periodic structure of the entire chain.
        
        periodic: str or boolean, optional
            If True, assume that the start point through the chain agrees with the end point.
            If False, a check will be made for this condition. If the check fails, an extension
            of the current chain of twice its length will be considered to calculate missing
            derivatives.
        '''
        # May have to not calculate any evaluation in advance ... (because the point at
        # length len(dchain) may not equal the one at start)
        if len(point) == 0:
            if hasattr(self, '_input'):
                point = self._input
        assert len(point) > 0, 'Reference point required.'

        # Check if situation is periodic
        L = len(self)
        if periodic == 'auto':
            if not hasattr(self, '_output'):
                warnings.warn(f"periodic: {periodic} and no '_output' field found. Evaluating at {point} ...")
                _ = self.__call__(*point, **kwargs) # TODO: maybe improve the behaviour here (load eval data)
            periodic = all([check_zero(point[k] - self._output[-1][k]) for k in range(len(point))])
            
        if not periodic:
            # Construct a chain twice as long as the current chain
            # TODO: only length L chain required: just continue with end point here.
            self._cycle = self.__class__(*[copy(f) for f in self.dfunctions], order=self.order, ordering=self.ordering*2, run_params=self.run_params)
            _ = self._cycle.eval(*point, **kwargs)
            jev = lambda k: self._cycle.jev(k)
        else:
            self._cycle = self
            jev = lambda k: self.jev(k%L)
        
        jev0 = jev(0)
        cycling_data = [[_jetp1(tile(jev0[component_index], ncopies=1), index=component_index, ncopies=L - 1) for component_index in range(len(jev0))]]
        # successively add copies and identity transformations to the given jet-evaluations, defining the traces to be tracked.
        for k in range(1, L):
            jevk = jev(k)
            cycling_data.append([_jetp1(tile(jevk[component_index], ncopies=k), 
                                       index=component_index, 
                                       ncopies=L - k) for component_index in range(len(jevk))])
        for k in range(L - 1):
            jevkpL = jev(k + L)
            cycling_data.append([_jetp1(tile(jevkpL[component_index], ncopies=L - k), 
                                       index=component_index, 
                                       ncopies=k, location=0) for component_index in range(len(jevkpL))])
            
        self._cycle_data_inp = cycling_data
        self._cycle_data = compose(*cycling_data, run_params=self.run_params)
        return [[jcmp[k] for jcmp in self._cycle_data[k + L - 1]] for k in range(L)]
    
    
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
        
    pattern: tuple or list
        A tuple or list of integers, representing the pattern.
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