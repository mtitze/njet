import numpy as np
from more_itertools import distinct_permutations, windowed
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
        
    run_params: tuple, optional
        A tuple containing the output of njet.extras._make_run_params. These values
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


class cderive:
    '''
    Derive a chain of vector-valued functions with repetitions. 
    The given functions should be unique, while their repetition in the chain
    will be given by an optional ordering. This might have better
    performance than deriving the entire chain of functions
    with the default 'derive' method.
    '''
    def __init__(self, functions, order: int=1, ordering=None, run_params=(), **kwargs):
        '''
        Parameters
        ----------
        functions: list
            A list of callable(s) or 'derive' classes related to the
            unique vector-valued*) functions in the chain to be derived.
            
            Alternatively, one can pass a list of 'derive' objects, containing
            vector-valued jet evaluation output. 
            This second case can be used to avoid re-calculating the derivatives
            in circumstances where it is not required.
            
            *) This means that the functions must return iterables in any case.
            
        order: int
            The maximal order of the derivatives to be computed.
            
        ordering: list
            The order defining how the unique functions are arranged in the chain.
            Hereby the index j must refer to the function at position j in the list.
            
        run_params: tuple, optional
            A tuple containing the output of njet.extras._make_run_params. These values
            can be send to the routine to avoid internal re-calculation when the routine
            it is called repeatedly.
            
        **kwargs
            Optional keyworded arguments passed to njet.ad.derive init.
        '''

        if ordering is None:
            ordering = list(range(len(functions)))
            
        # Check input consistency
        uord = np.unique(ordering) # np.unique already sorting
        assert len(uord) == len(functions), 'Number of functions not consistent with the unique items in the ordering.'
        assert (uord - np.arange(len(functions)) == 0).all(), 'Ordering malformed.'

        # Determine user input for the 'functions' parameter
        supported_objects = ['njet.ad.derive', 'njet.extras.cderive'] # objects of these kinds will not be instantiated with 'derive'
        self.dfunctions = [f if any([s in f.__repr__() for s in supported_objects]) else derive(f, order=order, **kwargs) for f in functions]
        
        self.n_functions = len(functions)
        self.ordering = ordering
        self.chain_length = len(ordering)
        self.order = order
        
        if len(run_params) > 0:
            self.factorials, self.run_indices = run_params
        else:
            self.factorials, self.run_indices = _make_run_params(order)

        # For every element in the chain, note a number at which a point
        # passing through the given chain will pass through the element.
        # So for example, self.path[k] = [j1, j2, j3, ...] means that
        # the first passage through element k occurs at global position j1,
        # the second passage through element k occurs at global position j2 etc.
        self.path = {k: [] for k in range(self.n_functions)}
        for j in range(self.chain_length):
            self.path[self.ordering[j]].append(j)
            
    def jetfunc(self, *point, **kwargs):
        '''
        Run a point through the chain once, to determine the point(s) at which
        the derivative(s) should be calculated.
        '''
        self._input = point
        out = [point]
        for k in range(self.chain_length):
            point = self.dfunctions[self.ordering[k]].jetfunc(*point, **kwargs)
            out.append(point)
        self._output = out
        return point
    
    def _probe(self, *point, **kwargs):
        '''
        Check if the points in the current output are in agreement with the one stored in the input functions.
        This check requires that the input functions have been evaluated and a probe run has been performed.
        '''
        if not all([hasattr(df, '_input') for df in self.dfunctions]):
            # Nothing can be compared, so the check will fail
            return False
        elif not hasattr(self, '_output'): 
            # Probe the current chain
            _ = self.jetfunc(*point, **kwargs)
            
        if not np.array([point[k] == self.dfunctions[self.ordering[0]]._input[k][0].array(0) for k in range(len(point))]).all():
            # If the input point disagrees with the stored input point, then return false.
            return False
        else:
            # Check the remaining points along the chain
            return all([np.array([self.dfunctions[self.ordering[k]]._input[component_index][self.path[self.ordering[k]].index(k)].array(0) == self._output[k][component_index] for component_index in range(len(self._output[k]))]).all() for k in range(1, self.chain_length)])
            
    def eval(self, *point, **kwargs):
        '''
        Evaluate the individual (unique) functions in the chain at the requested point.
        '''
        _ = self.jetfunc(*point, **kwargs)
        points_at_functions = [[self._output[l] for l in range(self.chain_length) if self.ordering[l] == k] for k in range(self.n_functions)]
        # let Q = points_per_function[j], so Q is a list of points which needs to be computed for function j
        # Then the first element in Q is the one which needs to be applied first, etc. (for element j) by this construction.
        for k in tqdm(range(self.n_functions), disable=kwargs.get('disable_tqdm', False)):
            function = self.dfunctions[k].jetfunc
            n_args_function = self.dfunctions[k].n_args
            points_at_function = points_at_functions[k]
            components = [np.array([points_at_function[j][l] for j in range(len(points_at_function))], dtype=np.complex128) for l in range(n_args_function)]
            _ = self.dfunctions[k].eval(*components, **kwargs)
        return self.compose(**kwargs)
    
    def compose(self, **kwargs):
        '''
        Compose the given derivatives for the entire chain.      
        '''
        assert all(hasattr(f, '_evaluation') for f in self.dfunctions), 'Composition requires function evaluations in advance.'
        evr = [e[0] for e in self.dfunctions[self.ordering[0]]._evaluation] # the start is the derivative of the first element at the point of interest
        self._evaluation_chain = [evr]
        for k in tqdm(range(1, self.chain_length), disable=kwargs.get('disable_tqdm', False)):
            func_index = self.ordering[k]
            index_passage = self.path[func_index].index(k)
            ev = [j[index_passage] for j in self.dfunctions[func_index]._evaluation]
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
            a cderive object will be returned here, instead).
            
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
        assert 2 <= size and size <= self.chain_length
        if positions is None:
            positions = []
            last_pos = -size
            k = 0
            for window in windowed(self.ordering, size):
                if window == pattern and last_pos + size <= k: # second condition ensures non-overlapping
                    positions.append(k)
                    last_pos = k
                k += 1
            if len(positions) == 0:
                raise RuntimeError('Pattern not found in sequence.')
        # Sections must not overlap & can be found in the ordering. Evaluation(s) must exist.
        n_patterns = len(positions)
        assert 0 <= min(positions) and max(positions) < self.chain_length - size + 1, 'Pattern positions out of bounds.'
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
        for k in tqdm(range(1, size), disable=kwargs.get('disable_tqdm', False)):
            ev = [e[[passage_indices[j][k] for j in range(n_patterns)]] for e in self.dfunctions[pattern[k]]._evaluation] # (**)
            evr = general_faa_di_bruno(ev, evr, run_params=(self.factorials, self.run_indices))

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
        ordering_w_placeholder = [self.ordering[j] if j not in pattern_positions else placeholder for j in range(self.chain_length)]
        k = 0
        new_functions = []
        unique_function_indices = [] # to ensure that we pic only the unique functions in the chain
        while k < self.chain_length:
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
        while j < self.chain_length:
            if ordering_w_placeholder[j] == placeholder:
                new_ordering.append(placeholder)
                j += size
            else:
                new_ordering.append(ordering_w_placeholder[j])
                j += 1
        new_ordering = _get_ordering(new_ordering)
        
        return self.__class__(functions=new_functions, order=self.order, ordering=new_ordering, run_params=(self.factorials, self.run_indices))
    
    
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