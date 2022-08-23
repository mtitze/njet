from .functions import exp, log
from .jet import jet as _jet
from .jet import factorials, check_zero, jetpoly

class jet(_jet):
    '''
    Class to store and operate with higher-order derivatives of a given function.
    '''
    def __pow__(self, other):
        other = self.convert(other)

        if other.order == 0:
            result = _jet.__pow__(self, other)
        else:
            '''
            General exponentiation, using exp and log functions.
            '''
            result = exp(log(self)*other)
            result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
def getNargs(f, **kwargs):
    '''
    Determine the number of arguments of a function. Raise an error if they can not be determined.
    
    Parameters
    ----------
    f: callable
        The function to be examined.
        
    Returns
    -------
    int
        The number of arguments of the given function.
    '''
    error_msg = 'The number of function arguments could not be determined. Try passing n_args parameter.'
    n_args = kwargs.get('n_args', 0)
    if n_args == 0:
        try:
            n_args = f.__code__.co_argcount
        except:
            raise RuntimeError(error_msg)
    assert n_args > 0, error_msg
    return n_args    
    
class derive:
    '''
    Class to handle the derivatives of a (jet-)function (i.e. a function consisting of a composition
    of elementary functions).
    
    Parameters
    ----------
    func: callable
        The function to be derived. Must be expressed in terms of polynomials and functions supported by njet.functions.

    order: int, optional
        The order up to which the function should be derived.

    n_args: int, optional
        The number of arguments on which func depends on; passed to getNargs routine.
    '''
    def __init__(self, func, order: int=1, **kwargs):
        self.func = func
        self.n_args = getNargs(func, **kwargs)
        #self.func, self.n_args = standardize_function(func, n_args=n_args)
        # Now in all cases self.func takes only a single subscriptable object as argument.
        self.set_order(order)
        self._Df = {}
        
    def set_order(self, order):
        self.order = order
        self._factorials = factorials(self.order)
        
    def eval(self, *z, **kwargs):
        '''
        Pass a jet of order self.order, having polynomials in its higher-order entries,
        through the given function.
        
        Parameters
        ----------
        z: subscriptable
            List of values at which the function and its derivatives should be evaluated.
            
        **kwargs:
            Optional keyword arguments passed to the underlying function.
                    
        Returns
        -------
        jet
            Jet containing the value of the function in its zero-th entry and the
            jet-polynomials in the higher-order entries.
        '''
        inp = []
        for k in range(self.n_args):
            jk = jet([z[k], jetpoly(1, index=k, power=1)], n=self.order)
            inp.append(jk)
        return self.func(*inp, **kwargs)
    
    def get_taylor_coefficients(self, ev, **kwargs):
        '''Extract the Taylor coefficients from a given jet-evaluation (the output of self.eval).
        
        Let m be the number of arguments of self.func. Then the k-th derivative of self.func has the form
        (D^k self.func)(z1, ..., zm) = sum_{j1 + ... + jm = k} Df[j1, ... jm]/(j1! * ... * jm!) * z1**j1 * ... * zm**jm
        = sum_{S(j1, ... jm), j1 + ... + jm = k} C(j1, ..., jm) * Df[j1, ..., jm] * z1**j1 * ... * zm**jm.
        with S(j1, ..., jm) an index denoting the set having the elements {j1, ..., jm},
        Df[j1, ..., jm] := \\partial^j1/\\partial_{z1}^j1 ... \\partial^jm/\\partial_{zm}^jm f (z1, ..., zm)
        and combinatorial factor
        C(j1, ..., jm) = (j1 + ... + jm)!/(j1! * ... * jm!) .
        
        Parameters
        ----------
        ev: jet
            A jet having jetpoly entries in the k-th order entries for k > 0.
            
        **kwargs
            Optional arguments passed to poly.get_taylor_coefficients routine.
            
            Note that one can control how to deal with multiplicities C(j1, ..., jm) (notation see above) by
            passing mult_drv and mult_prm attributes to this routine.
            
        Returns
        -------
        dict
            Dictionary which maps the tuples representing the indices and powers of the individual
            monomials to their coefficients, corresponding to the Taylor expansion of the given expression.
        '''
        assert isinstance(ev, jet), f"Object of type 'jet' expected. Input of type '{ev.__class__.__name__}'. Note that only single-valued functions are supported."
        
        Df = {}
        # add the constant (if non-zero):
        const = ev[0]
        if not check_zero(const):
            Df[(0,)*self.n_args] = const
        
        for entry in ev[1:]: # iteration over the derivatives of order >= 1.
            if not isinstance(entry, jetpoly): # skip any non-polynomial entry.
                continue
            Df.update(entry.get_taylor_coefficients(n_args=self.n_args, facts=self._factorials, **kwargs))
            # Since we loop over derivatives of a specific order, it is ensured that these Taylor coefficients are always different, 
            # so the above procedure does not overwrite existing keys.
            
        return Df
        
    def __call__(self, *z, **kwargs):
        '''Evaluate the derivatives of a (jet-)function at a specific point up to self.order.
        
        Parameters
        ----------
        z: subscriptable
            List of values at which the function and its derivatives should be evaluated.
            
        **kwargs:
            Optional arguments passed to self.get_taylor_coefficients routine.

        Returns
        -------
        dict
            Dictionary of compontens of the multivariate Taylor expansion of the given function self.func
        '''
        # perform the computation, based on the input vector
        Df = self.get_taylor_coefficients(self.eval(*z), **kwargs)
        self._Df = Df
        return Df
    
    def extract(self, k: int, **kwargs):
        '''
        Extract the components of the k-th derivative from the output of self.eval. 
        
        Parameters
        ----------
        k: int
            The order of the derivatives to extract.
        Df: dict, optional
            The output of self.eval, containing the entries of the derivative. If nothing
            specified, the last evaluation (stored in self._Df) will be used.
            
        Returns
        -------
        dict
            The components of the k-th derivative of self.func.
        '''
        assert k <= self.order
        if len(self._Df) == 0 and 'Df' not in kwargs.keys():
            raise RuntimeError('Derivative(s) need to be evaluated first.')
        D = kwargs.get('Df', self._Df)
        return {j: D[j] for j in D.keys() if sum(j) == k}
    
    @staticmethod
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
    
    def build_tensor(self, k: int, **kwargs):
        '''
        Convert the components of the k-th derivative into the entries of a (self.n_args)**k tensor.
        See also self.convert_indices for details.
        
        Parameters
        ----------
        k: int
            The degree of the derivatives to be considered.
        
        Returns
        -------
        dict
            Dictionary representing non-zero components of the tensor which describes the multivariate 
            Taylor map of order k.
            Only values unequal to zero will be stored, and only one member for each set of entries
            which can be obtained by suitable permutations of the indices.
        '''
        entries = self.extract(k=k, **kwargs)
        list_of_tuples = list(entries.keys())
        converted_indices = self.convert_indices(list_of_tuples)
        return {converted_indices[k]: entries[list_of_tuples[k]] for k in range(len(list_of_tuples))}
    
    def grad(self, *z, **kwargs):
        '''
        Returns the gradient of the function.
        
        Parameters
        ----------
        kwargs: dict
            Additional arguments passed to self.build_tensor
            
        z: subscriptable, optional
            Point at which to compute the gradient. If nothing specified (default),
            then the gradient is determined from the last evaluation.
            
        Returns
        -------
        dict
            Dictionary containing the components of the gradient.
        '''
        if len(z) > 0:
            _ = self(*z, **kwargs)
        return self.build_tensor(k=1, **kwargs)
    
    def hess(self, *z, **kwargs):
        '''
        Returns the Hessian of the function.
        
        Parameters
        ----------
        kwargs: dict
            Additional arguments passed to self.build_tensor
        
        z: subscriptable, optional
            Point at which to compute the Hessian. If nothing specified (default),
            then the Hessian is determined from the last evaluation.
            
        Returns
        -------
        dict
            Dictionary containing the components of the Hessian.
        '''
        if len(z) > 0:
            _ = self(*z, **kwargs)
        return self.build_tensor(k=2, **kwargs)
        