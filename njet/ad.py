from .functions import exp, log
from .jet import jet as _jet
from .jet import factorials, check_zero, jetpoly
from .common import convert_indices

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
        
        For details see njet.jet.get_taylor_coefficients routine.
        '''
        assert isinstance(ev, jet), f"Object of type 'jet' expected. Input of type '{ev.__class__.__name__}'. Note that only single-valued functions are supported."
        
        return ev.get_taylor_coefficients(n_args=self.n_args, facts=self._factorials, **kwargs)
        
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
        converted_indices = convert_indices(list_of_tuples)
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
        