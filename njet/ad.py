from .functions import exp, log, zero
from .jet import jet as _jet
from .jet import jetpoly
from .common import check_zero, convert_indices

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
    
def getNargs(func):
    '''
    Determine the number of arguments of a function. Raise an error if they can not be determined.
    
    Parameters
    ----------
    func: callable
        The function to be examined.
        
    Returns
    -------
    int
        The number of arguments of the given function.
    '''
    error_msg = 'The number of function arguments could not be determined.'
    n_args = 0
    try:
        n_args = func.__code__.co_argcount
    except:
        raise RuntimeError(error_msg)
    assert n_args > 0, error_msg
    return n_args

def get_taylor_coefficients(evaluation, output_format=0, **kwargs):
    '''
    Return the Taylor coefficients of a jet evaluation.
    
    Parameters
    ----------
    output_format: int, optional
        If 0 and the output would be a list of length 1, return the entry 
        of this list instead. If != 0, disable this behavior.
    
    **kwargs
        Parameters passed to njet.jet.get_taylor_coefficients routine.
        
    Returns
    -------
    dict
        One or more dictionaries, representing the Taylor-coefficients of the given
        evaluation (see njet.poly.jetpoly.get_taylor_coefficients for details).
    '''
    try:
        out = evaluation.get_taylor_coefficients(**kwargs)
        if output_format != 0:
            out = [out]
    except:
        out = (*[ev.get_taylor_coefficients(**kwargs) for ev in evaluation],) # also stored in ev._tc
        if len(out) == 1 and output_format == 0:
            out = out[0]
    return out

class derive:
    '''
    Class to handle the derivatives of a (jet-)function (i.e. a function consisting of a composition
    of elementary functions).
    
    Parameters
    ----------
    func: callable(s)
        The function(s) to be derived. Must be expressed in terms of functions (e.g. polynomials) supported by njet.functions. Note that the first function in the given chain will be executed first.

    order: int, optional
        The order up to which the function should be derived.

    n_args: int, optional
        The number of arguments on which func depends on; passed to getNargs routine.
        
    n_out: int, optional
        Define the number of output parameters of the function(s). This is required for multi-dimensional
        output.
        
    truncate: int, optional
        If given, truncate the jets after each iteration through the given functions.
    '''
    def __init__(self, func, order: int=1, n_args: int=0):
        self.order = order
        self.jetfunc = func
        
        # Determine the number of input parameters
        if n_args == 0:
            self.n_args = getNargs(func)
        else:
            self.n_args = n_args

    def jet_input(self, *z):
        inp = []
        for k in range(self.n_args):
            jk = jet(z[k], jetpoly(zero(z[k]) + 1, index=k, power=1), n=self.order) # add a zero here to produce the same shape or type as z[k]
            inp.append(jk)
        self._input = inp
        return inp
        
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
        self._evaluation = self.jetfunc(*self.jet_input(*z), **kwargs)
        return self._evaluation
            
    def __call__(self, *z, **kwargs):
        '''Evaluate the derivatives of a (jet-)function at a specific point up to self.order.
        
        Parameters
        ----------
        z: subscriptable
            List of values at which the function and its derivatives should be evaluated.
            
        **kwargs:
            Optional arguments passed to self._evaluation.get_taylor_coefficients routine.

        Returns
        -------
        dict
            Dictionary of compontens of the multivariate Taylor expansion of the given function self.jetfunc
        '''
        # These two keywords are reserved for the get_taylor_coefficients routine and will be removed from the input:
        mult_prm = kwargs.pop('mult_prm', True)
        mult_drv = kwargs.pop('mult_drv', True)
        # perform the computation, based on the input vector
        return get_taylor_coefficients(self.eval(*z, **kwargs), n_args=self.n_args, mult_prm=mult_prm, mult_drv=mult_drv)
        
    def build_tensor(self, k: int, **kwargs):
        '''
        Convert the components of the k-th derivative into the entries of a (self.n_args)**k tensor.
        See njet.jet.build_tensor for details.
        '''
        assert k <= self.order, f'Order ({self.order}) insufficient for requested number ({k}) of derivatives.'
        assert hasattr(self, '_evaluation'), 'Derivative(s) need to be evaluated first. Try passing a point.'
        return self._evaluation.build_tensor(k=k, **kwargs)
    
    def grad(self, *z, **kwargs):
        '''
        Returns the gradient of the current function.
        See njet.jet.grad for details.
        '''
        if len(z) > 0:
            _ = self(*z, **kwargs)
        return self.build_tensor(k=1, **kwargs)
    
    def hess(self, *z, **kwargs):
        '''
        Returns the Hessian of the function.
        See njet.jet.hess for details.
        '''
        if len(z) > 0:
            _ = self(*z, **kwargs)
        return self.build_tensor(k=2, **kwargs)
        