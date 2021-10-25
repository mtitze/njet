from .functions import exp, log
from .jet import jet as jet_source
from .jet import factorials
from .poly import jetpolynom


class jet(jet_source):
    '''
    Class to store and operate with higher-order derivatives of a given function.
    '''
    def __pow__(self, other):
        other = self.convert(other)

        if other.order == 0:
            result = jet_source.__pow__(self, other)
        else:
            '''
            General exponentiation, using exp and log functions.
            '''
            result = exp(log(self)*other)
            result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
    
class derive:
    '''
    Class to handle the derivatives of a (jet-)function (i.e. a function consisting of a composition
    of elementary functions).
    '''
    
    def __init__(self, func, order: int=1, **kwargs):
        self.func = func
        self.n_args = self.func.__code__.co_argcount # the number of any arguments of func (before *args)
        self.set_order(order)
        self._Df = {}
        
    def set_order(self, order):
        self.order = order
        self._factorials = factorials(self.order)
        
    def eval(self, z, mult=True):
        '''
        Evaluate the derivatives of a (jet-)function at a specific point up to self.order.

        Parameters
        ----------
        z: subscriptable 
            Vector at which the function and its derivatives should be evaluated.
        mult: Boolean, optional
            Whether or not to include the multiplicities C(j1, ..., jm). Default: True. (Details see below).
            If False, then the C*Df's are returned. If True, then the Df's are returned.

        Returns
        -------
        dict
            Dictionary of compontens of the multivariate Taylor expansion of the given function self.func:

            Let m be the number of arguments of self.func. 
            Then
            self.func(z1, ..., zm) = sum_{j1 + ... + jm = k} C(j1, ..., jm) * Df[j1, ... jm] * z1**j1 * ... * zm**jm
            with
            Df[j1, ..., jm] := \partial^j1/\partial_{z1}^j1 ... \partial^jm/\partial_{zm}^jm f (z1, ..., zm)
            and combinatorial factor
            C(j1, ..., jm) = (j1 + ... + jm)!/(j1! * ... * jm!) .
        '''
        # perform the computation, based on the input vector
        inp = []
        for k in range(self.n_args):
            jk = jet([z[k], jetpolynom(1, index=k, power=1)], n=self.order)
            inp.append(jk)
        evaluation = self.func(*inp)
        # extract Df from the result
        Df = {}
        for k in range(1, evaluation.order + 1): # the k-th derivative
            entry = evaluation.array(k)
            if not entry.__class__.__name__ == 'jetpolynom': # skip any non-polynomial entry
                continue
            for key, value in entry.values.items(): # loop over the individual polynomials of the k-th derivative
                # key corresponds to a specific frozenset, i.e. some indices and powers of a specific monomial.
                indices = [0]*self.n_args
                multiplicity = 1
                for index, power in key:
                    if power == 0: # the (k, 0)-entries correspond to the scalar 1 and will be ignored here. TODO: may need to improve this in jetpolynom class.
                        continue
                    indices[index] = power
                    multiplicity *= self._factorials[power]
                if mult:
                    value *= multiplicity/self._factorials[sum(indices)]
                Df[tuple(indices)] = value
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
            specified, the most recent output will be used.
            
        Returns
        -------
        dict
            The components of the k-th derivative of self.func.
        '''
        assert k <= self.order
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
    
    def grad(self, **kwargs):
        '''
        Returns the gradient of the function.
        
        Parameters
        ----------
        kwargs: dict
            Additional arguments passed to self.build_tensor
            
        Returns
        -------
        dict
            Dictionary containing the components of the gradient.
        '''
        return self.build_tensor(k=1, **kwargs)
    
    def hess(self, **kwargs):
        '''
        Returns the Hessian of the function.
        
        Parameters
        ----------
        kwargs: dict
            Additional arguments passed to self.build_tensor
            
        Returns
        -------
        dict
            Dictionary containing the components of the Hessian.
        '''
        return self.build_tensor(k=2, **kwargs)
        
        
        
