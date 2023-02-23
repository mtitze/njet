from .common import check_zero, factorials, nCr, convert_indices
from .poly import jetpoly

def sum_by_jetpoly(ls):
    # sum over the members of a given list so that each member e of jetpoly class appears on the
    # left. This can be used to prevent that in case of numpy the new values are numpy.arrays.
    return sum([e for e in ls if isinstance(e, jetpoly)]) + sum([e for e in ls if not isinstance(e, jetpoly)])

def general_leibniz_rule(f1, f2):
    '''
    Compute the higher-order derivatives of the product of two functions f1*f2.
    In the following denote by f^k the k-th derivative of f, evaluated at a specific point.
    It is assumed that len(f1) = len(f2) =: n.
    
    Parameters
    ----------
    f1: list
        List containing the values f1^k for k = 0, ..., n - 1.
    f2: list
        List containing the values f2^k for k = 0, ..., n - 1.
        
    Returns
    -------
    list
        List containing the values (f1*f2)^k for k = 0, ..., n - 1.
    '''
    nmax = len(f1) - 1 # len(f1): max number of summands
    nok = nCr(nmax)
    return [sum_by_jetpoly([nok[n][k]*f1[n-k]*f2[k] if isinstance(f1[n-k], jetpoly) else nok[n][k]*f2[k]*f1[n-k] for k in range(n + 1)]) for n in range(nmax + 1)]

def faa_di_bruno(f, g):
    '''
    Consider the composition f o g of two functions and for h in {f, g} denote by h^k its k-th derivative. Then
    (f o g)^1 = (f^1 o g) g^1
    (f o g)^2 = (f^2 o g) (g^1)**2 + (f^1 o g) g^2
    (f o g)^3 = ... and so on.
    
    Let F = [f o g, (f^1 o g), (f^2 o g), ...] and G = [g, g^1, g^2, ...] be given. Then this
    function will compute (f o g)^k up to the given (common) order, i.e. the
    chain rule for the k-th derivative.
    
    This routine hereby will make use of the equation
    (f^n o g) = sum_(k = 1)^n f^k B(n, k)(g^1, g^2, ..., g^(n - k + 1)) ,  
    where B(n, k) for n >= k are the incomplete exponential Bell polynomials.
    
    Parameters
    ----------
    f: list
        List containing the values F (see above).
    g: list
        List containing the values G (see above).
        
    Returns
    -------
    list
        List containing the values (f o g)^k for k = 0, ..., len(f) - 1.
    '''
    bell = bell_polynomials(*g[1:])
    return [sum_by_jetpoly([bell.get((n, k), 0)*f[k] for k in range(len(f))]) for n in range(len(f))] # or with numpy arrays: np.matmul(bell, f)

def bell_polynomials(*z):
    '''
    Compute the incomplete exponential Bell polynomials B(n, k, [z1, ..., z_{n - k + 1}]) for all 1 <= k <= n.

    Parameters
    ----------
    z: Values at which the Bell polynomials are supposed to be evaluated. 
       Note that n will be assumed to be the number of different z values.

    Returns
    -------
    dict
        dict of Bell polynomials up to the given order n. Its keys denote the indices (n, k) of the polynomials.
    '''
    n = len(z)
    B = {}
    B[(0, 0)] = 1.0
    nok = nCr(n)
    for jn in range(n + 1):
        for jk in range(1, jn + 1):
            B[(jn, jk)] = sum_by_jetpoly([nok[jn - 1][m - 1]*B.get((jn - m, jk - 1), 0)*z[m - 1] for m in range(1, jn - jk + 2)])
    return B
    
class jet:
    def __init__(self, *values, **kwargs):
        self.__array_priority__ = 1000 # prevent numpy __mul__ and force self.__rmul__ instead if multiplication from left with a numpy object
        self.set_array(*values, **kwargs)
        self.graph = kwargs.get('graph', [(1, '0'), self]) # for keeping track of the computation graph
        
    def get_array(self, **kwargs):
        n = kwargs.get('n', self.order)
        return [self.array(k) for k in range(n + 1)]
    
    def set_array(self, *values, **kwargs):
        omax = len(values) - 1
        if omax > 0:
            zero = values[0]*0
        else:
            zero = 0
        self.array = lambda k: values[k] if k <= omax else zero
        self.set_order(n=kwargs.get('n', omax))
            
    def set_order(self, n):
        self.order = n
        self._factorials = factorials(self.order)
        
    def __len__(self):
        return self.order + 1
    
    def __iter__(self): 
        # called whenever a jet is used in a for loop. If this routine does not exists, self.__getitem__ is used (which may not stop)
        return iter(self.get_array())
    
    def __getitem__(self, n):
        # called whenver a jet is subscribed in the form jet[index].
        result = self.__class__(n=self.order, graph=[(1, f'[{n}]'), self.graph])
        result.array = lambda k: self.array(k)[n]
        return result
        
    def __setitem__(self, index, value):
        current_array = self.get_array()
        current_array[index] = value
        self.set_array(*current_array)
                
    def __str__(self):
        outstr = ''
        for e in self.get_array():
            if not isinstance(e, type(self)):
                outstr += e.__str__()
            else:
                outstr += str(e)
            outstr += ', '
        return f'{self.order}-jet({outstr[:-2]})'

    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
    
    def __neg__(self):
        result = self.__class__(n=self.order, graph=[(1, '-'), self.graph])
        result.array = lambda n: -self.array(n)
        return result
    
    def __add__(self, other):
        other = self.convert(other)
        max_order = max([self.order, other.order])
        result = self.__class__(n=max_order, graph=[(2, '+'), self.graph, other.graph])
        result.array = lambda n: self.array(n) + other.array(n)
        return result
    
    def __radd__(self, other):
        other = self.convert(other)
        return self + other
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return -self + other
        
    def __mul__(self, other):
        other = self.convert(other)
        max_order = max([self.order, other.order])
        result = self.__class__(n=max_order, graph=[(2, '*'), self.graph, other.graph])
        # compute the derivatives      
        f1, f2 = self.get_array(n=max_order), other.get_array(n=max_order)
        glr = general_leibniz_rule(f1, f2)
        result.array = lambda n: glr[n] if n <= max_order else 0
        # result.array = lambda k: general_leibniz_rule(f1[:k + 1], f2[:k + 1]) if k <= max_order else 0
        # The next line would work for arbitrary orders, but it is also much slower instead of pre-loading the array:
        # result.array = lambda n: sum([n_over_k(n, k)*self.array(n - k)*other.array(k) for k in range(n + 1)])
        return result
    
    def __rmul__(self, other):
        other = self.convert(other)
        return self*other
    
    def __pow__(self, other):
        
        other = self.convert(other)
        assert other.order == 0 # only scalars are accepted here.
        m = other.array(0)
        f = self.array(0)
        n = self.order
                
        # compute the maximal number of derivatives of f**m
        for nmax in range(n + 1):
            if m - nmax == 0:
                break
        # N.B. nmax <= self.order
        
        # compute the coefficients in the series [f**m, m*f**(m - 1), m*(m - 1)*f**(m - 2), ...]
        coeff = 1
        h = m
        factors = []
        for j in range(nmax + 1):
            factors.append(coeff)
            coeff *= h
            h -= 1
            
        # compose the above series with the current jet
        result = self.__class__(*[f**(m - k)*factors[k] for k in range(nmax + 1)], n=self.order, graph=[(2, '**'), self.graph, other.graph])
        return result@self
    
    def __rpow__(self, other):
        other = self.convert(other)    
        return other**self
    
    def __truediv__(self, other):
        other = self.convert(other)
        return self.__mul__(other.inv())
    
    def __rtruediv__(self, other):
        other = self.convert(other)
        return other.__mul__(self.inv())
    
    def copy(self):
        '''
        Return a copy of this jet.
        '''
        result = self.__class__(n=self.order, graph=self.graph)
        result.array = self.array
        return result
    
    def truncate(self, max_power):
        result = self.__class__(n=self.order, graph=self.graph)
        def result_array(n):
            try:
                return self.array(n).truncate(max_power)
            except:
                return self.array(n)
        result.array = result_array
        return result
    
    def derive(self, **kwargs):
        '''
        Return the derivative of this jet, determined by shifting its array to the left by 1.
        
        Parameters
        ----------
        index: int, optional
            if an index is provided, then the individual entries of the jet are derived with respect
            to this index, if they are of class jetpoly.
        '''
        array = self.get_array()
        result = self.copy()
        if len(array) == 1:
            result.set_array(0)
        else:
            result.set_array(*array[1:])
        return result
    
    def __matmul__(self, other):
        '''
        Compute the composition of two jets, based on Faà di Bruno's formula.
        It is assumed that the array ar with ar = self.get_array() belongs to the values
        [f o g, f^1 o g, f^2 o g, ... ], where f^k denote the k-th derivative of the function f,
        so this function will compute the values
        [f o g, (f o g)^1, (f o g)^2, ...]
        '''        
        f = self.get_array()
        g = other.get_array()
        return self.__class__(*faa_di_bruno(f, g), n=self.order, graph=[(2, 'o'), self.graph, other.graph])
    
    def inv(self):
        '''
        Compute 1/jet. This will be done by applying the chain rule to
        "1/x o jet".
        '''
        f = self.array(0)
        assert not check_zero(f), 'Division by zero at requested point.'

        facts = self._factorials
        invf = [(-1)**n*facts[n]/f**(n + 1) for n in range(self.order + 1)]
        result = self.copy()
        result.set_array(*invf)
        result = result@self
        result.graph = [(1, '1/'), self.graph]
        return result
            
    def __eq__(self, other):
        if isinstance(self, type(other)):
            if self.order != other.order:
                return False
            else:
                return all([check_zero(self.array(k) - other[k]) for k in range(self.order + 1)])
        elif self.order > 1:
            return all([check_zero(self.array(k) - other) for k in range(self.order + 1)])
        else:
            return check_zero(self.array(0) - other)
            
    def convert(self, other, n=0):
        '''
        Convert an object to an instance of this class.
        '''
        if not isinstance(self, type(other)):
            return self.__class__(other, n=n)
        else:
            return other
        
    def conjugate(self):
        # N.B.: (f^(n)).conjugate() = (f.conjugate())^(n)
        result = self.__class__(n=self.order, graph=[(1, 'cg'), self.graph])
        result.array = lambda n: self.array(n).conjugate()
        return result
    
    def real(self):
        return (self + self.conjugate())/2
    
    def imag(self):
        return (self - self.conjugate())/2/1j
    
    def __call__(self, *z):
        result = []
        for k in range(len(self)):
            ek = self.array(k)
            if hasattr(ek, '__call__'):
                result.append(ek(*z))
            else:
                result.append(ek)
        return self.__class__(*result, n=self.order, graph=[(1, '(*)'), self.graph])
    
    def get_taylor_coefficients(self, n_args: int=0, **kwargs):
        '''Extract the Taylor coefficients from a given jet-evaluation (the output of njet.ad.derive.eval).
        
        Let m be the number of arguments of the involved poylnomials. 
        Then the k-th derivative of f has the form
        (D^k f)(z1, ..., zm) = sum_{j1 + ... + jm = k} Df[j1, ... jm]/(j1! * ... * jm!) * z1**j1 * ... * zm**jm
        = sum_{S(j1, ... jm), j1 + ... + jm = k} C(j1, ..., jm) * Df[j1, ..., jm] * z1**j1 * ... * zm**jm.
        with S(j1, ..., jm) an index denoting the set having the elements {j1, ..., jm},
        Df[j1, ..., jm] := \\partial^j1/\\partial_{z1}^j1 ... \\partial^jm/\\partial_{zm}^jm f (z1, ..., zm)
        and combinatorial factor
        C(j1, ..., jm) = (j1 + ... + jm)!/(j1! * ... * jm!) .
        
        !!! Attention
        The current jet should have jetpoly entries in the k-th order, k > 0.
        
        Parameters
        ----------
        n_args: int
            The total number of involved variables.
            
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
        # Use the longer factorials list; this is required if nested derivatives of different orders are used.
        facts = kwargs.get('facts', self._factorials)
        if len(facts) < len(self._factorials):
            facts = self._factorials
        kwargs['facts'] = facts
        
        # Get the actual Taylor-coefficients
        tc = {}
        # Add constants to tc; this needs to be done separately, because often it may happen that the first
        # entry of a jet is just a float, so that these constants will be missed in the next loop.
        self_array = self.get_array()
        const = self_array[0]
        if not check_zero(const):
            if not hasattr(const, 'get_taylor_coefficients'):
                tc[(0,)*n_args] = const
            else:
                tc.update(const.get_taylor_coefficients(n_args=n_args, **kwargs))
        
        for entry in self_array[1:]: # iteration over the derivatives of order >= 1.
            if not hasattr(entry, 'get_taylor_coefficients'): # skip any non-polynomial entry.
                continue
            tc.update(entry.get_taylor_coefficients(n_args=n_args, **kwargs))
            # Since we loop over derivatives of a specific order, it is ensured that these Taylor coefficients are always different, 
            # so the above procedure does not overwrite existing keys.
        self.tc = tc
        return tc
    
    def extract(self, k: int, **kwargs):
        '''
        Extract the components of the k-th derivative. 
        
        Parameters
        ----------
        k: int
            The order of the derivatives to extract.
            
        tc: dict, optional
            The output of self.get_taylor_coefficients, containing the entries of the derivative. If nothing
            specified, the last evaluation (stored in self.tc) will be used.
            
        Returns
        -------
        dict
            The components of the k-th derivative of self.func.
        '''
        if 'tc' in kwargs.keys():
            self.tc = kwargs['tc']
        if not hasattr(self, 'tc'):
            self.tc = self.get_taylor_coefficients(**kwargs)
        return {j: self.tc[j] for j in self.tc.keys() if sum(j) == k}
    
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
