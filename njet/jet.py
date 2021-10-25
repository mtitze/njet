def check_zero(value):
    # check if a value is zero; value may be an iterable
    if not hasattr(value, '__iter__'):
        return value == 0
    else:
        return all(value == 0)
    
def factorials(n: int):
    k = 1
    facts = [1]
    for j in range(1, n + 1):
        k *= j
        facts.append(k)
    return facts

def n_over_ks(n: int):
    facts = factorials(n)
    return [[facts[j]//(facts[k]*facts[j - k]) for k in range(j + 1)] for j in range(len(facts))]

def sum_by_name(ls, name='jetpolynom'):
    # sum over the members of a given list so that each member e having e.__class__.__name__ appears on the
    # left. This can be used to prevent that in case of numpy the new values are numpy.arrays.
    return sum([e for e in ls if e.__class__.__name__ == name]) + sum([e for e in ls if e.__class__.__name__ != name])

def general_leibnitz_rule(f1, f2):
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
    nok = n_over_ks(nmax)
    return [sum_by_name([nok[n][k]*f1[n-k]*f2[k] if f1[n-k].__class__.__name__ == 'jetpolynom' else nok[n][k]*f2[k]*f1[n-k] for k in range(n + 1)]) for n in range(nmax + 1)]

def faa_di_bruno(f, g):
    '''
    Consider the composition f o g of two functions and for h in [f, g] denote by h^k its k-th derivative. Then
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
    dfdg: list
        List containing the values (f o g)^k for k = 0, ..., len(f) - 1.
    '''
    bell = bell_polynomials(len(f) - 1, g[1:])
    return [sum_by_name([bell.get((n, k), 0)*f[k] for k in range(len(f))]) for n in range(len(f))] # or with numpy arrays: np.matmul(bell, f)

def bell_polynomials(n: int, z):
    '''
    Compute the incomplete exponential Bell polynomials B(n, k, [z1, ..., z(n - k + 1)]) for all 1 <= k <= n.

    Parameters
    ----------
    n: int
        The maximal order to which the Bell polynomials should be computed.
    z: subscriptable
        List of values at which the Bell polynomials are supposed to be evaluated. Note that it must hold: len(z) == n.

    Returns
    -------
    bp: dict
        dict of Bell polynomials up to the given order n. Its keys denote the indices (n, k) of the polynomials.
    '''
    assert len(z) == n
    B = {}
    B[(0, 0)] = 1.0
    nok = n_over_ks(n)
    for jn in range(n + 1):
        for jk in range(1, jn + 1):
            B[(jn, jk)] = sum_by_name([nok[jn - 1][m - 1]*B.get((jn - m, jk - 1), 0)*z[m - 1] for m in range(1, jn - jk + 2)])
    return B

    
class jet:
    def __init__(self, value=0, **kwargs):    
        self.set_array(value, **kwargs)
        self.graph = kwargs.get('graph', [(1, '0'), self]) # for keeping track of the computation graph
        
    def get_array(self, **kwargs):
        n = kwargs.get('n', self.order)
        return [self.array(k) for k in range(n + 1)]
    
    def set_array(self, array, **kwargs):
        if not hasattr(array, '__getitem__') or not hasattr(array, '__iter__'):
            array = [array]
        if hasattr(array, '__len__'):
            omax = len(array) - 1
            self.array = lambda k: array[k] if k <= omax else 0
        else:
            omax = 0
            self.array = lambda k: array[k]
        self.set_order(n=kwargs.get('n', omax))        
        
    def set_order(self, n):
        self.order = n
        
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
        return other + self
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
        
    def __mul__(self, other):
        other = self.convert(other)
        max_order = max([self.order, other.order])
        result = self.__class__(n=max_order, graph=[(2, '*'), self.graph, other.graph])
        # compute the derivatives      
        f1, f2 = self.get_array(n=max_order), other.get_array(n=max_order)
        glr = general_leibnitz_rule(f1, f2)
        result.array = lambda n: glr[n] if n <= max_order else 0
        # result.array = lambda k: general_leibnitz_rule(f1[:k + 1], f2[:k + 1]) if k <= max_order else 0
        # The next line would work for arbitrary orders, but it is also much slower instead of pre-loading the array:
        # result.array = lambda n: sum([n_over_k(n, k)*self.array(n - k)*other.array(k) for k in range(n + 1)])
        return result
    
    def __rmul__(self, other):
        other = self.convert(other)
        return other*self
    
    def __pow__(self, other):
        '''
        N.B.: Power defined here only for exponents in the given field (integer, float, complex numbers),
        not jets themselves as exponents.
        '''
        other = self.convert(other)
        f = self.array(0)
        g = other.array(0)
        n = self.order
        
        # compute the derivatives
        n_additional_zeros = 0
        for nmax in range(n + 1):
            if g - nmax == 0:
                n_additional_zeros = n - nmax
                break
        # nmax defines the depth of the iteration. Any remaining derivatives are set to zero.

        # fundamental equation:
        #  fp[(y, n)] = y*sum([n_over_k(n - 1, k)*fp[(y - 1, n - 1 - k)]*f[k + 1] for k in range(n)]),
        # so fp[(y, n)] depends on fp[(y - 1, n - 1)], fp[(y - 1, n - 2)], ..., fp[(y - 1, 0)]. This means for each "layer" with respect to y
        # we need to compute the entire set of derivatives up to the (n-1)th order. The iteration start begins with n=1 at the lowest layer.
        # The individual layers are labelled by the integer max_der.
        fp = {}
        fp[(0, 0)] = f**(g - nmax)
        layer = g - nmax + 1
        nok = n_over_ks(nmax - 1)
        for max_der in range(1, nmax + 1):
            fp[(max_der, 0)] = f**layer
            for order in range(1, max_der + 1):           
                fp[(max_der, order)] = layer*sum_by_name([nok[order - 1][k]*self.array(k + 1)*fp[(max_der - 1, order - 1 - k)] for k in range(order)])
            layer += 1
        # extract the derivatives
        df = [fp[(nmax, order)] for order in range(0, nmax + 1)] + [0]*n_additional_zeros
        result = self.__class__(df)
        result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
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
    
    def derive(self, **kwargs):
        '''
        Return the derivative of this jet, determined by shifting its array to the left by 1.
        
        Parameters
        ----------
        index: int, optional
            if an index is provided, then the individual entries of the jet are derived with respect
            to this index, if they are of class jetpolynom.
        '''
        array = self.get_array()
        result = self.copy()
        if len(array) == 1:
            result.set_array([0])
        else:
            result.set_array(array[1:])
        return result
    
    def compose(self, other):
        '''
        Compute the composition of two jets, based on Faà di Bruno's formula.
        It is assumed that the array ar with ar = self.get_array() belongs to the values
        [f o g, f^1 o g, f^2 o g, ... ], where f^k denote the k-th derivative of the function f,
        so this function will compute the values
        [f o g, (f o g)^1, (f o g)^2, ...]
        '''        
        f = self.get_array()
        g = other.get_array()
        dfg = faa_di_bruno(f, g)
        result = self.copy()
        result.set_array(dfg)
        return result
    
    def inv(self):
        '''
        Compute 1/jet. This will be done by applying the chain rule to
        "1/x o jet".
        '''
        f = self.array(0)
        assert not check_zero(f)

        facts = factorials(self.order)
        invf = [(-1)**n*facts[n]/f**(n + 1) for n in range(self.order + 1)]
        result = self.copy()
        result.set_array(invf)
        result = result.compose(self)
        result.graph = [(1, '1/'), self.graph]
        return result
            
    def __eq__(self, other):
        other = self.convert(other, n=self.order) # if 'other' is no jet class, convert 'other' to jet with same order as self
        if self.order != other.order:  # jets of different order are considered different
            return False
        else:
            array1 = self.get_array()
            array2 = other.get_array()
            return all([check_zero(array1[k] - array2[k]) for k in range(len(array1))])
    
    def convert(self, other, n=0):
        '''
        Convert an object to an instance of this class.
        '''
        if not other.__class__.__name__ == self.__class__.__name__:
            return self.__class__(value=other, n=n)
        else:
            return other
