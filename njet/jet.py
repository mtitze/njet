def check_zero(value):
    # check if a value is zero; value may be an iterable
    check = value == 0
    if hasattr(check, '__iter__'):
        return all(check)
    else:
        return check
    
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
    nok = n_over_ks(nmax)
    return [sum_by_jetpoly([nok[n][k]*f1[n-k]*f2[k] if isinstance(f1[n-k], jetpoly) else nok[n][k]*f2[k]*f1[n-k] for k in range(n + 1)]) for n in range(nmax + 1)]

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
    list
        List containing the values (f o g)^k for k = 0, ..., len(f) - 1.
    '''
    bell = bell_polynomials(len(f) - 1, g[1:])
    return [sum_by_jetpoly([bell.get((n, k), 0)*f[k] for k in range(len(f))]) for n in range(len(f))] # or with numpy arrays: np.matmul(bell, f)

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
    dict
        dict of Bell polynomials up to the given order n. Its keys denote the indices (n, k) of the polynomials.
    '''
    assert len(z) == n
    B = {}
    B[(0, 0)] = 1.0
    nok = n_over_ks(n)
    for jn in range(n + 1):
        for jk in range(1, jn + 1):
            B[(jn, jk)] = sum_by_jetpoly([nok[jn - 1][m - 1]*B.get((jn - m, jk - 1), 0)*z[m - 1] for m in range(1, jn - jk + 2)])
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
        
    def __len__(self):
        return self.order + 1
    
    def __iter__(self): 
        # called whenever a jet is used in a for loop. If this routine does not exists, self.__getitem__ is used (which may not stop)
        return iter(self.get_array())
    
    def __getitem__(self, n):
        # called whenver a jet is subscribed in the form jet[index].
        if isinstance(n, slice):
            start, stop, stride = n.indices(len(self))
            return [self.array(j) for j in range(start, stop, stride)]
        else:
            return self.array(n)
        
    def __setitem__(self, index, value):
        current_array = self.get_array()
        current_array[index] = value
        self.set_array(current_array)
        
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
        glr = general_leibniz_rule(f1, f2)
        result.array = lambda n: glr[n] if n <= max_order else 0
        # result.array = lambda k: general_leibniz_rule(f1[:k + 1], f2[:k + 1]) if k <= max_order else 0
        # The next line would work for arbitrary orders, but it is also much slower instead of pre-loading the array:
        # result.array = lambda n: sum([n_over_k(n, k)*self.array(n - k)*other.array(k) for k in range(n + 1)])
        return result
    
    def __rmul__(self, other):
        other = self.convert(other)
        return other*self
    
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
        result = self.__class__([f**(m - k)*factors[k] for k in range(nmax + 1)], n=self.order, graph=[(2, '**'), self.graph, other.graph])
        return result.compose(self)
    
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
            to this index, if they are of class jetpoly.
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
        return self.__class__(faa_di_bruno(f, g), n=self.order, graph=[(2, 'o'), self.graph, other.graph])
    
    def inv(self):
        '''
        Compute 1/jet. This will be done by applying the chain rule to
        "1/x o jet".
        '''
        f = self.array(0)
        assert not check_zero(f), 'Division by zero at requested point.'

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
            return all([check_zero(self[k] - other[k]) for k in range(self.order + 1)])
    
    def convert(self, other, n=0):
        '''
        Convert an object to an instance of this class.
        '''
        if not isinstance(self, type(other)):
            return self.__class__(value=other, n=n)
        else:
            return other
        
    def conjugate(self):
        # N.B.: (f^(n)).conjugate() = (f.conjugate())^(n)
        result = self.__class__(n=self.order, graph=[(1, 'cg'), self.graph])
        result.array = lambda n: self.array(n).conjugate()
        return result

    
class jetpoly:
    '''
    Implementation of a polynomial with arbitrary number of variables.
    
    The information of the coefficients is stored internally as dictionary in 
    the 'values' field.

    A polynomial p can be initiated with a value v and (optional) its index (describing
    the variable) and power n so that overall p = v*x_i**n. A general polynomial in several
    variables can then be constructed by successive addition and multiplication of different 
    classes with each other.
    
    Example
    -------    
    | self.values = {
    | frozenset({(0, 1)}): 11,
    | frozenset({(0, 4), (1, 22), (13, 1)}): 89,
    | frozenset({(4, 9), (7, 2), (12, 33)}): -0.22
    | }
    
    | The interpretation is:
    | 11*x0**1 +
    | 89*x0**4*x1**22*x13**1 +
    | -0.22*x4**9*x7**2*x12**33
    '''
    def __init__(self, value=0, index: int=0, power: int=0, **kwargs):
        self.values = kwargs.get('values', {frozenset([(index, power)]): value})
        
    def __str__(self):
        outstr = '['
        for key, value in self.values.items():
            fac = ''
            for e in key:
                if e[1] == 0: # do not show z**0
                    continue
                fac += f'*x{e[0]}**{e[1]}'
            outstr += f'{value}]{fac} + \n ['
        return outstr[:-6]
    
    def _repr_html_(self):
        outstr = self.__str__().replace('\n', '<br>')
        return f'<samp>{outstr}</samp>'
    
    def __add__(self, other):
        if not isinstance(self, type(other)):
            other = self.__class__(other)
        new_values = {}
        for k in set(self.values).union(set(other.values)):
            new_value = self.values.get(k, 0) + other.values.get(k, 0)
            if check_zero(new_value): # do not store 0-values; x may be float or numpy array etc.
                continue
            new_values[k] = new_value
        if len(new_values) == 0:
            return self.__class__(0)
        else:
            return self.__class__(values=new_values)
    
    def __radd__(self, other):
        other = self.__class__(other)
        return other + self
    
    def __neg__(self):
        new_values = {}
        for key, value in self.values.items():
            new_values[key] = -value
        return self.__class__(values=new_values)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    def __mul__(self, other):
        # Interpretation: (sum_j a_j)*(sum_k b_k) = sum_{j, k} a_j*b_k
        if not isinstance(self, type(other)):
            other = self.__class__(other)
        pol_prod = {}
        for aj, value1 in self.values.items():
            e1 = dict(aj) # e.g. e1 = {0: 0, 1: 5, 2: 3}
            for bk, value2 in other.values.items():
                value_prod = value1*value2
                e2 = dict(bk) # e.g. e2 = {0: 0, 1: 0, 2: 1}
                e_prod = frozenset([(k, e1.get(k, 0) + e2.get(k, 0)) for k in set(e1).union(set(e2))]) # e.g. e_prod = frozenset([(0, 0), (1, 5), (2, 4)])
                value_prod += pol_prod.get(e_prod, 0) # account for multiplicity
                pol_prod[e_prod] = value_prod
        pol_prod = {k: v for k, v in pol_prod.items() if not check_zero(v)} # remove zero values
        if len(pol_prod) == 0: # in this case zero(s) are produced
            return self.__class__(0)
        else:
            return self.__class__(values=pol_prod)

    def __rmul__(self, other):
        other = self.__class__(other)
        return other*self
    
    def __truediv__(self, other):
        assert not check_zero(other)
        return self.__class__(values={k: v/other for k, v in self.values.items()})
    
    def __pow__(self, other):
        assert type(other) == int
        assert other >= 0
        if other == 0:
            return self.__class__(1) # N.B. 0**0 := 1

        remainder = other%2
        half = self**(other//2)
        if remainder == 1:
            return self*half*half
        else:
            return half*half
        
    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return self.values == self.__class__(other).values
        else:
            return self.values == other.values
        
    def conjugate(self):
        # Consider the following example:
        # Let f = u + 1j*v be a complex function in which u and v are real functions from
        # R^2 -> R^2. Denote by Df := \partial f/\partial z and D'f := \partial f/\partial \bar z
        # the Wirtinger operators and for any g = x + 1j*y the complex conjugation g' = x - 1j*y.
        # Then we have for the differential of f:
        # df = Df*dz + D'f*dz'
        # (df)' = D(f')*dz + D'(f')*dz'
        # As our polynomials represent polynomials in differentials, in this example df plays 
        # the role of a polynomial of the two independent variables dz and dz' (the variables z and z' behave
        # independent when applying the Wirtinger operators to any function g(z, z'), as one can show). This means that
        # in the complex case conjugation should be propagated to the underlying function: f --> f',
        # without changing the differentials themselves (the keys below).
        #
        # However, this requires that the keys (related to the variables) are prepared in advance accordingly:
        # Every variable needs his complex-conjugate partner, and in the original expression complex conjugation
        # needs to be replaced by this partner variable.
        new_values = {}
        for key, value in self.values.items():
            new_values[key] = value.conjugate()
        return self.__class__(values=new_values)
        
    def get_taylor_coefficients(self, n_args: int, facts, mult_prm: bool=True, mult_drv: bool=True):
        '''
        Obtain the Taylor coefficients of the current polynomial.
        
        Parameters
        ----------
        n_args: int
            The total number of involved parameters.
            
        facts: list
            A list containing the factorial numbers up to the maximal order in the current polynomial.
            Hereby it must hold facts[k] = k!.
            
        mult_prm: bool, optional
            Whether or not to include multiplicities into the final result related to the permulation of expressions (e.g. derivatives)
            (default: True).
            
        mult_drv: bool, optional
            Whether or not to include multiplicities related to the derivatives of powers (default: True)
        
        Returns
        -------
        dict
            A dictionary of the form tuple: value, where tuple consists of series of powers, hereby the entry
            j denotes the power of the j-th variable.
        '''
        taylor_coeffs = {}
        for key, value in self.values.items(): # loop over the individual polynomials of the k-th derivative
            # key corresponds to a specific frozenset, i.e. some indices and powers of a specific monomial.
            indices = [0]*n_args
            multiplicity = 1
            for index, power in key:
                if power == 0: # the (k, 0)-entries correspond to the scalar 1 and will be ignored here. TODO: may need to improve this.
                    continue
                indices[index] = power
                if mult_prm or mult_drv:
                    multiplicity *= facts[power]
            if not check_zero(value): # only add non-zero values
                if mult_drv: # remove the factorials in the Taylor expansion, here related to the derivatives of the powers.
                    value *= multiplicity
                if mult_prm:
                    value /= facts[sum(indices)] # the denominator ensures to remove multiplicities emerging from permutations of derivatives.

                taylor_coeffs[tuple(indices)] = value
                
        return taylor_coeffs
        