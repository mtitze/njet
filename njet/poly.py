from .jet import factorials, check_zero

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
            outstr += f'{value}{fac} + \n'
        outstr = outstr[:-4]
        outstr += ']'
        return outstr
    
    def _repr_html_(self):
        outstr = self.__str__().replace('\n', '<br>')
        return f'<samp>{outstr}</samp>'
    
    def __add__(self, other):
        if not other.__class__.__name__ == self.__class__.__name__:
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
        if not other.__class__.__name__ == self.__class__.__name__:
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
        if not other.__class__.__name__ == self.__class__.__name__:
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
        
        