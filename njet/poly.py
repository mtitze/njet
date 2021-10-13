class polynom:
    '''
    Implementation of a polynom with arbitrary number of variables.
    
    The information of the coefficients is stored in self.values in form of a dictionary.
    
    Example
    -------    
    self.values = {frozenset({(0, 1)}): 11,
                   frozenset({(0, 4), (1, 22), (13, 1)}): 89,
                   frozenset({(4, 9), (7, 2), (12, 33)}): -0.22}
    
    The interpretation is:
    11*x0**1 +
    89*x0**4*x1**22*x13**1 +
    -0.22*x4**9*x7**2*x12**33
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
                fac += f'x{e[0]}**{e[1]} '
            outstr += f'{value}*{fac} + \n'
        outstr = outstr[:-5]
        outstr += ']'
        return outstr
    
    def _repr_html_(self):
        outstr = self.__str__().replace('\n', '<br>')
        return f'<samp>{outstr}</samp>'
    
    def __add__(self, other):
        if not other.__class__.__name__ == 'polynom':
            other = polynom(other)
        new_values = {}
        for k in set(self.values).union(set(other.values)):
            new_value = self.values.get(k, 0) + other.values.get(k, 0)
            if (lambda x: x == 0 if not hasattr(x, '__iter__') else all(x == 0))(new_value): # do not store 0-values; x may be float or numpy array etc.
                continue
            new_values[k] = new_value
        if len(new_values) == 0:
            return polynom(0)
        else:
            return polynom(values=new_values)
    
    def __radd__(self, other):
        other = polynom(other)
        return other + self
    
    def __neg__(self):
        new_values = {}
        for key, value in self.values.items():
            new_values[key] = -value
        return polynom(values=new_values)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    def __mul__(self, other):
        # Interpretation: (sum_j a_j)*(sum_k b_k) = sum_{j, k} a_j*b_k
        if not other.__class__.__name__ == 'polynom':
            other = polynom(other)
        pol_prod = {}
        for aj, value1 in self.values.items():
            e1 = dict(aj) # e.g. e1 = {0: 0, 1: 5, 2: 3}
            for bk, value2 in other.values.items():
                value_prod = value1*value2
                e2 = dict(bk) # e.g. e2 = {0: 0, 1: 0, 2: 1}
                e_prod = frozenset([(k, e1.get(k, 0) + e2.get(k, 0)) for k in set(e1).union(set(e2))]) # e.g. e_prod = frozenset([(0, 0), (1, 5), (2, 4)])
                value_prod += pol_prod.get(e_prod, 0) # account for multiplicity
                pol_prod[e_prod] = value_prod
        return polynom(values=pol_prod)
    
    def __rmul__(self, other):
        other = polynom(other)
        return other*self
    
    def __pow__(self, other):
        assert (type(other) == int) and (other >= 0)
        if other == 0:
            return polynom(1) # N.B. 0**0 := 1

        remainder = other%2
        half = self**(other//2)
        if remainder == 1:
            return self*half*half
        else:
            return half*half

