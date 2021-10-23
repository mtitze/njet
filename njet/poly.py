from .jet import factorials

def check_zero(value):
    # check if a value is zero; value may be an iterable
    if not hasattr(value, '__iter__'):
        return value == 0
    else:
        return all(value == 0)

class jetpolynom:
    '''
    Implementation of a polynom with arbitrary number of variables.
    
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
        if not other.__class__.__name__ == 'jetpolynom':
            other = jetpolynom(other)
        new_values = {}
        for k in set(self.values).union(set(other.values)):
            new_value = self.values.get(k, 0) + other.values.get(k, 0)
            if check_zero(new_value): # do not store 0-values; x may be float or numpy array etc.
                continue
            new_values[k] = new_value
        if len(new_values) == 0:
            return jetpolynom(0)
        else:
            return jetpolynom(values=new_values)
    
    def __radd__(self, other):
        other = jetpolynom(other)
        return other + self
    
    def __neg__(self):
        new_values = {}
        for key, value in self.values.items():
            new_values[key] = -value
        return jetpolynom(values=new_values)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    def __mul__(self, other):
        # Interpretation: (sum_j a_j)*(sum_k b_k) = sum_{j, k} a_j*b_k
        if not other.__class__.__name__ == 'jetpolynom':
            other = jetpolynom(other)
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
        return jetpolynom(values=pol_prod)
    
    def __rmul__(self, other):
        other = jetpolynom(other)
        return other*self
    
    def __pow__(self, other):
        assert (type(other) == int) and (other >= 0)
        if other == 0:
            return jetpolynom(1) # N.B. 0**0 := 1

        remainder = other%2
        half = self**(other//2)
        if remainder == 1:
            return self*half*half
        else:
            return half*half
        
    def flatten(self, collection=None):
        '''
        Flatten jetpolynom recursively to accumulate all orders, in case
        the jetpolynomial has jets as entries.
        
        Parameters
        ----------
        collection: dict, optional
            A dictionary mapping a collection of frozensets to specific values.
        
        Returns
        -------
        dict:
            The values of the new flattened polynomial.
            
        Example
        -------
        | j1 = jet([7, jetpolynom(1, index=3, power=2)], n=5)
        | j2 = jet([j1, jetpolynom(1, index=2, power=1), j1**2], n=5)
        
        | Now consider the polynomial
        
        | (1/j2).array(3)
        
        | This polynomial consists of nested jets:
        
        | [5-jet(0.8571428571428571, [-0.12244897959183676*x3**2], [0.034985422740524796*x3**4], [-0.014993752603082056*x3**6], 
        | [0.00856785863033245*x3**8], [-0.006119899021666472*x3**10])*x2**1 +
        | 5-jet(0.0, [0], [0], [0], [0], [0]) +
        | 5-jet(0.0, [0], [0], [0], [0], [0])*x2**2 +
        | 5-jet(-0.0024989587671803417, [0.0014279764383887665*x3**2], [-0.0010199831702776905*x3**4], [0.0008742712888094488*x3**6], 
        | [-0.0008742712888094488*x3**8], [0.0009991671872107942*x3**10])*x2**3]
        
        | The result of the flatten operation is to accumulate the different powers recursively. As a result we obtain:
        
        | {
        |  frozenset({(0, 0), (2, 1)}): 0.8571428571428571,
        |  frozenset({(0, 0), (2, 1), (3, 2)}): -0.12244897959183676,
        |  frozenset({(0, 0), (2, 1), (3, 4)}): 0.034985422740524796,
        |  frozenset({(0, 0), (2, 1), (3, 6)}): -0.014993752603082056,
        |  frozenset({(0, 0), (2, 1), (3, 8)}): 0.00856785863033245,
        |  frozenset({(0, 0), (2, 1), (3, 10)}): -0.006119899021666472,
        |  frozenset({(0, 0)}): 0,
        |  frozenset({(0, 0), (2, 3)}): -0.0024989587671803417,
        |  frozenset({(0, 0), (2, 3), (3, 2)}): 0.0014279764383887665,
        |  frozenset({(0, 0), (2, 3), (3, 4)}): -0.0010199831702776905,
        |  frozenset({(0, 0), (2, 3), (3, 6)}): 0.0008742712888094488,
        |  frozenset({(0, 0), (2, 3), (3, 8)}): -0.0008742712888094488,
        |  frozenset({(0, 0), (2, 3), (3, 10)}): 0.0009991671872107942
        | }
        '''
        if collection == None: # prevents overriding of default variable {}
            collection = {}
            
        for fs, v in self.values.items():
            if v.__class__.__name__ == 'jet':
                new_jet = v*jetpolynom(values={fs: 1})
                for p in new_jet.get_array():
                    collection = p.flatten(collection=collection)
            else:
                collection[fs] = collection.get(fs, 0) + v
        return collection
        

