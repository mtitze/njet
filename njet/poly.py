import warnings
from .common import check_zero, factorials

class jetpoly:
    '''
    Implementation of a polynomial with arbitrary number of variables.
    
    The information of the coefficients is stored internally as dictionary in 
    the 'terms' field.

    A polynomial p can be initiated with a value v and (optional) its index (describing
    the variable) and power n so that overall p = v*x_i**n. A general polynomial in several
    variables can then be constructed by successive addition and multiplication of different 
    classes with each other.
    
    Example
    -------    
    | self.terms = {
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
        self.__array_priority__ = 1000 # prevent numpy __mul__ and force self.__rmul__ instead if multiplication from left with a numpy object
        if 'terms' in kwargs.keys():
            self.terms = kwargs['terms']
        elif not check_zero(value):
            self.terms = {frozenset([(index, power)]): value}
        else:
            self.terms = {}
            
    def truncate(self, max_power):
        # drop all entries beyond a specific power
        new_terms = {}
        for k, v in self.terms.items():
            max_power_k = sum([e[1] for e in k])
            if max_power_k <= max_power:
                new_terms[k] = v            
        return self.__class__(terms=new_terms)
        
    def __str__(self):
        if len(self.terms) == 0:
            return '0'
        else:
            outstr = '['
            for key, value in self.terms.items():
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
        new_terms = {}
        for k in set(self.terms).union(set(other.terms)):
            new_value = self.terms.get(k, 0) + other.terms.get(k, 0)
            if check_zero(new_value): # do not store 0-terms; x may be float or numpy array etc.
                continue
            new_terms[k] = new_value
        if len(new_terms) == 0:
            return self.__class__(0)
        else:
            return self.__class__(terms=new_terms)
    
    def __radd__(self, other):
        other = self.__class__(other)
        return other + self
    
    def __neg__(self):
        new_terms = {}
        for key, value in self.terms.items():
            new_terms[key] = -value
        return self.__class__(terms=new_terms)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
    
    def __mul__(self, other):
        # Interpretation: (sum_j a_j)*(sum_k b_k) = sum_{j, k} a_j*b_k
        if not isinstance(self, type(other)):
            return self.__class__(terms={k: v*other for k, v in self.terms.items()})
        pol_prod = {}
        for aj, value1 in self.terms.items():
            e1 = dict(aj) # e.g. e1 = {0: 0, 1: 5, 2: 3}
            for bk, value2 in other.terms.items():
                value_prod = value1*value2
                e2 = dict(bk) # e.g. e2 = {0: 0, 1: 0, 2: 1}
                e_prod = frozenset([(k, e1.get(k, 0) + e2.get(k, 0)) for k in set(e1).union(set(e2))]) # e.g. e_prod = frozenset([(0, 0), (1, 5), (2, 4)])
                pol_prod[e_prod] = value_prod + pol_prod.get(e_prod, 0) # account for multiplicity
        pol_prod = {k: v for k, v in pol_prod.items() if not check_zero(v)} # remove zero values
        if len(pol_prod) == 0: # in this case zero(s) are produced
            return self.__class__(0)
        else:
            return self.__class__(terms=pol_prod)

    def __rmul__(self, other):
        other = self.__class__(other)
        return other*self
    
    def __truediv__(self, other):
        assert not check_zero(other)
        return self.__class__(terms={k: v/other for k, v in self.terms.items()})
    
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
            return self.terms == self.__class__(other).terms
        else:
            return self.terms == other.terms
        
    def __getitem__(self, n):
        return self.__class__(terms={k: v[n] for k, v in self.terms.items() if not check_zero(v[n])})
        
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
        new_terms = {}
        for key, value in self.terms.items():
            new_terms[key] = value.conjugate()
        return self.__class__(terms=new_terms)
    
    def real(self):
        return (self + self.conjugate())/2
    
    def imag(self):
        return (self - self.conjugate())/2/1j
        
    def get_taylor_coefficients(self, n_args: int=0, facts=[], mult_prm: bool=True, mult_drv: bool=True):
        '''
        Obtain the Taylor coefficients of the current polynomial.
        
        Parameters
        ----------
        n_args: int, optional
            The total number of involved parameters. If nothing specified, the number of involved variables
            is determined from the current indices of the polynomial.
            
        facts: list, optional
            A list containing the factorial numbers up to the maximal order in the current polynomial.
            Hereby it must hold facts[k] = k!
            
        mult_prm: bool, optional
            Whether or not to include multiplicities into the final result related to the permulation of expressions (e.g. derivatives)
            (default: True).
            
        mult_drv: bool, optional
            Whether or not to include multiplicities related to the derivatives of powers (default: True)
        
        Returns
        -------
        dict
            A dictionary of the form "tuple: value", where "tuple" consists of series of powers, hereby the entry
            j denotes the power of the j-th variable.
        '''
        if n_args == 0 or len(facts) == 0:
            # determine the number of arguments from the maximal index of the polynomial.
            order, indices_set = self.get_order(return_indices=True)
            try:
                n_args = max([n_args, max(indices_set) + 1])
            except:
                # in this case indices_set was empty, so the polynomial power is 0
                warnings.warn("'n_args' parameter recommended.")
                return {}
            if len(facts) == 0:
                facts = factorials(order)
                
        taylor_coeffs = {}
        for key, value in self.terms.items(): # loop over the individual polynomials of the k-th derivative
            # key corresponds to a specific frozenset, i.e. some indices and powers of a specific monomial.
            
            # Step 1: Construct the indices containing the powers of the variables at their respective positions.
            # If there is no index with a power > 0, then 'indices' will not be changed. Also determine
            # the multiplicity.
            multiplicity = 1
            indices = [0]*n_args
            for index, power in key:
                # index: the index of the variable; power: its respective power
                if power == 0: # the (k, 0)-entries correspond to the scalar 1 and will be ignored here.
                    continue
                indices[index] = power
                if mult_prm or mult_drv:
                    multiplicity *= facts[power]
                    
            # Step 2: Update the Taylor-coefficients dictionary
            if not check_zero(value): # only add non-zero values
                if mult_drv: # remove the factorials in the Taylor expansion, here related to the derivatives of the powers.
                    value *= multiplicity
                if mult_prm:
                    value = value/facts[sum(indices)] # the denominator ensures to remove multiplicities emerging from permutations of derivatives.

                tpl = tuple(indices)
                taylor_coeffs[tpl] = taylor_coeffs.get(tpl, 0) + value
                
        return taylor_coeffs
        
    def __call__(self, *z):
        '''
        Evaluate the current polynomial at a specific point.
        
        Note that a sufficient number of parameters must be given, according to the
        indices of the variables the polynomial represents. For example, a polynomial dependent
        on "x4" has to be provided with at least 4 parameters.
        '''
        result = 0
        for fs, v in self.terms.items():
            f = 1
            for index, power in fs:
                f *= z[index]**power
            result += f*v
        return result
          
    def get_order(self, return_indices=False):
        '''
        Return the order of the current polynomial.
        
        Parameters
        ----------
        return_indices: boolean, optional
            If true, also return the indices of the current polynomial.
            
        Returns
        -------
        int
            The order.
            
        set
            A set of indices as described above.
        '''
        ps = 0
        args = set()
        for monomial_basis in self.terms.keys():
            monomial_power = 0
            for index, power in monomial_basis:
                monomial_power += power
                args.add(index)
            ps = max([ps, monomial_power])
        if return_indices:
            return ps, args
        else:
            return ps
        
        