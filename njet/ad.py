'''    
    njet: A leightweight automatic differentiation library for 
          higher-order automatic differentiation
    
    Copyright (C) 2021  Malte Titze

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


from .functions import exp, log
from .jet import jet as jet_source
from .jet import factorials
from .poly import polynom


class jet(jet_source):    
    def __pow__(self, other):
        if not isinstance(other, jet):
            other = jet(value=other) # n.b. convert from ad.py would convert to 'jet_source', not jet'.

        if other.order == 0:
            result = jet_source.__pow__(self, other)
        else:
            '''
            General exponentiation, using exp and log functions.
            '''
            result = exp(other*log(self))
            result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
    def __rpow__(self, other):
        if not isinstance(other, jet):
            other = jet(value=other) # n.b. convert from ad.py would convert to 'jet_source', not jet'.
        return other**self
    
    
class derive:
    '''
    Class to handle the derivatives of a (jet-)function (i.e. a function consisting of a composition
    of elementary functions).
    '''
    
    def __init__(self, func, order=1, **kwargs):
        self.func = func
        self.n_args = self.func.__code__.co_argcount # the number of any arguments of func (before *args)
        self.set_order(order)
        self._Df = {}
        
    def set_order(self, order):
        self.order = order
        self._factorials = factorials(self.order)
        
    def D(self, z, mult=True):
        '''
        Compute the derivatives of a (jet-)function up to n-th order.

        Input
        -----
        z: vector at which the function and its derivatives should be evaluated.
        mult: (Boolean, default: True) Whether or not to include the factorials C(j1, ..., jm) (see below).
            If False, then the C*Df's are returned. If True, then the Df's are returned.

        Returns
        -------
        The tensor components Df_k, k = 1, ..., n, each representing the coefficients of a multilinear map.

        This multilinear map corresponds to the k-th derivative of func: Let m be the number of 
        arguments of func. Then
          Df_k(z1, ..., zm) = sum_{j1 + ... + jm = k} C(j1, ..., jm) * Df[j1, ... jm] * z1**j1 * ... * zm**jm
        with
          Df[j1, ..., jm] := \partial^j1/\partial_{z1}^j1 ... \partial^jm/\partial_{zm}^jm f 
        and combinatorial factor
          C(j1, ..., jm) = 1/(j1! * ... * jm!) .
        '''
        # perform the computation, based on the input vector
        inp = []
        for k in range(self.n_args):
            jk = jet([z[k], polynom(1, index=k, power=1)], n=self.order)
            inp.append(jk)
        evaluate = self.func(*inp)
        
        # extract Df from the result
        Df = {}
        for k in range(1, self.order + 1): # the k-th derivative
            polynomials_k = evaluate.array(k).values
            for key, value in polynomials_k.items(): # loop over the individual polynomials of the k-th derivative
                # key corresponds to a specific frozenset
                indices = [0]*self.n_args
                multiplicity = 1
                for tpl in key:
                    if tpl == (0, 0): # the (0, 0)-entries correspond to the scalar 1 and will be ignored here. TODO: may need to improve this in polynom class.
                        continue
                    index, power = tpl
                    indices[index] = power
                    if mult:
                        multiplicity *= self._factorials[power]
                Df[tuple(indices)] = value/multiplicity
                
        self._Df = Df
        return Df
    
    def extract(self, k, **kwargs):
        '''
        Extract the components of the k-th derivative.
        
        Input
        -----
        - Df (optional): The output of self.D, containing the entries of the derivative.
        '''
        assert k <= self.order
        D = kwargs.get('Df', self._Df)
        return {j: D[j] for j in D.keys() if sum(j) == k}
    
    @staticmethod
    def convert_indices(list_of_tuples):
        l1g = [[tpl[j]*[j] for j in range(len(tpl))] for tpl in list_of_tuples]
        return [tuple(e for subl in l1 for e in subl) for l1 in l1g] # flatten l1 and return list of converted tuples
    
    def build_tensor(self, k, **kwargs):
        '''
        Convert the components of the k-th derivative into the entries of a self.n_args**k tensor. Only values unequal
        to zero will be stored.
        '''
        entries = self.extract(k=k, **kwargs)
        list_of_tuples = list(entries.keys())
        converted_indices = self.convert_indices(list_of_tuples)
        return {converted_indices[k]: entries[list_of_tuples[k]] for k in range(len(list_of_tuples))}
    
    def grad(self, **kwargs):
        '''
        Returns the gradient of the function.
        '''
        return self.build_tensor(k=1, **kwargs)
    
    def hess(self, **kwargs):
        '''
        Returns the Hessian of the function.
        '''
        return self.build_tensor(k=2, **kwargs)
        
        
        
