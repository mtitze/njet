'''    
    AD-Lib: Automatic Differentiation Library
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

def factorial(n):
    k = 1
    for j in range(1, n + 1):
        k *= j
    return k

def n_over_k(n, k):
    return factorial(n)//(factorial(k)*factorial(n - k))

def general_leibnitz_rule(f1, f2):
    n = len(f1) - 1 # len(f1): number of summands
    return sum([n_over_k(n, k)*f1[n - k]*f2[k] for k in range(n + 1)])

def faa_di_bruno(f, g):
    '''
    Consider the composition f o g of two functions and for h in [f, g] denote by h^k its k-th derivative. Then
    (f o g)^1 = (f^1 o g) g^1
    (f o g)^2 = (f^2 o g) (g^1)**2 + (f^1 o g) g^2
    :
    and so on.
    
    Let F = [f o g, (f^1 o g), (f^2 o g), ...] and G = [g, g^1, g^2, ...] be given. Then this
    function will compute (f o g)^k up to the given (common) order, i.e. the
    chain rule for the k-th derivative.
    
    This routine hereby will make use of the equation
      (f^n o g) = sum_(k = 1)^n f^k B(n, k)(g^1, g^2, ..., g^(n - k + 1)) ,  
    where B(n, k) for n >= k are the incomplete exponential Bell polynomials.
    '''
    bell = bell_polynomials(len(f) - 1, g[1:])
    return [sum([bell.get((n, k), 0)*f[k] for k in range(len(f))]) for n in range(len(f))] # or with numpy arrays: np.matmul(bell, f)

def bell_polynomials(n, z):
    '''
    Compute the incomplete exponential Bell polynomials B(n, k, [z1, ..., z(n - k + 1)]) for all 1 <= k <= n.
    '''
    assert len(z) == n
    B = {}
    B[(0, 0)] = 1.0
    for jn in range(n + 1):
        for jk in range(1, jn + 1):
            B[(jn, jk)] = sum([n_over_k(jn - 1, m - 1)*z[m - 1]*B.get((jn - m, jk - 1), 0) for m in range(1, jn - jk + 2)])
    return B

    
class jet:
    def __init__(self, value=0, **kwargs):
        if hasattr(value, '__getitem__'):
            self.set_array(value, **kwargs)
        else:
            self.set_array([value], **kwargs)
        self.graph = kwargs.get('graph', [(1, '0'), self]) # for keeping track of the computation graph
        
    def get_array(self, **kwargs):
        n = kwargs.get('n', self.order)
        return [self.array(k) for k in range(n + 1)]
    
    def set_array(self, array, **kwargs):
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
        result = jet(n=self.order, graph=[(1, '-'), self.graph])
        result.array = lambda n: -self.array(n)
        return result
    
    def __add__(self, other):
        other = convert(other)
        max_order = max([self.order, other.order])
        result = jet(n=max_order, graph=[(2, '+'), self.graph, other.graph])
        result.array = lambda n: self.array(n) + other.array(n)
        return result
    
    def __radd__(self, other):
        other = convert(other)
        return other + self
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self
        
    def __mul__(self, other):
        other = convert(other)
        max_order = max([self.order, other.order])
        result = jet(n=max_order, graph=[(2, '*'), self.graph, other.graph])
        # compute the derivatives      
        f1, f2 = self.get_array(n=max_order), other.get_array(n=max_order)
        result.array = lambda k: general_leibnitz_rule(f1[:k + 1], f2[:k + 1]) if k <= max_order else 0
        # The next line would work for arbitrary orders, but it is also much slower instead of pre-loading the array:
        # result.array = lambda n: sum([n_over_k(n, k)*self.array(n - k)*other.array(k) for k in range(n + 1)])
        return result
    
    def __rmul__(self, other):
        other = convert(other)
        return other*self
    
    def __pow__(self, other):
        '''
        N.B.: Power defined here only for exponents in field (integer, float, complex numbers).
        '''
        other = convert(other)
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
        for max_der in range(1, nmax + 1):
            fp[(max_der, 0)] = f**layer
            for order in range(1, max_der + 1):
                fp[(max_der, order)] = layer*sum([n_over_k(order - 1, k)*fp[(max_der - 1, order - 1 - k)]*self.array(k + 1) for k in range(order)])
            layer += 1
        # extract the derivatives
        df = [fp[(nmax, order)] for order in range(0, nmax + 1)] + [0]*n_additional_zeros
        result = jet(df)
        result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
    def __rpow__(self, other):
        other = convert(other)    
        return other**self
    
    def __truediv__(self, other):
        other = convert(other)
        return self.__mul__(other.inv())
    
    def __rtruediv__(self, other):
        other = convert(other)
        return other.__mul__(self.inv())
    
    def copy(self):
        '''
        Return a copy of this jet.
        '''
        result = jet(n=self.order, graph=self.graph)
        result.array = self.array
        return result
    
    def derive(self):
        '''
        Return the derivative of this jet, determined by shifting its array to the left by 1.
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
        assert f != 0

        invf = [(-1)**n*factorial(n)/f**(n + 1) for n in range(self.order + 1)]
        result = self.copy()
        result.set_array(invf)
        result = result.compose(self)
        result.graph = [(1, '1/'), self.graph]
        return result

    
def convert(other):
    '''
    Convert an object to a (constant) jet.
    '''
    if not isinstance(other, jet):
        return jet(value=other)
    else:
        return other
    