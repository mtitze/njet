import os
import numpy as np

from njet.functions import log
from njet import derive

# Examples related from issue 11; this should go through without errors


######### Test1

xfilename = os.path.join(os.path.dirname(__file__), 'example1.npy')
x = np.load(xfilename)

def grad(fun):
    "A decorator to evaluate the gradient."
    
    def inner(x, **kwargs):
        # init the output
        out = np.zeros((9, *x.shape[2:]))
        
        # obtain the gradient and loop over items
        for key, value in fun.grad(*flatten(x), **kwargs).items():
            out[key] = value
        
        # reshape the gradient
        return out.reshape(3, 3, -1)
    
    return inner

def hess(fun):
    "A decorator to evaluate the hessian."

    def inner(x, **kwargs):
        # init the output
        out = np.zeros((9, 9, *x.shape[2:]))
        
        # obtain the hessian and loop over items
        for key, value in fun.hess(*flatten(x), **kwargs).items():
            out[key] = value
        
        # reshape the hessian
        return out.reshape(3, 3, 3, 3, -1)
    
    return inner

def trace(x):
    return x[0][0] + x[1][1] + x[2][2]

def det(x):
    a = x[0][0] * x[1][1] * x[2][2]
    b = x[0][1] * x[1][2] * x[2][0]
    c = x[0][2] * x[1][0] * x[2][1]
    d = x[2][0] * x[1][1] * x[0][2]
    e = x[2][1] * x[1][2] * x[0][0]
    f = x[2][2] * x[1][0] * x[0][1]
    return a + b + c - d - e - f

flatten = lambda x: x.reshape(9, -1)
reshape = lambda x: [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]]
# reshape = lambda x: np.array(x, dtype=object).reshape(3, 3, -1) # this fails

def tensorjet(fun):
    "A decorator to reshape the input."
    def inner(*x, **kwargs):
        return fun(reshape(x), **kwargs)
    return inner

@tensorjet
def fun(C):
    "The Neo-Hookean material formulation (isotropic hyperelasticity)."
    return trace(C) - log(det(C))

def test_examples1():
    '''
    Test issue 11 example
    '''
    dfun = derive(fun, order=2, n_args=9)
    dfundx = grad(dfun)(x)
    d2fundx2 = hess(dfun)(x)
    
    r1filename = os.path.join(os.path.dirname(__file__), 'example1_result1.npy')
    r1 = np.load(r1filename)
    r2filename = os.path.join(os.path.dirname(__file__), 'example1_result2.npy')
    r2 = np.load(r2filename)

    assert dfundx.shape == (3, 3, 2**4)
    assert d2fundx2.shape == (3, 3, 3, 3, 2**4)
    
    assert (dfundx == r1).all()
    assert (d2fundx2 == r2).all()

######### Test2

def trace2(x):
    return x[0, 0] + x[1, 1] + x[2, 2]

def det2(x):
    a = x[0, 0] * x[1, 1] * x[2, 2]
    b = x[0, 1] * x[1, 2] * x[2, 0]
    c = x[0, 2] * x[1, 0] * x[2, 1]
    d = x[2, 0] * x[1, 1] * x[0, 2]
    e = x[2, 1] * x[1, 2] * x[0, 0]
    f = x[2, 2] * x[1, 0] * x[0, 1]
    return a + b + c - d - e - f

def fun2(C):
    return trace2(C) - log(det2(C))

def test_examples2():
    '''
    Test a similar function as in the issue 11 example, but now where the 
    argument is assumed to contain a tensor variable. The result
    will not make sense in the original context; the test is only here to
    ensure that a result is produced.
    '''
    dfun2 = derive(fun2, order=2, n_args=1)
    y = dfun2(x)
    
    for k in range(3):
        rkfilename = os.path.join(os.path.dirname(__file__), f'example2_result_{k}.npy')
        ref_k = np.load(rkfilename)
        assert (ref_k == y[(k,)]).all()
        
    
if __name__ == '__main__':
    test_examples1()
    test_examples2()

