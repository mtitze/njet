import pytest

from njet import derive, jet
from njet.functions import sin, cos, exp
from njet.common import buildTruncatedJetFunction

import sympy
from sympy import Symbol
import numpy as np

######################
# Basic derivatives
######################

@pytest.mark.parametrize("point", (0, 0.32, 1.55, -1.268))
def test_cos_sin(point, order=9, tol=1e-15):
    dcos = derive(cos, order)
    dsin = derive(sin, order)

    tc_cos = dcos(point)
    tc_sin = dsin(point)
    
    c = np.cos(point)
    s = np.sin(point)
    
    for k in range(order):
        if k%4 == 0:
            c_expectation = c
            s_expectation = s
        elif k%4 == 1:
            c_expectation = -s
            s_expectation = c
        elif k%4 == 2:
            c_expectation = -c
            s_expectation = -s
        else:
            c_expectation = s
            s_expectation = -c
        
        if c_expectation != 0:
            assert abs(tc_cos[(k,)] - c_expectation) < tol
        if s_expectation != 0:
            assert abs(tc_sin[(k,)] - s_expectation) < tol

######################
# Chained expressions
######################

def test_ce1():
    H = lambda x, y, px, py: 0.24*(x**2 + px**2) + 0.86*(y**2 + py**2) + x**3 - y**3 + 9*x**4*y + sin(x)
    dH = derive(H, order=5)
    
    z = [1.1, 0.14, 1.26, 0.42]
    z0 = [0, 0, 0, 0]
    
    H_shift = lambda x, y, px, py: H(x + z[0], y + z[1], px + z[2], py + z[3])
    dH_shift = derive(H_shift, order=5)
    
    hdict = dH(*z)
    hshiftdict = dH_shift(*z0)
    
    check1 = all([hdict[k] == hshiftdict[k] for k in hdict.keys()])
    
    hdict2 = dH(*z)
    hshiftdict2 = dH_shift(*z0)

    check2 = all([hdict2[k] == hshiftdict2[k] for k in hdict2.keys()])
    test_failed = not (check1 and check2)
    if test_failed:
        print ('Shift test failed.')
    else:
        print ('Shift test succeeded.')
    assert not test_failed
    
    
@pytest.mark.parametrize("x0", (1.55, 0.42, -1.631))
def test_composition_1d(x0, tol=5e-12):
    '''
    Test if the composition @ of two jets agrees with the expectation (in the 1D case).
    '''
    f = lambda x: x**3 + 2*x
    f1 = lambda x: 3*x**2 + 2
    f2 = lambda x: 6*x
    f3 = lambda x: 6

    g = lambda x: 1/x
    g1 = lambda x: -1*x**-2
    g2 = lambda x: 2*x**-3
    g3 = lambda x: -6*x**-4

    # the expectation
    fg = lambda x: x**-3 + 2/x
    fg1 = lambda x: -3*x**-4 - 2*x**-2
    fg2 = lambda x: 12*x**-5 + 4*x**-3
    fg3 = lambda x: -60*x**-6 - 12*x**-4
     
    y0 = g(x0)
    fjet = jet(f(y0), f1(y0), f2(y0), f3(y0))
    gjet = jet(g(x0), g1(x0), g2(x0), g3(x0))
    fgjet = jet(fg(x0), fg1(x0), fg2(x0), fg3(x0))
    diff = fgjet - fjet@gjet
    
    assert all([abs(e) < tol for e in diff.get_array()])
    
    
######################## 
# Symbolic differentiation
########################
def test_sd1():
    d2 = derive(lambda x, y: sin(-x**2 + y), order=4)
    x, y = Symbol('x'), Symbol('y')
    K = d2(x, y)
    a = x**2 - y
    c, s = sympy.cos(a), sympy.sin(a)
    Kref = {}
    Kref[(0, 0)] = -s
    Kref[(1, 0)] = -2*x*c
    Kref[(0, 1)] = c
    Kref[(2, 0)] = 4*x**2*s - 2*c
    Kref[(1, 1)] = -2*x*s
    Kref[(0, 2)] = s
    Kref[(2, 1)] = -4*x**2*c - 2*s
    Kref[(0, 3)] = -c
    Kref[(3, 0)] = 8*x**3*c + 12*x*s
    Kref[(1, 2)] = 2*x*c
    Kref[(0, 4)] = -s
    Kref[(2, 2)] = -4*x**2*s + 2*c
    Kref[(3, 1)] = 8*x**3*s - 12*x*c
    Kref[(1, 3)] = 2*x*s
    Kref[(4, 0)] = -16*x**4*s + 48*x**2*c + 12*s
    test_failed = False
    for k in K.keys():
        check_k = sympy.nsimplify(K[k].expand() - Kref[k].expand()) # use of nsimplify due to: https://stackoverflow.com/questions/64761602/how-to-get-sympy-to-replace-ints-like-1-0-with-1
        if check_k != 0:
            print (k, check_k)
            test_failed = True
    if test_failed:
        print ('Symbolic differentiation test 1 failed.')
    else:
        print ('Symbolic differentiation test 1 succeeded.')
    assert not test_failed


def test_sd2():
    d2 = derive(lambda x, y: -sin(x**2 - y), order=4)
    df = lambda x, y: d2(x, y)[(2, 1)] # dx dx dy - component of the function above.
    df_nested = derive(df, order=3)
    x, y = Symbol('x'), Symbol('y')
    a = x**2 - y
    c, s = sympy.cos(a), sympy.sin(a)
    K2 = df_nested(x, y)
    Kref2 = {}
    Kref2[(0, 0)] = -4*x**2*c - 2*s
    Kref2[(1, 0)] = 8*x**3*s - 12*x*c
    Kref2[(0, 1)] = -4*x**2*s + 2*c
    Kref2[(2, 0)] = 16*x**4*c + 8*x**2*s + 40*x**2*s - 12*c
    Kref2[(1, 1)] = -8*x**3*c - 12*x*s
    Kref2[(0, 2)] = 4*x**2*c + 2*s
    Kref2[(2, 1)] = 16*x**4*s - 8*x**2*c - 40*x**2*c - 12*s
    Kref2[(0, 3)] = 4*x**2*s - 2*c
    Kref2[(3, 0)] = -32*x**5*s + 160*x**3*c + 120*x*s
    Kref2[(1, 2)] = -8*x**3*s + 12*x*c
    test_failed = False
    for k in K2.keys():
        check_k = sympy.nsimplify(K2[k].expand() - Kref2[k].expand())
        if check_k != 0:
            print (k, check_k)
            test_failed = True
    if test_failed:
        print ('Symbolic differentiation test 2 failed.')
    else:
        print ('Symbolic differentiation test 2 succeeded.')
    assert not test_failed


def test_sd3():
    xmpl2 = lambda a, b, c: sin(a**5 + b**2*c)
    dd = derive(xmpl2, order=3)
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    K3 = dd(a, b, c)
    arg = a**5 + b**2*c
    si = sympy.sin(arg)
    co = sympy.cos(arg)
    Kref3 = {}
    Kref3[(0, 0, 0)] = si
    Kref3[(0, 0, 1)] = b**2*co
    Kref3[(1, 0, 0)] = 5*a**4*co
    Kref3[(0, 1, 0)] = 2*b*c*co
    Kref3[(0, 2, 0)] = -4*b**2*c**2*si + 2*c*co
    Kref3[(0, 0, 2)] = -b**4*si
    Kref3[(1, 0, 1)] = -5*a**4*b**2*si
    Kref3[(0, 1, 1)] = -2*b**3*c*si + 2*b*co
    Kref3[(2, 0, 0)] = -25*a**8*si + 20*a**3*co
    Kref3[(1, 1, 0)] = -10*a**4*b*c*si
    Kref3[(2, 1, 0)] = -50*a**8*b*c*co - 40*a**3*b*c*si
    Kref3[(1, 1, 1)] = -10*a**4*b**3*c*co - 10*a**4*b*si
    Kref3[(0, 1, 2)] = -2*b**5*c*co - 4*b**3*si
    Kref3[(0, 2, 1)] = -4*b**4*c**2*co - 10*b**2*c*si + 2*co
    Kref3[(0, 3, 0)] = -8*b**3*c**3*co - 12*b*c**2*si
    Kref3[(3, 0, 0)] = -125*a**12*co - 300*a**7*si + 60*a**2*co
    Kref3[(2, 0, 1)] = -25*a**8*b**2*co - 20*a**3*b**2*si
    Kref3[(0, 0, 3)] = -b**6*co
    Kref3[(1, 2, 0)] = -20*a**4*b**2*c**2*co - 10*a**4*c*si
    Kref3[(1, 0, 2)] = -5*a**4*b**4*co
    test_failed = False
    for k in K3.keys():
        check_k = sympy.nsimplify(K3[k].expand() - Kref3[k].expand())
        if check_k != 0:
            print (k, check_k)
            test_failed = True
    if test_failed:
        print ('Symbolic differentiation test 3 failed.')
    else:
        print ('Symbolic differentiation test 3 succeeded.')
    assert not test_failed


########################
# Nested differentiation
########################
def test_nd1(tol=1e-12, verbose=False):
    f = lambda x, y: sin(1/x + y)
    df = derive(f, order=5)
    print ('Nested derivative test 1 ...')
    zref=[0.2, 1.1]
    nested_test_failed = False
    D = df(*zref)
    for a, b in [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1), (3, 0)]:
        dabf = lambda x, y: df(x, y)[(a, b)]
        ddabf = derive(dabf, order=2)
        D2 = ddabf(*zref)
        if verbose:
            print (f'--- D{a}{b} ---')
        for key in D2.keys():
            key2 = (key[0] + a, key[1] + b)
            rel_error = abs(D[key2] - D2[key])/D[key2]
            check = rel_error < tol
            if verbose:
                print ( rel_error, check )
            if check == False:
                nested_test_failed = True
        if verbose:
            print ('-----------')

    if nested_test_failed:
        print (f'Nested derivative test 1 failed with given tolerance {tol}.')
    else:
        print (f'Nested derivative test 1 succeeded with given tolerance {tol}.')
    assert not nested_test_failed

def test_nd2(tol=1e-12, verbose=False):
    f = lambda x, y: sin(1/x + y)
    df = derive(f, order=3)
    dxxf = lambda x, y: df(x, y)[(1, 1)] 
    dxxyf = lambda x, y: df(x, y)[(2, 1)]
    g = lambda x, y: f(x, y)/(1 + dxxf(x, y)) + dxxyf(x, y)**-3
    dg = derive(g, order=12)

    print ('Nested derivative test 2 ...')
    lin = np.linspace(0.02, 0.32, 11)
    nested_test2_failed = False
    try:
        res = dg(0.2, 1.1)
        res2 = dg(lin, 1.1)
    except:
        nested_test2_failed = True
        
    assert not nested_test2_failed

    for key in res.keys():
        rel_error = abs(res2[key][6] - res[key])/res[key]
        if rel_error > tol:
            if verbose:
                print (f'problem at key: {key}, {rel_error}')
            nested_test2_failed = True

    if nested_test2_failed:
        print (f'Nested derivative test 2 failed with given tolerance {tol}.')
    else:
        print (f'Nested derivative test 2 succeeded with given tolerance {tol}.')
    assert not nested_test2_failed
    
    
def test_nd3():
    def prime(f, k=0):
        df = derive(f, order=1)
        return lambda x: df.grad(x)[(k,)]
    xmpl4 = lambda x: sin(x**2)
    x = Symbol('x')
    c, s = sympy.cos(x**2), sympy.sin(x**2)
    d3f = prime(prime(prime(xmpl4)))(x)
    check = sympy.nsimplify(d3f.expand() - (-8*x**3*c - 12*x*s))
    nested_test3_failed = True
    if check == 0:
        nested_test3_failed = False
    if nested_test3_failed:
        print ('Nested derivative test 3 failed.')
    else:
        print ('Nested derivative test 3 succeeded.')
        
    assert not nested_test3_failed
    
######################## 
# Projection
########################
def test_projection1():
    f1 = lambda *z: z[0] + z[1]**3
    df1 = derive(f1, n_args=2, order=3)
    x = Symbol('x')
    y = Symbol('y')
    out1 = df1(x, 2*y)
    out1_ref = {(0, 0): x + 8*y**3, (1, 0): 1, (0, 1): 12*y**2, (0, 2): 12*y, (0, 3): 6}
    assert out1.keys() == out1_ref.keys()
    for k, v in out1.items():
        assert out1_ref[k] - v == 0

def test_projection2():
    f1 = lambda *z: z[0] + z[1]**3
    df1 = derive(f1, n_args=2, order=3)
    out2 = df1(np.array([2, 1]), np.array([2, 8]))
    out2_ref = {(0, 0): np.array([10., 513.]), (1, 0): np.array([1.0, 1.0]), (0, 1): np.array([12., 192.]), 
                (0, 2): np.array([12., 48.]), (0, 3): np.array([6., 6.])}
    for k, v in out2.items():
        assert (out2_ref[k] == v).all()
        assert v.shape == out2_ref[k].shape
    
def test_projection3():
    f3 = lambda z: z[0]*z[1]
    df3 = derive(f3, n_args=1, order=3)
    out3 = df3(np.array([2, 1, 5]))
    ref3 = {(0,): 2, (1,): 3, (2,): 2}
    for k in ref3.keys():
        assert ref3[k] == out3[k]

def test_projection4():
    f4 = lambda z, w, y: z[0] + z[1]**3 + z[2]*z[0] + y[0]*w[2]
    df4 = derive(f4, n_args=3, order=3)
    
    z0, w0, y0 = np.array([2, 1, 5]), np.array([4, 2, 1]), np.array([-1, -2, 2])
    out3_a = df4(z0, w0, y0)
    out3_b = df4(z0, w0, y0, mult_prm=False, mult_drv=False)
    
    ref_a = {(0, 0, 0): 12, 
             (0, 0, 1): 1,
             (1, 0, 0): 11,
             (0, 1, 0): -1,
             (0, 1, 1): 1,
             (2, 0, 0): 8,
             (3, 0, 0): 6}
    
    ref_b = {(0, 0, 0): 12, 
             (0, 0, 1): 1,
             (1, 0, 0): 11,
             (0, 1, 0): -1,
             (0, 1, 1): 2,
             (2, 0, 0): 8,
             (3, 0, 0): 6}
    
    for k in ref_a.keys():
        assert ref_a[k] == out3_a[k]
        assert ref_b[k] == out3_b[k]
        
def test_projection5():
    f5 = lambda z: z[0, 0] + z[0, 1]**3 + z[0, 2]*z[0, 0] + z[1, 0]*z[2, 2]
    df5 = derive(f5, n_args=1, order=3)
    out5 = df5(np.array([[2, 1, 5], [4, 2, 1], [-1, -2, 2]]))
    ref5 = {(0,): 21, (1,): 17, (2,): 10, (3,): 6}
    for k in ref5.keys():
        assert ref5[k] == out5[k]
        
def test_projection6():
    f6 = lambda z: z[0, :] + z[:, 1]**3 + z[0, :]*z[:, 0] + z[1, 0]*z[2, 2]
    df6 = derive(f6, n_args=1, order=3)
    out6 = df6(np.array([[2, 1, 5], [4, 2, 1], [-1, -2, 2]]))
    ref6 = {(0,): np.array([15., 21.,  0.]), (1,): np.array([14., 24., 23.]), 
            (2,): np.array([10., 16., -8.]), (3,): np.array([6., 6., 6.])}
    for k in ref6.keys():
        assert len(ref6[k]) == len(out6[k])
        assert (ref6[k] == out6[k]).all()

########################
# Differentiation of vector-valued functions
########################

def test_2d_function():
    f2d = lambda x, y: [x**2 + y, x*y + y**3]
    df2d = derive(f2d, n_args=2, order=5)
    x0, y0 = 4, 9
    out2d = df2d(x0, y0)
    ref2d = ({(0, 0): 25.0, (1, 0): 8.0, (0, 1): 1.0, (2, 0): 2.0},
             {(0, 0): 765.0, (1, 0): 9.0, (0, 1): 247.0, (1, 1): 1.0, (0, 2): 54.0, (0, 3): 6.0})
    for j in range(2):
        for k in out2d[j].keys():
            assert out2d[j][k] == ref2d[j][k]


########################
# Truncation
########################
@pytest.mark.parametrize("point, truncate", [(0.23, 1), (0.23, 3), (np.array([0.35, 1.55]), 1), (np.array([0.35, 1.55]), 4)]) 
def test_truncate_1d(point, truncate, order=7):
    '''
    Test if truncation gives the low-order derivatives
    '''
    chain = [cos, sin, lambda x: sin(x)**2, lambda x: x**2 - 3]
    
    def reference_func(z):
        for c in chain:
            z = c(z)
        return z
    dref = derive(reference_func, order=order, n_args=1)
    
    truncated_chain = buildTruncatedJetFunction(*chain, truncate=truncate, n_args=1)
    dtrunc = derive(truncated_chain, order=order, n_args=1)
    
    reference = dref(point)
    result = dtrunc(point)
    for k in result.keys():
        assert (np.array(result[k] - reference[k]) == 0).all()
        