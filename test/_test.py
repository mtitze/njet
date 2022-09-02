from njet import derive
from njet.functions import sin, exp

import sympy
from sympy import Symbol
import numpy as np

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
# Performance
########################
def test_performance(tol1=5, tol2=0.0025, n_points=6000):
    print (f'Performance test using tolarances {tol1}, {tol2} (Attention: Tolerances may have to be adjusted depending on machine) ...')
    import time
    d2 = derive(lambda x, y: exp(-x**2 + y), order=3)

    lin = np.linspace(3, 4, n_points)
    # Long time tests
    start_time = time.time()
    for l in lin:
        r = d2(2.1, l)
    end_time = time.time()
    time1_long = end_time - start_time
    print (f'required time 1: {time1_long}')

    start_time = time.time()
    for l in lin:
        r = d2(2.1, l)
    end_time = time.time()
    time2_long = end_time - start_time
    print (f'required time 2: {time2_long}')

    start_time = time.time()
    for l in lin:
        r = d2(2.1, l)
    end_time = time.time()
    time3_long = end_time - start_time
    print (f'required time 3: {time3_long}')

    # Short time tests
    start_time = time.time()
    r = d2(2.1, lin)
    end_time = time.time()
    time1_short = end_time - start_time
    print (f'required time 4: {time1_short}')

    start_time = time.time()
    r = d2(2.1, lin)
    end_time = time.time()
    time2_short = end_time - start_time
    print (f'required time 5: {time2_short}')

    start_time = time.time()
    r = d2(2.1, lin)
    end_time = time.time()
    time3_short = end_time - start_time
    print (f'required time 6: {time3_short}')

    if np.average([time1_long, time2_long, time3_long]) < tol1 and np.average([time1_short, time2_short, time3_short]) < tol2:
        print ('Performance test succeeded.')
        performance_test = True
    else:
        print ('Performance test failed.')
        performance_test = False
    assert performance_test



if __name__ == '__main__':
    test_ce1()
    test_sd1()
    test_sd2()
    test_sd3()
    test_nd1()
    test_nd2()
    test_nd3()
    test_performance()
    
