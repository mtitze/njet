import numpy as np

from njet import derive
from njet.extras import general_faa_di_bruno, symtensor_call

def test_symtensor_call():
    
    tol = 1e-15
    
    terms = {frozenset({(0, 1), (2, 1)}): 2.0,
             frozenset({(0, 2)}): 2.0,
             frozenset({(0, 0), (1, 2)}): -0.0003242141771667308}
    
    ref1 = 44.796693015392904
    ref2 = 5.9967578582283325
    
    result1 = symtensor_call([[7, 2, 4], [2.1, 5.1, 1.1]], terms)
    result2 = symtensor_call([[1, 2, 4], [2, 5, 1]], terms)

    assert abs(result1 - ref1) < tol
    assert abs(result2 - ref2) < tol

def test_general_faa_di_bruno():
    
    def f1(*x):
        return [x[0]**2 - (x[1] + 10 - 0.5*1j)**(-1) + x[2]*x[0], 1 + x[0]**2*x[1] + x[1]**3, x[2]]

    def g1(*x):
        return [(x[0] + x[1]*x[2]*(0.9*1j + 0.56) + 1)**(-3), 2*x[1]*4j + 5, x[0]*x[1]**6]
    
    start = [0.4, 1.67 + 0.01*1j, -3.5 + 2.1*1j]
    
    n = 7
    dg1 = derive(g1, order=n, n_args=3)
    df1 = derive(f1, order=n, n_args=3)
    
    evg1 = dg1.eval(*start)
    evf1 = df1.eval(*g1(*start))
    
    # Reference
    dfg1 = derive(lambda *z: f1(*g1(*z)), order=n, n_args=3)
    ref = dfg1.eval(*start)
    
    # General Faa di Bruno
    gen_fb = general_faa_di_bruno(evf1, evg1)
    
    tolerances = [[1e-15, 5e-15, 5e-15, 5e-14, 5e-14, 1e-12, 5e-12],
                  [1e-15, 5e-12, 1e-12, 5e-15, 5e-15, 1e-14, 2e-13],
                  [1e-15, 1e-15, 8e-14, 1e-15, 1e-15, 1e-15, 1e-15]]

    for k in range(3):
        diff = (gen_fb[k] - ref[k]).get_array()[1:]
        j = 0
        for e in diff:
            check = np.abs(list(e.terms.values()))
            if len(check) > 0:
                check = max(check)
            else:
                check = 0
            assert check < tolerances[k][j]
            j += 1
    