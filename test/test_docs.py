# Tests of some of the examples in the documentation
import numpy as np

from njet import taylor_coefficients
from njet.extras import cderive
from njet.functions import sin, cos

def dequal(A, B, tol):
    # Convenience function to check if the values of two dictionaries agree up to a given tolerance
    if not A.keys() == B.keys():
        return False
    for k in A.keys():
        if not abs(A[k] - B[k]) < tol:
            return False
    return True

def test_docs_extras(tol=5e-15):

    g = 1
    per = lambda *z, **kwargs: [z[0], z[1] - g*z[1]**2]
    rot = lambda *z, alpha=0: [cos(alpha)*z[0] - sin(alpha)*z[1], sin(alpha)*z[0] + cos(alpha)*z[1]]

    z0 = [0.2, 0.1]
    
    M = 15
    ordering = [0, 1]*M
    drp = cderive(rot, per, order=2, ordering=ordering, n_args=2)
    
    alpha=1.22
    result1 = drp(*z0, alpha=alpha)
    
    ref1 = ({(0, 0): 0.18073767467258015,
             (1, 0): 0.3704449136384162,
             (0, 1): 0.3778714745973873,
             (0, 2): -1.2481054532394522,
             (2, 0): -2.5881230191830347,
             (1, 1): -0.74624063677712},
            {(0, 0): -0.012836565325653162,
             (1, 0): -0.4426107548482863,
             (0, 1): 0.8026126813486407,
             (0, 2): -0.49946504768391176,
             (2, 0): 0.7197445629662029,
             (1, 1): -0.3181227776491683})
    
    assert all([dequal(result1[k], ref1[k], tol=tol) for k in range(2)])
    
    result2 = drp.jev(4)[0].taylor_coefficients(n_args=2)
    
    assert result2 == {(0, 0): -0.0911100230191827,
                       (1, 0): 0.34364574631604705,
                       (0, 1): -0.9390993563190676}
    
    mysubchain = drp[1:13]
    result3 = mysubchain.jev(3)[0].taylor_coefficients(n_args=2)
    
    assert result3 == result2
    
    drpm = drp.merge(pattern=(1, 0, 1), positions=[1])
    assert len(drpm) == 28 and len(drp) == 30
    
    c1 = drp.compose()
    c2 = drpm.compose()

    result4 = taylor_coefficients(c1, n_args=2)
    assert result4 == result1
    
    result5 = taylor_coefficients(c2, n_args=2)
    assert all([dequal(result5[k], result1[k], tol=tol) for k in range(2)])
    
    cyc = drp.cycle(*z0, alpha=alpha)
    assert len(cyc) == len(drp)
    
    tc0 = taylor_coefficients(cyc[0], n_args=2)
    assert tc0 == ref1
        
    ordering2 = [1] + [0, 1]*(M - 1) + [0]
    drp2 = cderive(rot, per, order=2, ordering=ordering2, n_args=2)
    tc1 = taylor_coefficients(cyc[1], n_args=2)
    assert tc1 == drp2(*rot(*z0, alpha=alpha), alpha=alpha)
    
    z1 = np.array([[0.02, -0.056, z0[0]], [0.0031, 0.0118, z0[1]]])
    cyc_arr = drp.cycle(*z1, alpha=alpha)
    result6 = taylor_coefficients(cyc_arr[0], n_args=2)
    result7 = taylor_coefficients(cyc_arr[1], n_args=2)
    
    for k in range(2):
        for key in result6[k].keys():
            assert abs(result6[k][key][-1] - ref1[k][key]) < tol
            
    for k in range(2):
        for key in result7[k].keys():
            assert abs(result7[k][key][-1] - tc1[k][key]) < tol
