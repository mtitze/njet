import numpy as np
import pytest

from njet import derive
from njet.functions import cos, sin
from njet.extras import general_faa_di_bruno, symtensor_call, cderive

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
    
@pytest.mark.parametrize("x, y", [(0, 0), (0.56, 0.67), (np.array([0.1, -0.2]), np.array([0.526, 1.84]))])
def test_cderive1(x, y, phi=0.224*np.pi, n_reps: int=5, tol=1e-15):
    '''
    Test the derivatives of the 'n_reps'-times composition of a rotation,
    where the rotation takes an additional parameter (the rotation angle).
    '''
    assert n_reps >= 2
    ordering = [0]*n_reps
    rot = lambda *z, alpha=0: [cos(alpha)*z[0] - sin(alpha)*z[1], sin(alpha)*z[0] + cos(alpha)*z[1]]
    
    dcrot = cderive(functions=[rot], ordering=ordering, order=1, n_args=2)
    dref = derive(dcrot.jetfunc, order=1, n_args=2)
    ref = dref(x, y, alpha=phi/len(ordering))
    
    drot_phi = dcrot(x, y, alpha=phi/len(ordering))
    ref0 = rot(x, y, alpha=phi)
    refx = rot(1, 0, alpha=phi)
    refy = rot(0, 1, alpha=phi)
    
    # consistency checks
    if (0, 0) in drot_phi[0].keys():
        assert (np.abs(drot_phi[0][0, 0] - ref0[0]) < tol).all()
    if (0, 0) in drot_phi[1].keys():
        assert (np.abs(drot_phi[1][0, 0] - ref0[1]) < tol).all()
        
    assert (np.abs(drot_phi[0][1, 0] - refx[0]) < tol).all() # the derivative of the first component of rot in x-direction (== rotation x-component [0] in x-direction)
    assert (np.abs(drot_phi[0][0, 1] - refy[0]) < tol).all() # the derivative of the first component of rot in y-direction (== rotation x-component [0] in y-direction)

    assert (np.abs(drot_phi[1][1, 0] - refx[1]) < tol).all() # the derivative of the second component of rot in x-direction (== rotation y-component [1] in x-direction)
    assert (np.abs(drot_phi[1][0, 1] - refy[1]) < tol).all() # the derivative of the second component of rot in y-direction (== rotation y-component [1] in y-direction)
    
    dcrotm = dcrot.merge((0, 0), positions=[0, 3])
    result1 = dcrotm(x, y, alpha=phi/len(ordering))
    
    dcrotm_all = dcrot.merge()
    result2 = dcrotm_all(x, y, alpha=phi/len(ordering))
    
    dcrotm2 = dcrotm.merge()
    result3 = dcrotm2(x, y, alpha=phi/len(ordering))
    
    for result in [drot_phi, result1, result2, result3]:
        for k in range(2):
            assert all([(np.abs(ref[k][key] - result[k][key]) < tol).all() for key in ref[k].keys()])

    
ordering1 = [0, 1, 2, 3, 2, 1, 2, 4, 2, 1, 2, 4, 2, 1, 0]
pattern1_1 = (0, 1)
pattern1_2 = (2, 3, 2)
pattern1_3 = (1, 0)
pattern1_4 = (4, 2)

ordering2 = [0, 1, 2, 3, 2, 1, 2, 1, 2, 1, 4, 4, 2, 1, 2, 3, 2, 1, 0]
pattern2_1 = (2, 1, 2)
pattern2_2 = (4, 4)
pattern2_3 = (1, 2, 3)

point = [0.2, 0.14]

tolerances = {(0, 0): 1e-15, (1, 0): 5e-15, (0, 1): 1e-15, (0, 2): 5e-15, (2, 0): 1e-14, 
              (1, 1): 1e-14, (2, 1): 5e-13, (0, 3): 1e-13, (3, 0): 1e-12, (1, 2): 5e-13}

@pytest.mark.parametrize("pattern, ordering", [(pattern1_1, ordering1), (pattern1_2, ordering1), (pattern1_3, ordering1), 
                                               (pattern1_4, ordering1), (pattern2_1, ordering2), (pattern2_2, ordering2),
                                               (pattern2_3, ordering2)])
def test_cderive2(pattern, ordering, point=point, tolerances=tolerances, order=3):
        
    # Define a chain of vector-valued functions
    f1 = [-0.41210319-0.93736294*1j, -0.12546181+0.99583435*1j, -0.05060304+0.72636541*1j, -0.57277969+0.7030248*1j]
    g1 = [ 0.55362006+0.09802837*1j, -0.8305559 +0.49600788*1j,  0.03343595+0.58669087*1j, 0.94227846-0.58840718*1j]
    h1 = [ 0.68077738+0.24660968*1j,  0.73098886+0.87092588*1j, -0.86900825-0.97546189*1j, -0.21817777+0.71363444*1j]

    f2 = [-0.20674188-0.04716913*1j, -0.60197835-0.65897942*1j,  0.79173922-0.20115845*1j, -0.39323279-0.47489544*1j]
    g2 = [ 0.25888809+0.74213104*1j, -0.46120294-0.45297721*1j, -0.68132361+0.36020504*1j, 0.36831401+0.8925468*1j ]
    h2 = [ 0.67546745-0.25333062*1j, -0.24332418+0.01746218*1j, -0.1762223 +0.70213807*1j,-0.26571277-0.59505904*1j]

    f3 = [ 0.76785636-0.27781605*1j,  0.46282915+0.27441113*1j, -0.49791468-0.57996221*1j, -0.48946897+0.62037892*1j]
    g3 = [ 0.28804634-0.98764309*1j, -0.87406307+0.52595492*1j,  0.46174138+0.29365197*1j, -0.00551896-0.47692102*1j]
    h3 = [-0.48132238-0.5247383*1j , -0.62638017+0.26876055*1j, -0.31827884+0.06467531*1j, 0.13965988-0.20589607*1j]
    
    f4 = [-0.8096255 +0.96735152*1j,  0.06001575-0.31173884*1j, -0.69704779+0.33130702*1j, -0.1666179 +0.98778434*1j]
    g4 = [-0.66590557+0.74032939*1j,  0.76306924+0.50447005*1j, -0.29480535+0.81453141*1j, -0.38647232+0.60297993*1j]
    h4 = [-0.205739  +0.94352981*1j, -0.66742403+0.68390301*1j,  0.23363041-0.50400264*1j, 0.47711142-0.51922051*1j]

    f5 = [-0.8096255 +0.96735152*1j,  0.06001575-0.31173884*1j, -0.69704779+0.33130702*1j, -0.1666179 +0.98778434*1j]
    g5 = [-0.66590557+0.74032939*1j,  0.76306924+0.50447005*1j, -0.29480535+0.81453141*1j, -0.38647232+0.60297993*1j]
    h5 = [-0.205739  +0.94352981*1j, -0.66742403+0.68390301*1j,  0.23363041-0.50400264*1j, 0.47711142-0.51922051*1j]
 
    fl = [f1, f2, f3, f4, f5]
    gl = [g1, g2, g3, g4, g5]
    hl = [h1, h2, h3, h4, h5]

    def opk(*z, f=[], g=[], h=[]):
        return [f[0]*z[0]**3 + f[1]*z[0]**2 + f[2]*z[0] + g[0]*z[1]**3 + \
                g[1]*z[1]**2 + g[2]*z[1], h[0]*z[0] + h[1]*z[1]]
    
    opchain = [lambda *z, k=k: opk(*z, f=fl[k], g=gl[k], h=hl[k]) for k in range(5)] # k=k because of Python's late bindings, see https://stackoverflow.com/questions/49617380/list-comprehension-with-lambda-function
    
    dopchain = cderive(functions=opchain, ordering=ordering, order=order, n_args=2)
    
    cref = derive(dopchain.jetfunc, order=order, n_args=2)
    ref = cref(*point)#, disable_tqdm=disable_tqdm)
    
    r0 = dopchain(*point)

    dopm = dopchain.merge(pattern)
    r1 = dopm(*point)
    
    dopm2 = dopm.merge()
    r2 = dopm2(*point)
        
    for r in [r0, r1, r2]:
        for k in range(2):
            assert all([abs(ref[k][key] - r[k][key]) < tolerances[key] for key in ref[k].keys()])
