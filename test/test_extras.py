import numpy as np
import pytest

from njet import derive
from njet.functions import cos, sin
from njet.extras import general_faa_di_bruno, symtensor_call, cderive

def test_point_change(order=2):
    '''
    Basic test to confirm if derivatives are re-calculated if
    the input point was changed.
    '''
    f = lambda x, y: [x**2 + 0.42*x*y - y, 0.25*(x - y)**3]
    g = lambda x, y: [y**2 - 0.21*y - x**5, 0.883*y*x**2 + 0.32*y]
    
    z1 = [0.033, -1.13]
    z2 = [1.158, 0.332]
    
    ref1 = f(*g(*z1))
    ref2 = f(*g(*z2))
    
    dfg = cderive(g, f, order=order)
    r1 = dfg(*z1)
    r2 = dfg(*z2)
    
    assert all([r1[k][0, 0] == ref1[k] for k in range(2)])
    assert all([r2[k][0, 0] == ref2[k] for k in range(2)])

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
    
phi1 = 0 # no rotation at all
phi2 = 0.224*np.pi

@pytest.mark.parametrize("x, y, phi", [(0, 0, phi1), (0.56, 0.67, phi1), 
                                       (np.array([0.1, -0.2]), np.array([0.526, 1.84]), phi1), 
                                       (0, 0, phi2), (0.56, 0.67, phi2), 
                                       (np.array([0.1, -0.2]), np.array([0.526, 1.84]), phi2)])
def test_cderive1(x, y, phi, n_reps: int=5, tol=1e-15):
    '''
    Test the derivatives of the 'n_reps'-times composition of a rotation,
    where the rotation takes an additional parameter (the rotation angle).
    '''
    assert n_reps >= 2
    ordering = [0]*n_reps
    rot = lambda *z, alpha=0: [cos(alpha)*z[0] - sin(alpha)*z[1], sin(alpha)*z[0] + cos(alpha)*z[1]]
    
    dcrot = cderive(rot, ordering=ordering, order=1, n_args=2)
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
        
    assert (np.abs(drot_phi[0].get((1, 0), 0) - refx[0]) < tol).all() # the derivative of the first component of rot in x-direction (== rotation x-component [0] in x-direction)
    assert (np.abs(drot_phi[0].get((0, 1), 0) - refy[0]) < tol).all() # the derivative of the first component of rot in y-direction (== rotation x-component [0] in y-direction)

    assert (np.abs(drot_phi[1].get((1, 0), 0) - refx[1]) < tol).all() # the derivative of the second component of rot in x-direction (== rotation y-component [1] in x-direction)
    assert (np.abs(drot_phi[1].get((0, 1), 0) - refy[1]) < tol).all() # the derivative of the second component of rot in y-direction (== rotation y-component [1] in y-direction)
    
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

opchain1 = [lambda *z, k=k: opk(*z, f=fl[k], g=gl[k], h=hl[k]) for k in range(5)] # "k=k" because of Python's late bindings, see https://stackoverflow.com/questions/49617380/list-comprehension-with-lambda-function

@pytest.mark.parametrize("pattern, ordering", [(pattern1_1, ordering1), (pattern1_2, ordering1), (pattern1_3, ordering1), 
                                               (pattern1_4, ordering1), (pattern2_1, ordering2), (pattern2_2, ordering2),
                                               (pattern2_3, ordering2)])
def test_cderive2(pattern, ordering, point=point, tolerances=tolerances, order=3):
    
    dopchain = cderive(*opchain1, ordering=ordering, order=order, n_args=2)
    
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

def test_cderive3(point=[0.01, -0.053], order=3):
    '''
    Test if extracting two chains and putting their results together will give the same result
    as if we would derive (or compose) the entire chain directly.
    '''
    
    dopchain = cderive(*opchain1, ordering=ordering2, order=order, n_args=2)
    
    dr = derive(dopchain.jetfunc, order=order, n_args=2)
    
    _ = dopchain.eval(*point)
    
    part1 = dopchain[:10]
    part2 = dopchain[10:]
    
    part3 = part1.merge((2, 1))
    
    part4 = part2.merge((2, 1))
    part5 = part2.merge((4, 4))
    part6 = part2.merge((1, 2, 3))
    
    dr1 = part1.compose()
    dr2 = part2.compose()
    
    dr3 = part3.compose()
    
    dr4 = part4.compose()
    dr5 = part5.compose()
    dr6 = part6.compose()
    
    dr12 = general_faa_di_bruno(dr2, dr1)
    dr14 = general_faa_di_bruno(dr4, dr1)
    dr15 = general_faa_di_bruno(dr5, dr1)
    dr16 = general_faa_di_bruno(dr6, dr1)
    
    dr32 = general_faa_di_bruno(dr2, dr3)
    dr34 = general_faa_di_bruno(dr4, dr3)
    dr35 = general_faa_di_bruno(dr5, dr3)
    dr36 = general_faa_di_bruno(dr6, dr3)
    
    ref = dr.eval(*point)
    
    tolerances1 = [1e-15, 1e-15, 1e-14, 5e-14]
    tolerances2 = [1e-15, 1e-15, 1e-14, 2e-13]
    tolerances12 = [tolerances1, tolerances2]

    for r in [dr12, dr14, dr15, dr16, dr32, dr34, dr35, dr36]:
        for k in range(2):
            assert abs((ref[k] - r[k]).array(0)) < tolerances12[k][0] 
            for j in range(1, 4):
                assert  max(np.abs(list((ref[k] - r[k]).array(j).terms.values()))) < tolerances12[k][j]    

#################
# Cycling tests #
#################

def check_jet(A, tolerances: list):
    '''
    Convenience function to check if a jets containing jetpoly objects (as a result
    of a jet-evaluation) is close to zero.
    
    A list of tolerances will be used to compare the maximum of the 
    absolute values of the k-th array with its k-th entry.
    '''
    assert len(tolerances) == A.order + 1
    
    # Check the 0-parts:
    diff0 = np.abs(A.array(0))
    if hasattr(diff0, '__iter__'):
        if len(diff0) > 0:
            diff0 = np.max(diff0)
        else:
            diff0 = 0
    assert diff0 < tolerances[0], f'at 0: {diff0} >= {tolerances[0]}'
    
    # Check the higher-order parts:
    diff = A.get_array()[1:]
    j = 1
    for diffj in diff:
        check = np.abs(list(diffj.terms.values()))
        if len(check) > 0:
            check = np.max(check)
        else:
            check = 0
        assert check < tolerances[j], f'at {j}: {check} >= {tolerances[j]}'
        j += 1
                
cyc_point0 = (0, 0)
cyc_point1 = (0.0004, -0.0005)
                
cyc_ordering0 = [0, 1, 2, 3, 4]
cyc_ordering1 = [0, 1, 2, 3, 4, 3, 4, 3, 2, 3, 4, 3, 1, 0, 2, 0]
cyc_ordering2 = [1, 0, 2, 0, 1, 1, 4, 3, 2, 3, 4, 0, 1, 2, 2, 0, 4, 4, 3, 3]

cyc_tolerances0 = [1e-15, 1e-15, 1e-14, 5e-13]
cyc_tolerances1 = [1e-14, 2e-12, 5e-10, 1e-9]
cyc_tolerances2 = [1e-14, 2e-12, 5e-10, 1e-9]

@pytest.mark.parametrize("point, ordering, tolerances", [(cyc_point0, cyc_ordering0, cyc_tolerances0), 
                                                         (cyc_point0, cyc_ordering1, cyc_tolerances1),
                                                         (cyc_point0, cyc_ordering2, cyc_tolerances2),
                                                         (cyc_point1, cyc_ordering0, cyc_tolerances0),
                                                         (cyc_point1, cyc_ordering1, cyc_tolerances1),
                                                         (cyc_point1, cyc_ordering2, cyc_tolerances2)])
def test_cycling1(point, ordering, tolerances, order=3):
    '''
    Test if we receive the same results if we compute the derivatives of a chain
    separately or using the cycle feature.
    '''
    n_args = 2
    
    # Construct a series of chains, where every chain is cyclic shifted in comparison to the next:
    chains = []
    for k in range(len(ordering)):
        ordering_k = ordering[k:] + ordering[:k]
        chains.append(cderive(*opchain1, ordering=ordering_k, order=order, n_args=n_args))

    # To derive each chain separately we construct derive classes for the cyclic functions (cfunc's):
    def make_cfunc(k):
        c = chains[k]
        def cfunc(*z):
            for e in c:
                z = e.jetfunc(*z)
            return z
        return cfunc
    dchains = []
    for k in range(len(chains)):
        cfunc = make_cfunc(k) # the chain of functions
        dchains.append(derive(cfunc, order=order, n_args=n_args))

    # Compute the reference jet-evaluations (derivatives) at the point of interest:
    # First we have to make a function transporting the start point through the chain up to position k:
    def make_partf(k):
        functions = [opchain1[j] for j in ordering[:k]]
        def partf(*z):
            for f in functions:
                z = f(*z)
            return z
        return partf
    # Now compute the derivatives:
    refdirs = []
    for k in range(len(ordering)):
        partf_k = make_partf(k)
        refdirs.append(dchains[k].eval(*partf_k(*point)))
    
    # Construct the chain-derive class for the entire chain with respect to the given ordering and order:
    dopchain1 = cderive(*opchain1, ordering=ordering, order=order, n_args=n_args)
    # Consistency check: No evaluation results in this class should be taken over from anything previously:
    assert not hasattr(dopchain1, '_evaluation')
    assert not any([hasattr(dopchain1.dfunctions[k], '_evaluation') for k in range(len(opchain1))])
    
    # Cycle through the chain at the point of interest:
    cyc = dopchain1.cycle(*point)
    
    # Compare the results against the references:
    for k in range(len(ordering)):
        for cmp in range(n_args):
            check_jet(refdirs[k][cmp] - cyc[k][cmp], tolerances=tolerances[:order + 1])
        
        
cyc2_ordering = [0, 1, 3, 1, 2, 4, 0, 1, 0, 0, 0, 1, 1, 2, 2]
cyc2_tolerances = [1e-15, 5e-15, 5e-14, 5e-13, 5e-11]

cyc2_q1 = np.array([0.001, -0.0009])
cyc2_p1 = np.array([-0.0018, -0.0018])

cyc2_q2 = np.array([[0.00133, -0.000824], [0.00007, -0.00031]]) 
cyc2_p2 = np.array([[-0.00171, -0.00122], [0.000012, -0.000212]])

@pytest.mark.parametrize("q, p, ordering, tolerances", [(cyc2_q1, cyc2_p1, cyc2_ordering, cyc2_tolerances),
                                                        (cyc2_q2, cyc2_p2, cyc2_ordering, cyc2_tolerances)])
def test_cycling2(q, p, ordering, tolerances, order=4):
    '''
    Test if cycling works with numpy arrays.
    '''
    n_args = 2
    
    # build the chain-derive object
    dopchain1 = cderive(*opchain1, ordering=ordering, order=order, n_args=n_args)
    
    # derive the chain of functions directly to provide a reference
    df = derive(dopchain1.jetfunc, order=order, n_args=n_args)
    ref = df.eval(q, p)
    
    # verify that dopchain1 has not yet any evaluation results stored
    assert not hasattr(dopchain1, '_evaluation')
    assert not any([hasattr(f, '_evaluation') for f in dopchain1.dfunctions])
    
    cyc = dopchain1.cycle(q, p)
    
    for j in range(n_args):
        check_jet(ref[j] - cyc[0][j], tolerances=tolerances)
        
        
cyc3_point1 = 0.0003, 0.00063
cyc3_point2 = -0.00037, 0.00049
cyc3_point3 = np.array([[cyc3_point1[0], cyc3_point2[0]], [cyc3_point1[1], cyc3_point2[1]]])

cyc3_ordering = [0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 1, 4, 3, 2, 4]
cyc3_tolerances = [1e-15, 5e-15, 1e-13, 1e-11]

@pytest.mark.parametrize("point, ordering, tolerances", [(cyc3_point1, cyc3_ordering, cyc3_tolerances),
                                                         (cyc3_point2, cyc3_ordering, cyc3_tolerances),
                                                         (cyc3_point3, cyc3_ordering, cyc3_tolerances)])
def test_cycling3(point, ordering, tolerances, order=3):
    '''
    Test cycling if returning a cderive object:
    1) Comparison to default cycling.
    2) Merging & Comparison.
    3) Using numpy arrays. (Note that in a previous test, the result of using numpy arrays with the 'default' outf-parameter
       already agrees with the single-point results, so the test on equality between the 'default' array-result to the
       array-result for cderive objects should imply also agreement with the single-point results.)
    '''
    
    n_args = 2
    d_ref = cderive(*opchain1, order=order, ordering=ordering, n_args=n_args)
    d1 = cderive(*opchain1, order=order, ordering=ordering, n_args=n_args)

    cyc_ref = d_ref.cycle(*point, outf='default')
    dcyc = d1.cycle(*point, outf=None)

    dcycm = dcyc.merge(pattern=(ordering[2], ordering[3]))

    cyc = dcyc.compose()
    cycm = dcycm.compose()
        
    for j in range(2):
        tc_j = cyc[j]
        tcm_j = cycm[j]
            
        for k in range(len(ordering)):
            tc_ref_j = cyc_ref[k][j]
            check_jet(tc_j[k] - tc_ref_j, tolerances=tolerances)
            check_jet(tcm_j[k] - tc_ref_j, tolerances=tolerances)
            
            
def test_cycling4():
    '''
    This test is similar to the cderive1 test above, 
    but produces some jet entries which do not contain jetpoly entries in their higher-order entries
    and may thus test the _jbuild routine for such cases.
    '''
    per = lambda *z, **kwargs: [z[0], z[1] - z[1]**2]
    rot = lambda *z, alpha=0: [cos(alpha)*z[0] - sin(alpha)*z[1], sin(alpha)*z[0] + cos(alpha)*z[1]]
    z1 = [0.2, 0.1]
    M = 15
    ordering = [0, 1]*M
    drp = cderive(rot, per, order=2, ordering=ordering, n_args=2)
    drp.eval(*z1, alpha=0.15*np.pi, compose=False)
    cyc = drp.cycle(outf=None, warn=True, periodic=True, pdb=True)