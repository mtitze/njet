Automatic differentiation
=========================

Automatic (forward-mode) differentiation (AD) is performed here by first
defining a function as a composition of elementary functions (as shown
in the introduction).

There is NumPy and SymPy support. For example, running each
AD step separately ...

.. code-block:: python

    import numpy as np
    from njet import derive
    from njet.functions import exp

    d2 = derive(lambda x, y: exp(-x**2 + y), order=3)
    lin = np.linspace(3, 4, 4)

    for e in lin:
        print( d2(2.1, e) )
        
    {(0, 0): 0.24414328315343706, (1, 0): -1.0254017892444356, (0, 1): 0.24414328315343706, (2, 0): 3.818400948519755, (1, 1): -1.0254017892444356, (0, 2): 0.24414328315343706, (2, 1): 3.8184009485197556, (0, 3): 0.24414328315343706, (3, 0): -11.935676826805235, (1, 2): -1.0254017892444356}
    {(0, 0): 0.34072939947024816, (1, 0): -1.4310634777750424, (0, 1): 0.34072939947024816, (2, 0): 5.329007807714682, (1, 1): -1.4310634777750424, (0, 2): 0.34072939947024816, (2, 1): 5.329007807714681, (0, 3): 0.34072939947024816, (3, 0): -16.657578881301497, (1, 2): -1.4310634777750422}
    {(0, 0): 0.47552618349279985, (1, 0): -1.9972099706697595, (0, 1): 0.47552618349279985, (2, 0): 7.4372295098273895, (1, 1): -1.9972099706697595, (0, 2): 0.47552618349279985, (2, 1): 7.4372295098273895, (0, 3): 0.47552618349279985, (3, 0): -23.247524058596003, (1, 2): -1.9972099706697595}
    {(0, 0): 0.6636502501363193, (1, 0): -2.7873310505725413, (0, 1): 0.6636502501363193, (2, 0): 10.379489912132033, (1, 1): -2.7873310505725413, (0, 2): 0.6636502501363193, (2, 1): 10.379489912132033, (0, 3): 0.6636502501363193, (3, 0): -32.44453342866438, (1, 2): -2.787331050572541}

... can be speed up by directly passing the NumPy arrays into the code:

.. code-block:: python
   
    d2(2.1, lin)
    
    {(0, 0): array([0.24414328, 0.3407294 , 0.47552618, 0.66365025]),
     (1, 0): array([-1.02540179, -1.43106348, -1.99720997, -2.78733105]),
     (0, 1): array([0.24414328, 0.3407294 , 0.47552618, 0.66365025]),
     (2, 0): array([ 3.81840095,  5.32900781,  7.43722951, 10.37948991]),
     (1, 1): array([-1.02540179, -1.43106348, -1.99720997, -2.78733105]),
     (0, 2): array([0.24414328, 0.3407294 , 0.47552618, 0.66365025]),
     (2, 1): array([ 3.81840095,  5.32900781,  7.43722951, 10.37948991]),
     (0, 3): array([0.24414328, 0.3407294 , 0.47552618, 0.66365025]),
     (3, 0): array([-11.93567683, -16.65757888, -23.24752406, -32.44453343]),
     (1, 2): array([-1.02540179, -1.43106348, -1.99720997, -2.78733105])}

It is also possible to pass SymPy symbols:

.. code-block:: python

    from sympy import Symbol
    d2(Symbol('x'), Symbol('y'))
    
    {(0, 0): 1.0*exp(-x**2 + y),
     (1, 0): -2.0*x*exp(-x**2 + y),
     (0, 1): 1.0*exp(-x**2 + y),
     (2, 0): 4.0*x**2*exp(-x**2 + y) - 2.0*exp(-x**2 + y),
     (1, 1): -2.0*x*exp(-x**2 + y),
     (0, 2): 1.0*exp(-x**2 + y),
     (2, 1): 4.0*x**2*exp(-x**2 + y) - 2.0*exp(-x**2 + y),
     (0, 3): 1.0*exp(-x**2 + y),
     (3, 0): -8.0*x**3*exp(-x**2 + y) + 12.0*x*exp(-x**2 + y),
     (1, 2): -2.0*x*exp(-x**2 + y)}

Nested expressions
==================

Expressions containing higher-order derivatives can straightforwardly be derived. For example, in the case
of one variable:

.. code-block:: python

    def prime(f, k=0):
        # Return \partial f / \partial x_k
        df = derive(f, order=1)
        return lambda x: df.grad(x)[(k,)]
     
    from njet.functions import sin     
    xmpl = lambda x: sin(x**2)
        
    from sympy import Symbol
    f3 = prime(prime(prime(xmpl)))(Symbol('x'))
    f3.expand() 
  > -8*x**3*cos(x**2) - 12*x*sin(x**2)

Here a more sophisticated example for two variables:

.. code-block:: python

    f = lambda x, y: sin(1/x + y)
    df = derive(f, order=3)
    
    dxxf = lambda x, y: df(x, y)[(1, 1)] 
    dxxyf = lambda x, y: df(x, y)[(2, 1)]
    
    g = lambda x, y: f(x, y)/(1 + dxxf(x, y)) + dxxyf(x, y)**-3
    
We obtain the derivatives of the function ``g``, containing the function ``f`` itself
and some of its higher-order derivatives, up to fourth order:

.. code-block:: python

    dg = derive(g, order=4)
    dg(0.2, 1.1)    
 
    {(0, 0): 0.05125472033924497,
     (1, 0): -1.2893245550004897,
     (0, 1): 0.07784343804658536,
     (0, 2): 1.0912311590829322,
     (2, 0): 608.7138384279668,
     (1, 1): -25.28585653666342,
     (2, 1): 13059.768934688353,
     (0, 3): 22.863930873679625,
     (3, 0): -321809.8708462019,
     (1, 2): -540.6030970198217,
     (3, 1): -9115018.77817841,
     (0, 4): 638.7760280114579,
     (1, 3): -15256.040672728306,
     (4, 0): 227762162.9608126,
     (2, 2): 370126.0266704479}
     
Of course, this also synergizes with either NumPy arrays or SymPy symbols. E.g.:

.. code-block:: python

    dg(np.linspace(0.2, 0.64, 5), 1.1)
    
    {(0, 0): array([0.05125472, 0.10722629, 0.37526799, 0.10042023, 0.39647601]), 
     (1, 0): array([-1.28932456,  0.8244532 , 10.59476301,  1.63373827, 12.93969049]),
     (0, 1): array([ 0.07784344, -0.00505105, -1.19829647, -0.41567626, -3.08965192]),
     (0, 2): array([ 1.09123116e+00,  1.70120125e-02,  1.48647193e+01, -1.90986854e+00, 6.21825808e+01]),
     (2, 0): array([608.71383843,   6.65953138, 770.48242475, -21.18301586, 946.75436686]),
     (1, 1): array([ -25.28585654,   -0.24971327, -108.54451995,    5.59886998, -245.80653651]),
     (2, 1): array([ 1.30597689e+04, -8.68288739e+00, -1.32996630e+04, -1.96661374e+02, -2.38423630e+04]),
     (0, 3): array([ 2.28639309e+01, -3.49863419e-02, -2.75171895e+02, -1.34122433e+01, -1.70211185e+03]),
     (3, 0): array([-3.21809871e+05,  1.13907036e+02,  8.86275855e+04,  8.81153961e+02, 8.62690730e+04]),
     (1, 2): array([-5.40603097e+02,  6.12535213e-01,  1.93861032e+03,  4.78129009e+01, 6.43635888e+03]),
     (3, 1): array([-9.11501878e+06, -3.41856494e+02, -2.10459443e+06,  7.83829609e+03, -2.60956636e+06]),
     (0, 4): array([ 6.38776028e+02,  1.56152245e-01,  6.79315246e+03, -1.18250949e+02, 5.53778566e+04]),
     (1, 3): array([-1.52560407e+04, -2.16266355e+00, -4.69698546e+04,  4.43941387e+02, -2.04118615e+05]),
     (4, 0): array([ 2.27762163e+08,  3.56314244e+03,  1.36083587e+07, -3.62297999e+04, 9.01586775e+06]),
     (2, 2): array([ 3.70126027e+05,  2.87900980e+01,  3.17958465e+05, -1.80550952e+03, 7.37728082e+05])}
  

Complex differentiation
=======================

Wirtinger calculus can (currently) be realized by modifying a function so that every variable has their
complex conjugate counterpart. So what will *not* work is 

.. code-block:: python

    f = lambda z: z.conjugate()
    df = derive(f)  
    df(Symbol('z'))
  > {(0,): conjugate(z), (1,): 1.0}

The correct way to deal with this anti-holomorphic function would be:

.. code-block:: python

    f = lambda z, zc: zc
    df = derive(f)  
    df(Symbol('z'), Symbol('z').conjugate())
  > {(0, 0): conjugate(z), (0, 1): 1.0}

In the following the AD routines are explained in more detail.

.. automodule:: njet.ad
    :members:

