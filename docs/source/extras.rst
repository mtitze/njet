Vector-valued functions
=======================

In the case of a vector-valued function (i.e. functions which return an iterable object), the
``derive`` class will automatically return a list of jet objects upon evaluation, one for each component. Although ``derive`` can handle any composition of (vector-valued) functions in this way, it may be of use to combine multi-dimensional jet output in case they were produced in different processes. Support for the handling of such cases is given in the dedicated ``njet.extras`` module. In particular, there exist a ``general_fa_di_bruno`` routine and the ``cderive`` class which we shall describe in the following.

Of course, everything what will be discussed here can also be applied to the special 1D case.
However, some of the routines in ``njet.extras`` (in particular those of the ``cderive`` class) will take advantage of jets carrying NumPy arrays and so this should be taken into account. A jet representing a single-valued function may be converted to a jet carrying a 1D NumPy array by means of the ``njet.extras.tile`` routine.

Faa di Bruno's formula
----------------------

Faa di Bruno's formula describes the relation between the higher-order derivatives
of a composition function and those of its constituents. In the one-dimensional
case this relation
reads: [1]_

.. math::

    \frac{d^n}{dt^n} f(g(t)) = \sum_{k = 1}^n f^{(k)}(g(t)) \cdot B_{n, k} (g'(t), g''(t), ..., g^{(n - k + 1)}(t)) ,

where :math:`B_{n, k}` denotes the exponential incomplete Bell-polynomials, which can be defined with
a generating function:

.. math::

    \exp \left(u \sum_{j = 1}^\infty x_j \frac{t^j}{j!} \right) =: \sum_{n \geq k \geq 0} B_{n, k} (x_1, ..., x_{n - k + 1}) \frac{t^n}{n!} u^k .

These polynomials obey a recurrence relation by which internally ``njet`` computes the composition of two jets (by means of the ``@``-operator):

.. math::

    B_{n, k} = \sum_{i = 1}^{n - k + 1} \left(\begin{array}{c} n - 1 \\ i - 1 \end{array}\right) x_i B_{n - i, k - 1} .

A similar -- but more complicated -- expression can be derived in the multi-dimensional case, i.e. where
:math:`f` takes more than one argument and :math:`g` is vector-valued. [2]_ Without loss of generality let :math:`f` be single-valued, so let :math:`f \colon \mathbb{K}^m \to \mathbb{K}` and
:math:`g \colon \mathbb{K}^l \to \mathbb{K}^m` be two (sufficiently often) differentiable functions.
Since the higher-order partial derivatives of the map :math:`f \circ g \; \colon \mathbb{K}^l \to \mathbb{K}` commute, the respective symmetric multilinear map :math:`(f \circ g)^{(n)} \colon (\mathbb{K}^l)^{\otimes n} \to \mathbb{K}` is characterized by its action on the 'diagonal' :math:`z^{\otimes n}`, :math:`z \in \mathbb{K}^l` as

.. math::

    \frac{(f \circ g)^{(n)}(z^{\otimes n})}{n!} = \sum_{\substack{n_1 + ... + n_r = n\\r \geq 1, n_j \geq 1}} \frac{(f^{(r)} \circ g)}{r!} \left(\frac{g^{(n_1)} (z^{\otimes n_1})}{n_1!} \otimes ... \otimes \frac{g^{(n_r)} (z^{\otimes n_r})}{n_r!} \right) ,

by means of the polarization formula [3]_

.. math::

    z_1 \otimes z_2 \otimes ... \otimes z_k = \frac{1}{2^k k!} \sum_{\epsilon_j = \pm 1} \epsilon_1 \cdot ... \cdot \epsilon_k (\epsilon_1 z_1 + ... + \epsilon_k z_k)^{\otimes k} .

Regarding the polynomials in the ``njet`` module, this generalized Faa di Bruno formula can therefore be implemented by summing up those jet-components which correspond to a partition of the integer :math:`n` according to the above equation.

Implementation
--------------

The generalized Faa di Bruno formula can be imported with

.. code-block:: python

    from njet.extras import general_faa_di_bruno

Define two vector-valued functions and compute their jet evaluation at a specific point ``z``:

.. code-block:: python

    f = lambda *x: [x[0]**2 - (x[1] + 10 - 0.5*1j)**(-1) + x[2]*x[0], 1 + x[0]**2*x[1] + x[1]**3, x[2]]
    g = lambda *x: [(x[0] + x[1]*x[2]*(0.9*1j + 0.56) + 1)**(-3), 2*x[1]*4j + 5, x[0]*x[1]**6]
        
    from njet import derive
    
    z = [0.4, 1.67 + 0.01*1j, -3.5 + 2.1*1j]
        
    dg = derive(g, order=2, n_args=3)
    df = derive(f, order=2, n_args=3)

    evg = dg.eval(*z)
    evf = df.eval(*g(*z))
    
The two lists ``evg`` and ``evf`` contain the jet-evaluations of the two functions at the points ``z`` and ``g(*z)`` for every component, and hence their Taylor-coefficients. For example,
we can get the Taylor-coefficients of ``f`` at ``g(*z)`` for the first component (the ``n_args`` argument here is just to tell ``njet`` how to present the 0th-order and not immediately necessary):

.. code-block:: python

    evf[0].taylor_coefficients(n_args=3)
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (0.0008719493399318281+0.0045037973966777466j),
     (1, 0, 0): (8.673861926635063+0.3207111009469555j),
     (0, 1, 0): (0.00038016081672203066-0.002549222116121679j),
     (1, 0, 1): (1+0j),
     (2, 0, 0): (2+0j),
     (0, 2, 0): (0.00013975244997413352+0.00022126191190187668j)}
    
Of course, we could have obtained this result directly by calling ``df`` at ``g(*z)``:

.. code-block:: python

    df(*g(*z))[0]
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (0.0008719493399318281+0.0045037973966777466j),
     (1, 0, 0): (8.673861926635063+0.3207111009469555j),
     (0, 1, 0): (0.00038016081672203066-0.002549222116121679j),
     (1, 0, 1): (1+0j),
     (2, 0, 0): (2+0j),
     (0, 2, 0): (0.00013975244997413352+0.00022126191190187668j)}

Here we are interested in the Taylor-coefficients of the composition function :math:`f \circ g`. In the
conventional approach we would have to derive the composition function:

.. code-block:: python

    dfg = derive(lambda *x: f(*g(*x)), order=2, n_args=3)
    ref = dfg(*z)
    ref[0]
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (-0.009661433404866623+0.03378161852355409j),
     (0, 1, 0): (0.020635651416517554+0.06139363375476808j),
     (1, 0, 0): (0.028801731735594405+0.11295869722633461j),
     (1, 0, 1): (-0.01697989169984845+0.10671633370097759j),
     (1, 1, 0): (0.0058767806547268325+0.16073332310475294j),
     (0, 0, 2): (-0.02701945022246847+0.031297336355818564j),
     (0, 2, 0): (-0.01332575692398203+0.04362074226492914j),
     (0, 1, 1): (-0.022839564957718664+0.04217330624559906j),
     (2, 0, 0): (0.07990334788570935+0.07626091978109452j)}
     
However, making use of the general Faa di Bruno formula, we can deduce the same result by combining
the previously computed multi-dimensional jet-evaluations ``evg`` and ``evf``:
     
.. code-block:: python

    gfb = general_faa_di_bruno(evf, evg)
    gfb[0].taylor_coefficients(n_args=3)
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (-0.009661433404866623+0.033781618523554095j),
     (1, 0, 0): (0.02880173173559441+0.11295869722633461j),
     (0, 1, 0): (0.02063565141651755+0.06139363375476808j),
     (1, 0, 1): (-0.016979891699848447+0.1067163337009776j),
     (1, 1, 0): (0.005876780654726825+0.16073332310475294j),
     (0, 2, 0): (-0.013325756923981996+0.04362074226492907j),
     (0, 0, 2): (-0.02701945022246847+0.031297336355818564j),
     (0, 1, 1): (-0.02283956495771866+0.04217330624559905j),
     (2, 0, 0): (0.07990334788570935+0.07626091978109452j)}
     
In this way it is possible to calculate and combine intermediate steps of a chain, without the need of passing the jets through all of the members of the chain in a single large calculation.
This can become effective in particular if there are repetitions of functions in the chain.

Function chains
---------------

In the case that a chain of functions :math:`f_N \circ f_{N - 1} ... \circ f_1` needs to be differentiated, and there are repetitions of :math:`f_k`'s in the chain, the generalized Faa di Bruno formula may help in reducing the amount of calculations required.

For example, if we want to compute a chain of :math:`2^N` repetitions of the same function :math:`f`, then we can first compute the composition :math:`f_2 := f \circ f`, then the compositon :math:`f_4 := f_2 \circ f_2` and so on, until we are finished after :math:`N` steps.

Moreover, if a chain indeed admits repetitions, a single point will pass through a specific function several times while it is transported from :math:`f_1` to :math:`f_N`. We can take advantage of this by calculating the derivatives at these 'intersecting' points for each unique element of the chain in parallel, using NumPy.
 
In order to manage function chains, the dedicated class ``cderive`` has been created. This class takes a series of functions, defining the *unique* functions in the chain, an ``order`` parameter, defining the number of derivatives to be computed and an optional ``ordering`` parameter, which is a list of integers and defines the chain itself. For example, consider an alternation of a rotation and some
polynomial perturbation :math:`M = 15` times:

.. code-block:: python

    from njet.functions import sin, cos

    g = 1
    per = lambda *z, **kwargs: [z[0], z[1] - g*z[1]**2]
    rot = lambda *z, alpha=0: [cos(alpha)*z[0] - sin(alpha)*z[1], sin(alpha)*z[0] + cos(alpha)*z[1]]

    M = 15
    ordering = [0, 1]*M
    drp = cderive(rot, per, order=2, ordering=ordering, n_args=2)

To compute the derivatives of this chain, we simply call the ``cderive`` object at the point of interest. Hereby we can pass any additional keyworded arguments to the respective
functions (currently every function in the chain will be called with the same set of keyworded parameters, that's why we have provided ``per`` with an ``**kwargs`` parameter). For example:

.. code-block:: python

    z0 = [0.2, 0.1]
    alpha = 1.22
    drp(*z0, alpha=alpha)
  > ({(0, 0): (0.18073767467258015+0j),
      (1, 0): (0.3704449136384162+0j),
      (0, 1): (0.3778714745973873+0j),
      (0, 2): (-1.2481054532394522+0j),
      (2, 0): (-2.5881230191830347+0j),
      (1, 1): (-0.74624063677712+0j)},
     {(0, 0): (-0.012836565325653162+0j),
      (1, 0): (-0.4426107548482863+0j),
      (0, 1): (0.8026126813486407+0j),
      (0, 2): (-0.49946504768391176+0j),
      (2, 0): (0.7197445629662029+0j),
      (1, 1): (-0.3181227776491683+0j)})

The ``cderive`` object ``drp`` in the above example contains two unique elements, which are
instances of the ``derive`` class. These unique elements can be accessed in the ``.dfunctions`` field:

.. code-block:: python

    drp.dfunctions
  > [<njet.ad.derive at 0x7f6a4c09db40>, <njet.ad.derive at 0x7f6a4c09f250>]

If one is interested in obtaining the function at position ``k`` in the chain, then one can type ``drp[k]``, e.g.

.. code-block:: python

    drp[13]
  > <njet.ad.derive at 0x7f6a4c09f250>

In fact, one can iterate over the ``drp`` object as with ordinary lists. The ordering of the chain is stored in ``drp.ordering``. If one wishes to change the ordering on the same object, there is a dedicated routine for it, ``drp.set_ordering``, because with a change in the ordering also other internal changes will become necessary. In particular, any previously computed jet evaluations may become invalid, since points passing through the newly ordered chain will 'hit' the unique functions at different places. Access to these jet-evaluations (the derivatives) are conveniently done by the ``.jev`` function. For example, the Taylor-coefficients of the first component in the current jet evaluation at ``z0`` in the above chain at position 4 can be obtained as follows:

.. code-block:: python

    drp.jev(4)[0].taylor_coefficients(n_args=2)
  > {(0, 0): (-0.0911100230191827+0j),
     (1, 0): 0.34364574631604705,
     (0, 1): -0.9390993563190676}

In some circumstances, however, previously computed jet-evaluations remain valid. For example, this is the case if we want to extract a subchain of a given chain, defined by a slice of neighbouring indices. We can extracted subchains from the original chain in the same manner as it can be done with lists. For example:

.. code-block:: python

    mysubchain = drp[1:13]
    mysubchain.jev(3)[0].taylor_coefficients(n_args=2)
  > {(0, 0): (-0.0911100230191827+0j),
     (1, 0): 0.34364574631604705,
     (0, 1): -0.9390993563190676}

Notice that the index of the same data is now at 3, because the sub-chain starts at index 1 of the original chain.

Another scenario where jet-evaluation data remains valid is by merging patterns occuring in the ordering by means of the general Faa di Bruno formula. A pattern of functions in a chain can be merged by the ``.merge`` command, which will return a new ``cderive`` object in which the pattern has been merged. For example, if we want to merge the first occurence of the pattern ``(1, 0, 1)`` in the chain ``drp`` we could do the following:

.. code-block:: python

    drpm = drp.merge(pattern=(1, 0, 1), positions=[1])
    
The new chain ``drpm`` now has ``len(drpm) = 28``, since we have merged 3 elements of the original
chain of length 30 and added 1 new 'merged' chain. Moreover, the new chain still maintains any previously computed jet-evaluations (although the ones of the merged functions will vanish internally).

The ``.merge`` command requires that jet-evaluation data has been computed in advance. This can be done with the ``.eval`` command (preferably by using the ``compose=False`` option to prevent the automatic composition calculation before the merging command). After merging, the jet-evaluation of the entire chain can be combined by means of successive 'Faa di Bruno' operations using the ``.compose`` routine. In scenarios where chains admit repeated function patterns, these three steps may drastically improve performance in comparison to the conventional ``derive`` approach.

Coming back to our example, we confirm that the results before and after merging are the same:

.. code-block:: python
    
    c1 = drp.compose()
    c2 = drpm.compose()
    
    from njet import taylor_coefficients
    
    taylor_coefficients(c1, n_args=2)
  > ({(0, 0): (0.18073767467258015+0j),
      (1, 0): (0.3704449136384162+0j),
      (0, 1): (0.3778714745973873+0j),
      (0, 2): (-1.2481054532394522+0j),
      (2, 0): (-2.5881230191830347+0j),
      (1, 1): (-0.74624063677712+0j)},
     {(0, 0): (-0.012836565325653162+0j),
      (1, 0): (-0.4426107548482863+0j),
      (0, 1): (0.8026126813486407+0j),
      (0, 2): (-0.49946504768391176+0j),
      (2, 0): (0.7197445629662029+0j),
      (1, 1): (-0.3181227776491683+0j)})

    taylor_coefficients(c2, n_args=2)
  > ({(0, 0): (0.18073767467258015+0j),
      (1, 0): (0.3704449136384162+0j),
      (0, 1): (0.3778714745973873+0j),
      (0, 2): (-1.2481054532394522+0j),
      (2, 0): (-2.5881230191830347+0j),
      (1, 1): (-0.7462406367771199+0j)},
     {(0, 0): (-0.012836565325653162+0j),
      (1, 0): (-0.4426107548482863+0j),
      (0, 1): (0.8026126813486407+0j),
      (0, 2): (-0.49946504768391176+0j),
      (2, 0): (0.7197445629662029+0j),
      (1, 1): (-0.3181227776491683+0j)})

In case one has a periodic system, where the chain is traversed again and again, one might
be interested in the derivatives along the chain for every possible cycle. A cycle of start index :math:`k` is hereby understood as the chain of :math:`N` functions :math:`f_{k - 1} \circ ... \circ f_1 \circ f_N \circ f_{N - 1} \circ ... \circ f_k`. 
Such a calculation would have to be done for every :math:`k`, but it can also be performed in parallel using jets carrying NumPy entries. For this purpose there exist a dedicated routine: ``.cycle``. This routine is called with the same input parameters as the ordinary ``cderive`` function. Returning to our example this would read:

.. code-block:: python

    cyc = drp.cycle(*z0, alpha=alpha)
    
The object ``cyc`` is a list of length ``len(drp)`` which contains the jet-evaluation results for every cycle. This includes our previous result:

.. code-block:: python

    taylor_coefficients(cyc[0], n_args=2)
  > ({(0, 0): (0.18073767467258015+0j),
      (1, 0): (0.3704449136384162+0j),
      (0, 1): (0.3778714745973873+0j),
      (0, 2): (-1.2481054532394522+0j),
      (2, 0): (-2.5881230191830347+0j),
      (1, 1): (-0.74624063677712+0j)},
     {(0, 0): (-0.012836565325653162+0j),
      (1, 0): (-0.4426107548482863+0j),
      (0, 1): (0.8026126813486407+0j),
      (0, 2): (-0.49946504768391176+0j),
      (2, 0): (0.7197445629662029+0j),
      (1, 1): (-0.3181227776491683+0j)})
    
as well as any other result around the cycle. For example, at the next position we check:

.. code-block:: python

    ordering2 = [1] + [0, 1]*(M - 1) + [0]
    drp2 = cderive(rot, per, order=2, ordering=ordering2, n_args=2)
    
    taylor_coefficients(cyc[1], n_args=2) == drp2(*rot(*z0, alpha=alpha), alpha=alpha)
  > True

In case that the underlying functions take parameters, it is important to remember that these parameters must be included in the call of the ``.cycle`` routine. In case some results look weird, there is a ``warn`` switch which might provide some clues.

Similar to other routines, also the ``.cycle`` routine can handle multi-dimensional NumPy arrays:

.. code-block:: python

    z1 = np.array([[0.02, -0.056, z0[0]], [0.0031, 0.0118, z0[1]]])
    cyc = drp.cycle(*z1, alpha=alpha)
    taylor_coefficients(cyc[0], n_args=2)
  > ({(0, 0): array([ 0.01863169+0.j, -0.04082189+0.j,  0.18073767+0.j]),
      (1, 0): array([0.84531442+0.j, 0.80706613+0.j, 0.37044491+0.j]),
      (0, 1): array([0.5277127 +0.j, 0.51823247+0.j, 0.37787147+0.j]),
      (0, 2): array([ 0.57344527+0.j,  1.20526504+0.j, -1.24810545+0.j]),
      (2, 0): array([-0.72251201+0.j,  1.81211122+0.j, -2.58812302+0.j]),
      (1, 1): array([ 0.10110713+0.j,  0.33841931+0.j, -0.74624064+0.j])},
     {(0, 0): array([-0.00775766+0.j,  0.0386481 +0.j, -0.01283657+0.j]),
      (1, 0): array([-0.51833409+0.j, -0.49550034+0.j, -0.44261075+0.j]),
      (0, 1): array([0.85906428+0.j, 0.81893139+0.j, 0.80261268+0.j]),
      (0, 2): array([ 0.55558523+0.j, -0.11809903+0.j, -0.49946505+0.j]),
      (2, 0): array([ 0.25984599+0.j, -0.73940164+0.j,  0.71974456+0.j]),
      (1, 1): array([ 0.13269792+0.j,  0.94135934+0.j, -0.31812278+0.j])})
      
As one can see in the above example, the last entry in each coefficient agrees with our previous result.

Last but not least I would like to stress that the ``.extras`` module is more experimental. Therefore, please check your results carefully and let me know if you encounter any problems or bugs.

Module synopsis
---------------

.. automodule:: njet.extras
    :members:
    :undoc-members:
    :exclude-members: accel_asc, symtensor_call, _get_ordering

.. [1] https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula

.. [2] https://mathoverflow.net/questions/106323/faa-di-brunos-formula-for-vector-valued-functions

.. [3] Note that the operator ':math:`\otimes`' can be considered as being commutative when used in an argument of a symmetric tensor.
