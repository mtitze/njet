Vector-valued functions
=======================

In the case of a vector-valued function (i.e. functions which return an iterable object), the
``derive`` class will automatically return a list of jet objects upon evaluation, one for each component. Although ``derive`` can handle any composition of (vector-valued) functions in this way, it may be of use to combine multi-dimensional jet output in case they were produced in different processes. Support of handling such output is addressed in the dedicated ``njet.extras`` module. In particular there exist a ``general_fa_di_bruno`` routine and the ``cderive`` class which we shall describe in this section.

Faa di Bruno's formula
----------------------

Faa di Bruno's formula describes the relation between the higher-order derivatives
of a composition function and those of its constituents. In the one-dimensional
case this relation
reads: [1]_

.. math::

    \frac{d^n}{dt^n} f(g(t)) = \sum_{k = 1}^n f^{(k)}(g(t)) \cdot B_{n, k} (g'(t), g''(t), ..., g^{(n - k + 1)}(t)) ,

where :math:`B_{n, k}` denotes the exponential incomplete Bell-polynomials, which can be defined by means
of a generating function as

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

    def f(*x):
        return [x[0]**2 - (x[1] + 10 - 0.5*1j)**(-1) + x[2]*x[0], 1 + x[0]**2*x[1] + x[1]**3, x[2]]

    def g(*x):
        return [(x[0] + x[1]*x[2]*(0.9*1j + 0.56) + 1)**(-3), 2*x[1]*4j + 5, x[0]*x[1]**6]
        
    from njet import derive
    
    z = [0.4, 1.67 + 0.01*1j, -3.5 + 2.1*1j]
        
    dg = derive(g, order=3, n_args=3)
    df = derive(f, order=3, n_args=3)

    evg = dg.eval(*z)
    evf = df.eval(*g(*z))
    
The two lists ``evg`` and ``evf`` contain the jet-evaluations of the two functions at the point ``z`` and ``g(*z)`` for every component, respectively, hence their Taylor-coefficients. For example,
we can get the Taylor-coefficients of ``f`` at ``g(*z)`` for the first component:

.. code-block:: python

    evf[0].get_taylor_coefficients(n_args=3)
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (0.0008719493399318281+0.0045037973966777466j),
     (1, 0, 0): (8.673861926635063+0.3207111009469555j),
     (0, 1, 0): (0.00038016081672203066-0.002549222116121679j),
     (1, 0, 1): (1+0j),
     (2, 0, 0): (2+0j),
     (0, 2, 0): (0.00013975244997413352+0.00022126191190187668j),
     (0, 3, 0): (-3.812406690451878e-05-1.1629372340048169e-05j)}
    
Of course, we could have obtained this result directly by calling ``df`` at ``g(*z)``:

.. code-block:: python

    df(*g(*z))[0]
  > {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (0.0008719493399318281+0.0045037973966777466j),
     (1, 0, 0): (8.673861926635063+0.3207111009469555j),
     (0, 1, 0): (0.00038016081672203066-0.002549222116121679j),
     (1, 0, 1): (1+0j),
     (2, 0, 0): (2+0j),
     (0, 2, 0): (0.00013975244997413352+0.00022126191190187668j),
     (0, 3, 0): (-3.812406690451878e-05-1.1629372340048169e-05j)}

Here we are interested in the Taylor-coefficients of the composition function :math:`f \circ g`. In the
conventional approach we would have to derive the composition function:

.. code-block:: python

    dfg = derive(lambda *z: f(*g(*z)), order=3, n_args=3)
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
     (2, 0, 0): (0.07990334788570935+0.07626091978109452j),
     (2, 1, 0): (0.026157193796522264+0.07905790062294853j),
     (1, 1, 1): (-0.06141922486075783+0.12183805526819907j),
     (0, 1, 2): (-0.044387555537168666+0.028103692172068484j),
     (0, 2, 1): (-0.024739262212972474+0.03699831536003705j),
     (0, 0, 3): (-0.055420714785168794+0.025062593767745747j),
     (1, 2, 0): (-0.006176730345186394+0.14902665971458978j),
     (1, 0, 2): (-0.07169409252671222+0.11241965597172777j),
     (0, 3, 0): (-0.009606971742834805+0.04834992719064144j),
     (3, 0, 0): (0.1063846737609027+0.01989470820185046j),
     (2, 0, 1): (0.05072513398243774+0.12332960939244737j)}
     
However, making use of the general Faa di Bruno formula, we can deduce the same result by combining
the previously computed multi-dimensional jet-evaluations ``evg`` and ``evf``:
     
.. code-block:: python

    gfb = general_faa_di_bruno(evf, evg)
  > gfb[0].get_taylor_coefficients(n_args=3)
    {(0, 0, 0): (-0.032316721543419864+0.07248263230025642j),
     (0, 0, 1): (-0.009661433404866623+0.033781618523554095j),
     (1, 0, 0): (0.02880173173559441+0.11295869722633461j),
     (0, 1, 0): (0.02063565141651755+0.06139363375476808j),
     (1, 0, 1): (-0.016979891699848447+0.1067163337009776j),
     (1, 1, 0): (0.005876780654726825+0.16073332310475294j),
     (0, 2, 0): (-0.013325756923981996+0.04362074226492907j),
     (0, 0, 2): (-0.02701945022246847+0.031297336355818564j),
     (0, 1, 1): (-0.02283956495771866+0.04217330624559905j),
     (2, 0, 0): (0.07990334788570935+0.07626091978109452j),
     (2, 1, 0): (0.026157193796522343+0.0790579006229485j),
     (1, 1, 1): (-0.06141922486075783+0.12183805526819907j),
     (0, 1, 2): (-0.044387555537168666+0.028103692172068467j),
     (0, 2, 1): (-0.024739262212972454+0.03699831536003698j),
     (0, 0, 3): (-0.0554207147851688+0.025062593767745747j),
     (1, 2, 0): (-0.006176730345186361+0.1490266597145896j),
     (1, 0, 2): (-0.0716940925267122+0.11241965597172776j),
     (0, 3, 0): (-0.00960697174283448+0.048349927190641684j),
     (3, 0, 0): (0.10638467376090271+0.01989470820185046j),
     (2, 0, 1): (0.05072513398243774+0.12332960939244737j)}
     
In this way it is possible to calculate and combine intermediate steps of a chain of functions, without taking the derivative of the entire chain in one go.

Function chains
---------------

In the case that a chain of functions :math:`f_1 \circ f_2 ... \circ f_N` needs to be differentiated, and there are repetitions of :math:`f_k`'s in the chain, the generalized Faa di Bruno formula may help in reducing the amount of calculations required.

The main idea behind this goes as follows (docs will be updated soon)
 

.. [1] https://en.wikipedia.org/wiki/Fa%C3%A0_di_Bruno%27s_formula

.. [2] https://mathoverflow.net/questions/106323/faa-di-brunos-formula-for-vector-valued-functions

.. [3] Note that the operator ':math:`\otimes`' can be considered as a commutative when used in an argument of a symmetric tensor.
