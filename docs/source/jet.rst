The n-jet class
===============

The n-jet class is the fundamental object by which higher-order derivatives are computed. Import the class via

.. code-block:: python

    from njet import jet

Create an n-jet:

.. code-block:: python

    j1 = jet(3, 1, n=5)
    j1
  > 5-jet(3, 1, 0, 0, 0, 0)
 
In this example we have generated a jet of order 5, having value 3 and value 1 as it first derivative. 
The order defines the number of derivatives which should be calculated and displayed. 
Since this jet has a non-zero first order derivative, this jet 
corresponds to a variable, while a jet of the form

.. code-block:: python

    j2 = jet(3, n=5)
    j2
  > 5-jet(3, 0, 0, 0, 0, 0)
 
corresponds to the scalar 3. To get the k-th entry of ``j2``, type ``j2.array(k)``. 
A jet is defined by its order and its internal array function. If required, one can
change both of these properties on a specific jet. In our example the array function 
of the jet ``j2`` can be set either directly by redefining ``j2.array``, 
or by using the routine ``j2.set_array``, which may be more convenient if the entries
should be defined according to a specific list.

These jets can now be used in further operations:

.. code-block:: python

    j1*j1
  > 5-jet(9, 6, 2, 0, 0, 0)
  
To better understand what's going on, we can use SymPy variables:

.. code-block:: python

   from sympy import Symbol
   j3 = jet(Symbol('x'), 1, n=5)
   j3**2
 > 5-jet(x**2, 2*x, 2, 0, 0, 0)
 
So we see that the jet ``j3**2`` has the first and second derivative stored. This also works with several variables, but we have
to pay attention:

.. code-block:: python

   j4 = jet(Symbol('y'), 1, n=5)
   j4 - j3
 > 5-jet(-x + y, 0, 0, 0, 0, 0)
   
This will just subtract the first orders and thus we end up with a basic scalar, because the code does not know that we are considering two
*different* variables. What we would have to do is this instead:

.. code-block:: python
   
   j3 = jet(Symbol('x'), Symbol('dx'), n=5)
   j4 = jet(Symbol('y'), Symbol('dy'), n=5)
   j4 - j3
 > 5-jet(-x + y, -dx + dy, 0, 0, 0, 0)

Now we are ready to work with more complicated functions, for example

.. code-block:: python

   from njet.functions import log
   log(j4)*j3
 > 5-jet(x*log(y), dx*log(y) + 1.0*dy*x/y, 2.0*dx*dy/y - 1.0*dy**2*x/y**2, -3.0*dx*dy**2/y**2 + 2.0*dy**3*x/y**3, 8.0*dx*dy**3/y**3 - 6.0*dy**4*x/y**4, -30.0*dx*dy**4/y**4 + 24.0*dy**5*x/y**5)

In particular, we can get the higher-order derivatives at specific points by using symbols for the first derivatives

.. code-block:: python

   point = [2, 3]
   j3a = jet(point[0], Symbol('dx'), n=5)
   j4a = jet(point[1], Symbol('dy'), n=5)
   j3a*j4a**2
 > 5-jet(18, 9*dx + 12*dy, 12*dx*dy + 4*dy**2, 6*dx*dy**2, 0, 0)

and taking into account the corresponding multiplicities.

Internally *njet* will however not work with SymPy symbols, but has its own class *jetpoly* which is taylored specifically for the task to obtain the higher-order derivatives.

There is also NumPy support:

.. code-block:: python

    import numpy as np
    jnp = jet(np.array([2.1, 3.47]), np.array([4.3, -1.2]), n=5)
    jnp
  > 5-jet([2.1 3.47], [ 4.3 -1.2], 0, 0, 0, 0)
    jnp**2
  > 5-jet([ 4.41 12.0409], [18.06 -8.328], [36.98 2.88], 0, 0, 0)

Two jets can also be composed together, to represent the derivative of a composition function. If ``F`` represents the values [f o g, f^1 o g, f^2 o g, ... ],
where f^k denote the k-th derivative of a function f, and ``G`` represents
the values [g, g^1, g^2, ...], then ``F@G`` will compute the values
[f o g, (f o g)^1, (f o g)^2, ...] according to Faa di Bruno's formula (here for single-valued functions).

In the following we list some functions of the jet class.

.. automodule:: njet.jet
    :members:
    :exclude-members: convert, jetpoly

