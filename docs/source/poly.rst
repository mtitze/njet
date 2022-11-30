Polynomials
===========


The *jetpoly* class represents a polynomial in several variables. 

Example to generate two monomials ``p1 = (2 + i)*x0`` and ``p2 = -0.5*x1**2`` (where ``p1`` has
complex coefficient):

.. code-block:: python

    from njet import jetpoly
    
    p1 = jetpoly(2 + 1j, index=0, power=1)
    p2 = jetpoly(-0.5, index=1, power=2)
    87*p1**1 + (p1 - p2)*(p1 + p2) + 6
  > [(174+87j)*x0**1 +
    6 +
    (3+4j)*x0**2 +
    -0.25*x1**4]

These polynomials can be used as jet entries:

.. code-block:: python

    from njet import jet

    j1p = jet(p1, p1, 0)
    j2p = jet(0.22, p2, 0, 0)
    j1p*j2p
  > 3-jet([(0.44+0.22j)*x0**1], [(0.44+0.22j)*x0**1 + (-1-0.5j)*x0**1*x1**2], [(-2-1j)*x0**1*x1**2], [0])
  
Note that elementary functions can be applied only on those jets containing polynomials in the *higher-order* components. The next example
will work:

.. code-block:: python

    from njet.functions import sin
    
    j3p = jet(1, p1, 0, 0)
    sin(j2p*j3p)
  > 3-jet(0.21822962308086932, [(0.4293948777054664+0.2146974388527332j)*x0**1 + -0.48794872466530276*x1**2], [-0.05455740577021733*x1**4 + (-0.03168694127134222-0.0422492550284563j)*x0**2 + (-1.8557738645056285-0.9278869322528143j)*x0**1*x1**2], [(0.6446451181643271+0.859526824219103j)*x0**2*x1**2 + (-0.9767350275217077-0.48836751376085386j)*x0**1*x1**4 + (-0.020782712080944576-0.11430491644519516j)*x0**3 + 0.12198718116632569*x1**6])

However, an expression of the form ``sin(j1p)`` will not work. The reason is that an elementary function applied on a class of type *jetpoly* is not defined.

.. automodule:: njet.jet
    :members:
    :exclude-members: convert, jet, bell_polynomials, general_leibniz_rule, faa_di_bruno

