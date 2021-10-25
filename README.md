# njet: Lightweight automatic differentiation

A lightweight AD package, using forward-mode automatic differentiation, in order to determine the
higher-order derivatives of a given function in several variables.

## Features

- Higher-order (forward-mode) automatic differentiation in several variables.
- Support for NumPy, SymPy and mpmath.
- Differentiation of expressions containing nested higher-order derivatives.
- Lightweight and easy to use.

## Installation

It is recommended to perform the installation inside a virtual environment.

Install this module with pip

```sh
pip install njet
```

## Quickstart

An example function we want to differentiate
```python
from njet.functions import exp
xmpl = lambda x, y, z: exp(-0.23*x**2 - 0.33*x*y - 0.11*z**2)
```

Generate a class to handle the derivatives of the given function (here up to order 3)
```python
from njet import derive
d1 = derive(xmpl, order=3)
```

Evaluate the derivatives at a specific point
```python
z = [3, 2, 1]
d1.eval(z)

{(0, 0, 1): -0.0034336627423962197,
 (1, 0, 0): -0.03183941815676495,
 (0, 1, 0): -0.015451482340782988,
 (0, 0, 2): -0.002678256939069051,
 (0, 2, 0): 0.015296967517375158,
 (1, 0, 1): 0.007004671994488288,
 (0, 1, 1): 0.0033993261149722572,
 (2, 0, 0): 0.05777293639660839,
 (1, 1, 0): 0.02637052986160297,
 (2, 1, 0): -0.03618119104917744,
 (1, 1, 1): -0.005801516569552653,
 (0, 1, 2): 0.0026514743696783604,
 (0, 2, 1): -0.0033653328538225343,
 (0, 3, 0): -0.015143997842201405,
 (3, 0, 0): -0.08856452554485736,
 (2, 0, 1): -0.012710046007253848,
 (0, 0, 3): 0.002100028133249528,
 (1, 2, 0): -0.02100783539052855,
 (1, 0, 2): 0.0054636441557008655}
```
The indices here correspond to the powers of the variables x, y, z
in the multivariate Taylor expansion. They can be translated to
the tensor indices of the corresponding multilinear map using a
built-in routine. Example:

Obtain the gradient and the Hessian
```python
d1.grad()

{(2,): -0.0034336627423962197,
 (0,): -0.03183941815676495,
 (1,): -0.015451482340782988}
```

```python
d1.hess()

{(2, 2): -0.002678256939069051,
 (1, 1): 0.015296967517375158,
 (0, 2): 0.007004671994488288,
 (1, 2): 0.0033993261149722572,
 (0, 0): 0.05777293639660839,
 (0, 1): 0.02637052986160297}
```

## Further reading

https://njet.readthedocs.io/en/latest/index.html

## License

This file is part of njet.

njet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

njet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with touscheklib.  If not, see <https://www.gnu.org/licenses/>.
