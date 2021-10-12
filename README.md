# njet: Automatic Differentiation Library

A leightweight AD package, using forward-mode automatic differentiation, in order to determine the
higher-order derivatives of a given function in multiple variables.

## Features

- Higher-order forward-mode automatic differentiation with multiple variables.
- Leightweight and easy to use.
- Support for Sympy.

## Installation

It is recommended to perform the installation inside a virtual environment.

Install this module with pip

```sh
pip install -U git+https://github.com/mtitze/njet.git
```

## Quickstart

An example function we want to differentiate
```python
from njet.functions exp
xmpl = lambda x, y, z: exp(-0.23*x**2 - 0.33*x*y - 0.11*z**2)
```

Generate a class to handle the derivatives of the given function (here up to order 3)
```python
from njet import derive
d1 = derive(xmpl, order=3)
```

Compute the derivatives at a specific point
```python
z = [3, 2, 1]
d1.D(z)

{(0, 0, 1): -0.0034336627423962197,
 (1, 0, 0): -0.03183941815676495,
 (0, 1, 0): -0.015451482340782988,
 (0, 0, 2): -0.0013391284695345256,
 (0, 2, 0): 0.007648483758687579,
 (1, 0, 1): 0.007004671994488288,
 (0, 1, 1): 0.0033993261149722572,
 (2, 0, 0): 0.028886468198304194,
 (1, 1, 0): 0.02122003574800864,
 (2, 1, 0): -0.00758358753285629,
 (1, 1, 1): -0.0001359730445988916,
 (0, 1, 2): 0.004725063299811438,
 (0, 2, 1): -0.0016826664269112674,
 (0, 3, 0): -0.002523999640366901,
 (3, 0, 0): -0.014760754257476227,
 (2, 0, 1): -0.004775538142124663,
 (0, 0, 3): 0.00035000468887492136,
 (1, 2, 0): -0.0003059393503475026,
 (1, 0, 2): 0.009736494072338722}
```
The indices here correspond to the powers of the variables x, y, z
in the multivariate Taylor expansion. They can be translated to
the tensor indices of the corresponding multilinear map using a
built-in routine. Eg.:

Get the gradient and Hessian:
```python
d1.grad()
{(2,): -0.0034336627423962197,
 (0,): -0.03183941815676495,
 (1,): -0.015451482340782988}
```

```python
d1.hess()
{(2, 2): -0.0013391284695345256,
 (1, 1): 0.007648483758687579,
 (0, 2): 0.007004671994488288,
 (1, 2): 0.0033993261149722572,
 (0, 0): 0.028886468198304194,
 (0, 1): 0.02122003574800864}
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
