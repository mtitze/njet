# njet: Lightweight automatic differentiation

A lightweight AD package, using forward-mode automatic differentiation, in order to determine the
higher-order derivatives of a given function in several variables.

## Features

- Higher-order (forward-mode) automatic differentiation in several variables.
- Support for NumPy, SymPy and mpmath.
- Differentiation of expressions containing nested higher-order derivatives.
- Complex differentiation (Wirtinger calculus) possible.
- Faa di Bruno's formula for vector-valued functions implemented.
- Lightweight and easy to use.

## Installation

Install this module with pip

```sh
pip install njet
```

## Quickstart

An example function we want to differentiate
```python
from njet.functions import exp
f = lambda x, y, z: exp(-0.23*x**2 - 0.33*x*y - 0.11*z**2)
```

Generate a class to handle the derivatives of the given function (in this example up to order 3)
```python
from njet import derive
df = derive(f, order=3)
```

Evaluate the derivatives at a specific point
```python
df(0.4, 2.1, 1.73)

{(0, 0, 0): 0.5255977986928584,
 (0, 0, 1): -0.2000425221825019,
 (1, 0, 0): -0.46094926945363685,
 (0, 1, 0): -0.06937890942745731,
 (0, 0, 2): -0.03949533176976862,
 (0, 2, 0): 0.009158016044424365,
 (1, 0, 1): 0.1754372919540542,
 (0, 1, 1): 0.026405612928090252,
 (2, 0, 0): 0.1624775219121247,
 (1, 1, 0): -0.11260197000076322,
 (2, 1, 0): 0.2827794849469999,
 (1, 1, 1): 0.04285630978229049,
 (0, 1, 2): 0.005213383793609458,
 (0, 2, 1): -0.0034855409065079135,
 (0, 3, 0): -0.0012088581178640162,
 (3, 0, 0): 0.2815805411804125,
 (2, 0, 1): -0.061838944839754675,
 (0, 0, 3): 0.10305063303187477,
 (1, 2, 0): 0.03775850015116166,
 (1, 0, 2): 0.034637405962087094}
```
The indices here correspond to the powers of the variables x, y, z
in the multivariate Taylor expansion. They can be translated to
the tensor indices of the corresponding multilinear map using a
built-in routine. Example:

Obtain the gradient and the Hessian
```python
df.grad()

{(2,): -0.2000425221825019,
 (0,): -0.46094926945363685,
 (1,): -0.06937890942745731}
```

```python
df.hess()

{(2, 2): -0.03949533176976862,
 (1, 1): 0.009158016044424365,
 (0, 2): 0.1754372919540542,
 (1, 2): 0.026405612928090252,
 (0, 0): 0.1624775219121247,
 (0, 1): -0.11260197000076322}
```

## Further reading

https://njet.readthedocs.io/en/latest/index.html

## License

njet: Automatic Differentiation Library

Copyright (C) 2021, 2022, 2023 by Malte Titze

njet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

njet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with njet. If not, see <https://www.gnu.org/licenses/>.
