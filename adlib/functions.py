'''    
    AD-Lib: Automatic Differentiation Library
    Copyright (C) 2021  Malte Titze

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


# Collection of elementary functions whose derivatives are known as close expressions up to any order.

from .jet import jet

import numpy as np
import sympy

def detect_code(x):
    # Find the appropriate code to perform operations.
    # We shall assume that the code labels are identical (otherwise we may need to
    # implement a dictionary translating between the various names).
    code = np
    if hasattr(x, '__module__'):
        if x.__module__[0:5] == 'sympy':
            code = sympy
    return code

def sin(x, **kwargs):
    x0 = x.array(0)
    code = kwargs.get('code', detect_code(x0))
    s = code.sin(x0)
    c = code.cos(x0)
    result = jet(s, n=x.order, graph=[(1, 'sin'), x.graph])

    # compute the derivatives
    result.array = lambda n: [s, c, -s, -c][n%4]
    return result.compose(x)

def exp(x, **kwargs):
    x0 = x.array(0)
    code = kwargs.get('code', detect_code(x0))
    e = code.exp(x0)
    result = jet(e, n=x.order, graph=[(1, 'exp'), x.graph])

    # compute the derivatives
    result.array = lambda n: e
    return result.compose(x)

def log(x, **kwargs):
    x0 = x.array(0)
    code = kwargs.get('code', detect_code(x0))
    ln = code.log(x0)
    graph=[(1, 'log'), x.graph]

    # compute the derivatives
    dx = x.copy().derive()
    drx_arr = (dx/x).get_array()[:-1]
    result = jet(n=x.order, graph=graph)
    result.set_array([ln] + drx_arr)
    return result