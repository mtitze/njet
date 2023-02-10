# Collection of elementary functions whose derivatives are known to any order.

import numpy
import sympy
import mpmath

def get_function(code: str, name: str):
    '''
    Return the function to handle objects of a specific code.
    
    Parameters
    ----------
    code: str
        The name of the code. Currently supported codes are: 'numpy', 'mpmath',
        'sympy', 'njet'.
        
    name: str
        The name of the function to be returned. See the source code of this
        module for a list of supported functions. 
    '''
    
    sympy_dict = {'sin': sympy.sin,
                  'cos': sympy.cos,
                  'sinh': sympy.sinh,
                  'cosh': sympy.cosh,
                  'exp': sympy.exp,
                  'log': sympy.log}
    
    numpy_dict = {'sin': numpy.sin,
                  'cos': numpy.cos,
                  'sinh': numpy.sinh,
                  'cosh': numpy.cosh,
                  'exp': numpy.exp,
                  'log': numpy.log}
    
    mpmath_dict = {'sin': mpmath.sin,
                  'cos': mpmath.cos,
                  'sinh': mpmath.sinh,
                  'cosh': mpmath.cosh,
                  'exp': mpmath.exp,
                  'log': mpmath.log}
    
    jet_dict = {'sin': sin,
                'cos': cos,
                'sinh': sinh,
                'cosh': cosh,
                'exp': exp,
                'log': log}
    
    if code == 'numpy':
        return numpy_dict[name]
    elif code == 'mpmath':
        return mpmath_dict[name]
    elif code == 'sympy':
        return sympy_dict[name]
    elif code == 'njet':
        return jet_dict[name]
    else:
        return numpy_dict[name] # fall-back to numpy as default
    
def get_package_name(*x):
    '''
    Routine intended to get the package name of a specific object.
    
    Parameters
    ----------
    x: obj
        The object to be examined.
        
    Returns
    -------
    str
        A string denoting the code to be used on the object. 
    '''
    return str(x[0].__class__.__mro__[0].__module__).split('.')[0]
    
def jetfunc(func):
    '''
    General wrapper containing the treatment of input
    which are no njets.
    '''
    def inner(x, **kwargs):
        code = kwargs.get('code', get_package_name(x))
        name = func.__name__
        if code != 'njet':
            return get_function(code, name)(x)
        else:
            # The next lines are always executed,
            # because the built-in functions need to be applied to the
            # 0-th component of the jet:
            x0 = x.array(0)
            code_x0 = get_package_name(x0)
            f = get_function(code_x0, name)
            fx0 = f(x0)
            return func(x, x0=x0, code_x0=code_x0, f=f, fx0=fx0)
    return inner

def zero(x, **kwargs):
    code = kwargs.get('code', get_package_name(x))
    if code == 'numpy':
        return numpy.zeros(x.shape)
    elif code == 'mpmath':
        return x*0
    elif code == 'sympy':
        return x*0
    elif code == 'njet':
        return x*0
    else:
        return x*0

@jetfunc    
def sin(x, **kwargs):
    '''
    Compute the sin of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    s = kwargs.get('fx0')
    c = get_function(kwargs.get('code_x0'), 'cos')(kwargs.get('x0'))
    result = x.__class__(s, n=x.order, graph=[(1, 'sin'), x.graph])
    # compute the derivatives
    result.array = lambda n: [s, c, -s, -c][n%4]
    return result@x

@jetfunc
def cos(x, **kwargs):
    '''
    Compute the cos of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    c = kwargs.get('fx0')
    s = get_function(kwargs.get('code_x0'), 'sin')(kwargs.get('x0'))
    result = x.__class__(c, n=x.order, graph=[(1, 'cos'), x.graph])
    # compute the derivatives
    result.array = lambda n: [c, -s, -c, s][n%4]
    return result@x

@jetfunc
def exp(x, **kwargs):
    '''
    Compute the exponential of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    e = kwargs.get('fx0')
    result = x.__class__(e, n=x.order, graph=[(1, 'exp'), x.graph])
    # compute the derivatives
    result.array = lambda n: e
    return result@x

@jetfunc
def log(x, **kwargs):
    '''
    Compute the natural logarithm of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    ln = kwargs.get('fx0')
    # compute the derivatives
    dx = x.copy().derive()
    drx_arr = (dx/x).get_array()[:-1]
    result = x.__class__(n=x.order, graph=[(1, 'log'), x.graph])
    result.set_array(ln, *drx_arr)
    return result

@jetfunc
def sinh(x, **kwargs):
    '''
    Compute the sinh of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    sh = kwargs.get('fx0')
    ch = get_function(kwargs.get('code_x0'), 'cosh')(kwargs.get('x0'))
    result = x.__class__(sh, n=x.order, graph=[(1, 'sinh'), x.graph])
    # compute the derivatives
    result.array = lambda n: [sh, ch][n%2]
    return result@x

@jetfunc
def cosh(x, **kwargs):
    '''
    Compute the cosh of a jet.

    Parameters
    ----------
    x: jet

    Returns
    -------
    jet
    '''
    ch = kwargs.get('fx0')
    sh = get_function(kwargs.get('code_x0'), 'sinh')(kwargs.get('x0'))
    result = x.__class__(ch, n=x.order, graph=[(1, 'cosh'), x.graph])
    # compute the derivatives
    result.array = lambda n: [ch, sh][n%2]
    return result@x

