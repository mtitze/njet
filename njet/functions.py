# Collection of elementary functions whose derivatives are known to any order.

import numpy
import sympy

def detect_code(x, name):
    # Find the appropriate code to perform operations.
    # We shall assume that the code labels are identical (otherwise we may need to
    # implement a dictionary translating between the various names).
    
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
    
    jet_dict = {'sin': sin,
                'cos': cos,
                'sinh': sinh,
                'cosh': cosh,
                'exp': exp,
                'log': log}

    if hasattr(x, '__module__'):
        if x.__module__[0:5] == 'sympy':
            func = sympy_dict[name]
        elif x.__module__[0:4] == 'njet':
            func = jet_dict[name]
        else:
            raise NotImplementedError('Unknown object.')
    else:
        func = numpy_dict[name]
        
    return func

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='sin'))
    func2 = kwargs.get('code', detect_code(x0, name='cos'))
    s = func(x0)
    c = func2(x0)
    result = x.__class__(s, n=x.order, graph=[(1, 'sin'), x.graph])

    # compute the derivatives
    result.array = lambda n: [s, c, -s, -c][n%4]
    return result.compose(x)

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='sin'))
    func2 = kwargs.get('code', detect_code(x0, name='cos'))
    s = func(x0)
    c = func2(x0)
    result = x.__class__(c, n=x.order, graph=[(1, 'cos'), x.graph])

    # compute the derivatives
    result.array = lambda n: [c, -s, -c, s][n%4]
    return result.compose(x)

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='exp'))
    e = func(x0)
    result = x.__class__(e, n=x.order, graph=[(1, 'exp'), x.graph])

    # compute the derivatives
    result.array = lambda n: e
    return result.compose(x)

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='log'))
    ln = func(x0)
    graph=[(1, 'log'), x.graph]

    # compute the derivatives
    dx = x.copy().derive()
    drx_arr = (dx/x).get_array()[:-1]
    result = x.__class__(n=x.order, graph=graph)
    result.set_array([ln] + drx_arr)
    return result

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='sinh'))
    func2 = kwargs.get('code', detect_code(x0, name='cosh'))
    sh = func(x0)
    ch = func2(x0)
    result = x.__class__(sh, n=x.order, graph=[(1, 'sinh'), x.graph])

    # compute the derivatives
    result.array = lambda n: [sh, ch][n%2]
    return result.compose(x)

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
    x0 = x.array(0)
    func = kwargs.get('code', detect_code(x0, name='sinh'))
    func2 = kwargs.get('code', detect_code(x0, name='cosh'))
    sh = func(x0)
    ch = func2(x0)
    result = x.__class__(ch, n=x.order, graph=[(1, 'cosh'), x.graph])

    # compute the derivatives
    result.array = lambda n: [ch, sh][n%2]
    return result.compose(x)

