"""Compute the generator operator of an Ito diffusion acting on a function."""

import helperFunctions as hf
from sympy import simplify

def generator(dX, f, varf, position=None):
    """Return the generator of an Ito diffusion acting on a function.

    One consequence of a stochastic process being an Ito diffusion 
    is the existence of an associated second-order partial 
    differential operator known as the generator (of the diffusion).
    
    For a given Ito diffusion represented by the SDE it satisfies, dX, 
    we compute the generator's action on a given function, f. The user 
    also has the option to evaluate this new function at any point.

    Parameters
    ----------
    dX : SDE
           The SDE satisfied by the Ito diffusion whose generator we wish
           to compute.
    f : Symbolic expression e.g. sp.sin(x), 0, x**4, [x*y, z, t]
           This is the function being acted on by the generator.
    varf : Non-empty ordered iterable of SymPy 'symbols'
           The first member of 'varf' is assumed to represent time. 
           In general, the order should correspond to the ordering of variables
           in 'dX.var', if dX.var is empty, ordering does not matter beyond
           the 'position' input.
    position : Ordered iterable of numeric classes, optional
           If this input is 'None' (the default setting) or an empty
           iterable (e.g. [], ()) a new process is not initiated 
           and a symbolic expression is returned. Otherwise,
           the input should be an ordered iterable with entries
           corresponding to the ordering of 'varf'. The input can also be a
           numeric if the domain has dimension 1.

    Returns
    -------
    Af : Symbolic expression or numeric (if position input given)
           The generator applied to 'f'. This new function is 
           evaluated at a particular position if a positional
           input is provided.

    Example(s)
    ----------
    Graph of the Ornstein-Uhlenbeck process on f(t, x) = sin(t*x)
    
    >>> theta, mu, sigma, X_0, X, t, x = symbols('theta mu sigma X_0 X t x')
    >>> dX = SDE(['X_0', 'X'],
                 [1, theta*(mu-X)],
                 [0, sigma],
                 var=[X_0, X])
    >>> f = sin(t*x)
    >>> generator(dX, f, [t, x])
    >>> -0.5*sigma**2*t**2*sin(t*x) + t*theta*(mu - x)*cos(t*x) + x*cos(t*x)
    
    """
    #Standardize the drift and diffusion of dX
    drift = hf.changeVar(dX.drift, dX.var, varf)
    diffusion = hf.changeVar(dX.diffusion, dX.var, varf)
    
    #Compute useful first and second order derivatives of f
    gradF = hf.gradient(f, varf)
    Hf = hf.hessian(f, varf)

    #Compute the "Frobenius inner product" of two matrices
    symDiff = hf.symMatrix(diffusion)
    frobInner = hf.hadamardMul(symDiff, Hf)

    #Compute the generator applied to f
    Af = (hf.dot(drift, gradF)
          + 1/2 * hf.componentSum(frobInner))
        
    #Evalaute the (new) function at a point
    if position != None: #allows position = 0
        Af = hf.evalFunc(Af, varf, position)

    return Af
