"""Implement Ito's formula."""

import helperFunctions as hf
from sde import SDE

def ito(f, currVars, dX):
    """Perform the chain rule on a multidimensional Ito process.

    Ito's formula states a sufficiently smooth transformation
    of an Ito process is again an Ito process and that the
    transformed process, Y = f(t, X), satisfies the SDE:
    
        dY_i = df_i/dt(t, X)dt
             + sum_j df_i/dx_j(t, X)dX_j
             + 1/2 sum_{j, k} d^2f_i/(dx_jdx_k)(t, X)(dX_j)(dX_k)

    Parameters
    ----------
    f : Symbolic expression e.g. sp.sin(x), 0, x**4, [x*y, z, t]
            A twice-differentiable function with domain having dimension:
            1 + M where M is the dimension of Wiener process. This function 
            applied to the process, X, gives the process whose SDE will
            be returned.
    currVars : Non-empty ordered iterable of SymPy 'symbols'
            The first member of 'currVars' is assumed to represent time. 
            The other entries should correspond to how the Wiener
            process is transformed.
    dX : SDE
            SDE of the Ito process being transformed by 'f'.

    Returns
    -------
    df : SDE
           The SDE satisfied by 'f(t, X)'.

    Notes
    -----
    A dummy variable indicating time as a variable is considered should
    always be the first component in 'currVars'. If the action of 'f' on
    'dX' is independent of time this is still the case. If we want to
    compute the derivitive of Y = (W_t)^3 then: dX = dW_t written as
    (dX = SDE('W', 0, 1)), f = x**3, and currVars = [t, x] (or [y, x] where
    y is any sympy symbol besides 'x').

    Example(s)
    ----------
    3-Dimensional Bessel Process

    >>> dX = SDE(itoProcess=['B_1', 'B_2', 'B_3'],
                drift=[0, 0, 0], diffusion = identity(3))
    >>> f = sp.sqrt(x**2 + y**2 + z**2)
    >>> fVars = [t, x, y, z]
    >>> ito(f, fVars, dX)
    >>> d(sqrt(B_1^2 + B_2^2 + B_3^2))
            = [1.0/sqrt(B_1^2 + B_2^2 + B_3^2)]dt
            + [B_1/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_1)
            + [B_2/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_2)
            + [B_3/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_3)
         
    """ 
    #Rewrite f in terms of the Ito process
    varf = hf.strToSymbol(['t'] + dX.process)
    f = hf.changeVar(f, currVars, varf)
    imDim = len(f)
    
    #Compute required first and second derivatives
    Jt = hf.jacobian(f, varf[0])
    Jx = hf.jacobian(f, varf[1:])
    Hx = hf.vectorHessian(f, varf[1:])

    #Compute matrices related to the diffusion
    diffusionTranspose = hf.transpose(dX.diffusion)
    gramDiffusion = hf.gramMatrix(dX.diffusion)
    
    secondOrder = [
        hf.componentSum(hf.hadamardMul(Hx[i], gramDiffusion))
        for i in range(imDim)]
    
    #Compute the drift of the new process
    deterministic = [Jt[i][0]
                     + hf.dot(Jx[i], dX.drift)
                     + 1/2 * secondOrder[i] for i in range(imDim)]

    #Compute the diffusion of the new process
    stochastic = [0] * imDim
    for i in range(imDim):
        stochastic[i] = [hf.dot(Jx[i], diffusionTranspose[j])
                         for j in range(dX.dimDiffusion)]

    Y = [hf.rewrite(f_i) for f_i in f]
    df = SDE(Y, deterministic, stochastic)
    
    return df
