"""Numerical solutions to systems of SDEs."""

import helperFunctions as hf
from sde import SDE

def eulerMaruyamaSimulation(dX, steps):
    """Perform one simulation of the Euler-Maruyama method.

    An extension of Euler's method for solving ODEs, the
    Euler-Maruyama (EM) method provides a Markov chain
    approximation for systems of SDEs. In its most general
    form the EM approximation is given by the recursive formula:

        Y^i_{n+1} =  Y^i_n
                  +  u^i(t_n, Y_n)*(t_n - t_{n-1})
                  +  sum_j V_{i, j}(t_n, Y_n)W^j_n

    where i is the ith component of the Ito process,
    u is the drift vector, V is the diffusion matrix,
    t_n is an element from the discretization of the
    time interval, and W_n is a vector of samples from
    the appropriate dimensional Wiener process.
    
    In this implementation of EM, components are not
    iterated simultaneously, instead each component found
    is used in the computation of any remaining components
    in that iteration. This increases accuracy without affecting
    the number of computations required.

    Parameters
    ----------
    dX : SDE
        SDE or system of SDEs to be solved.
    steps : int
        The 'fineness' of the discretization. The larger
        the number of steps the better the approximation
        and slower the program runs. 

    Returns
    -------
    seed : list of floats
        Approximation of a solution to the SDE, or systems of SDEs.

    """
    dt = dX.lenTime / steps
    seed = [x for x in dX.initial]
    
    for i in range(0, steps + 1):
        #Iterate EMM
        dW_ = hf.dW(dt, dX.dimDiffusion)
        t = dX.startTime + i*dt
        
        for k in range(dX.enumSystem):
            #Compute components of the new approximation
            drft_ = hf.evalFunc(dX.drift[k], dX.var, [t] + seed)[0]
            diff_ = hf.evalFunc(dX.diffusion[k], dX.var, [t] + seed)
            seed[k] += (drft_ * dt
                        + hf.dot(diff_, dW_))
            
    return seed

def eulerMaruyama(dX, steps, sims):
    """Average simulations the EM method on systems of SDEs.

    Example(s)
    ---------
    Approximating geometric Brownian motion on R

    >>> alpha, r, X = symbols('alpha, r, X')
    >>> dX = SDE('X', r*X, alpha*X, var=X)
    >>> eulerMaruyama(dX, 10**5, 10**5)
    >>> [-0.0332920181711253*alpha + 0.505*r + 1]

    """
    res = [0] * len(dX.initial)
    for _ in range(sims):
        trial = eulerMaruyamaSimulation(dX, steps)
        res = [res[i] + tr/sims for i, tr in enumerate(trial)]
        
    return res

def MilsteinSimulation(dX, steps):
    """Perform one simulation of the Milstein method.

    Similar to Euler-Maruyama, the Milstein method provides a
    Markov chain approximation for systems of SDEs. In its most
    general form the Milstein approximation is given by the
    recursive formula:

        Y^i_{n+1} =  Y^i_n
                  +  u^i(t_n, Y_n) * (t_n - t_{n-1})
                  +  sum_j V_{i, j}(t_n, Y_n) W^j_n
                  +  1/2 * sum_j J^i(t_n, Y_n) W^j_n

    where i is the ith component of the Ito process,
    u is the drift vector, V is the diffusion matrix,
    t_n is an element from the discretization of the
    time interval, W_n is a vector of samples from
    the appropriate dimensional Wiener process, and
    J is the time-restricted Jacobian matrix of the diffusion.

    In this implementation of the Milstein method, components
    are not iterated simultaneously, instead each component found
    is used in the computation of any remaining components
    in that iteration. This increases accuracy without
    extra computations and minimal overhead.
    
    Parameters
    ----------
    dX : SDE
        SDE or system of SDEs to be solved.
    steps : int
        The 'fineness' of the discretization. The larger
        the number of steps the better the approximation
        and slower the program funs. 

    Returns
    -------
    seed : list of floats
        Approximation of a solution to the SDE, or systems of SDEs.
        
    """
    dt = dX.lenTime / steps
    seed = [x for x in dX.initial]

    #Differential operators provide a means
    #to approximate the second order Taylor term
    vecDiffusion = [sum(dX.diffusion[i])
                        for i in range(dX.enumSystem)]
    jacDiffusion = hf.jacobian(vecDiffusion, dX.var[1:])
    diffOps = hf.matrixMul(jacDiffusion, dX.diffusion)
    
    for i in range(0, steps + 1):
        """Iterate the Milstein method."""
        dW_ = hf.dW(dt, dX.dimDiffusion)
        t = dX.startTime + i*dt
        
        #Double integration against the same process is computed
        higherOrd = [dw**2 - dt for dw in dW_]
        
        for k in range(dX.enumSystem):
            """Compute components of the new approximation."""
            drft_k = hf.evalFunc(
                        dX.drift[k], dX.var, [t] + seed)[0]
            diff_k = hf.evalFunc(
                        dX.diffusion[k], dX.var, [t] + seed,
                        toVec=True)
            calledDiffOps = [hf.evalFunc(f, dX.var, [t] + seed)[0]
                             for f in diffOps[k]]

            seed[k] += (drft_k * dt
                       + hf.dot(diff_k, dW_)
                       + 1/2 * hf.dot(higherOrd, calledDiffOps))
            
    return seed

def milstein(dX, steps, sims):
    """Average simulations of the Milstein method on systems of SDEs.

    Example(s)
    ---------
    Approximating geometric Brownian motion on R

    >>> alpha, r, X = symbols('alpha, r, X')
    >>> dX = SDE('X', r*X, alpha*X, var=X)
    >>> milstein(dX, 10**5, 10**5)
    >>> [-0.0226723622013503*alpha + 0.505*r + 1]


    """
    res = [0] * len(dX.initial)
    for _ in range(sims):
        trial = MilsteinSimulation(dX, steps)
        res = [res[i] + tr/sims for i, tr in enumerate(trial)]
        
    return res
