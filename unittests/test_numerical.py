"""Numerical unittesting"""

import unittest
from sympy import symbols
import sys
sys.path.append('../StochCalc')
from sde import SDE
import helperFunctions as hf

def avgProcedure(dX, sims, trials):
    """The averaging procedure used in 'eulerMaruyama' and 'milstein'.

    Parameters
    ----------
    dX : SDE
        SDE or system of SDEs to be numerically solved.
    sims : int
        Number of samples to be averaged
    trials : list of lists
        The numbers to be sampled; 'trials[i]' is the samples
        for iteration 'i'.

    Returns
    -------
    seed : list
        A list of floats where 'seed[i]' is the average of
        the 'tr[i] for tr in trials'.

    Example(s)
    ----------
    >>> t = sp.Symbol('t')
    >>> f = t**2 + t
    >>> dX = SDE('X', 5, f)
    >>> trials = [[1], [4], [7]]
    >>> avgProc(dX, sims, trials)
    >>> [4.0]

    """
    res = [0] * len(dX.initial)
    for k in range(sims):
        trial = trials[k]
        res = [res[i] + tr/sims for i, tr in enumerate(trial)]
        
    return res

def updateComponents(dX, steps, seed, vec=False):
    """Drift/Diffusion proccess used in 'eulerMaruyama' and 'milstein'.

    Parameters
    ----------
    dX : SDE
        SDE or system of SDEs to be numerically solved.
    steps : int
        Fineness of discretization for time interval.
    seed : list
        The current best approximation of the process.

    Returns
    -------
    drft_ : float
        The drift of the current approximation.
    diff_ : list
        The diffusion of the current approximation.

    Example(s)
    ----------
    >>> t = sp.Symbol('t')
    >>> f = t**2 + t
    >>> dX = SDE('X', f, 10)
    >>> seed = [10]
    >>> updateDrift(dX, steps, seed)
    >>> 

    """
    dt = dX.lenTime / steps
    start, end = 0, steps
    mid = (start + end) // 2
    drft = [0] * 3
    diff = [0] * 3
    pts = [start, mid, end]

    for i in range(len(pts)):
        x = pts[i]
        t = dX.startTime + x*dt

        drft[i] = hf.evalFunc(dX.drift[0], dX.var, [t] + seed)
        diff[i] = hf.evalFunc(dX.diffusion[0], dX.var, [t] + seed, toVec=vec)

    return drft, diff

def isClose(a, b, error=10**(-10)):
    """Return if two numerics are within a distance of one another.

    This function will be used to return if two floats are essentially
    equal.

    """
    return True if abs(a-b) <= error else False

class simulationTest(unittest.TestCase):
    """Test determinitic aspects of the 'eulerMaruyama' and 'milstein' functions."""

    def test_averaging(self):
        """Test the averaging procedure."""
        
        dX = SDE('X', 5, 7)
        trials = [[1], [4], [7]]
        self.assertEqual([4], avgProcedure(dX, 3, trials))

        dY = SDE('X', 5, 4, initialVal=[0, 0])
        trials = [[1, 2], [4, 3], [7, 4]]
        self.assertEqual([4, 3], avgProcedure(dY, 3, trials))

    def test_updateComp(self):
        """Test the update component procedure."""

        t, x, y = symbols('t x y')
        dX = SDE('X', t**2 + x, y, timeInterval=[0,2], var=[t,x,y])

        seed = [5]
        drft, diff = updateComponents(dX, 5, seed)

        drftAns = [[5], [5.64], [9]]
        diffAns = [[y],[y],[y]]

        self.assertTrue(all(isClose(a[0], b[0]) for a, b in zip(drft, drftAns)))
        self.assertTrue(all(isClose(a[0], b[0]) for a, b in zip(diff, diffAns)))

        t, x, y = symbols('t x y')
        dX = SDE('X', [t**2 + x, 5, x**2], [y, t, 1],
                 timeInterval=[0,2], var=[t,x,y])

        seed = [1, 2, 3]
        drft, diff = updateComponents(dX, 6, seed)

        drftAns = [[1], [2], [5]]
        diffAns = [[2],[2],[2]]

        self.assertTrue(all(isClose(a[0], b[0]) for a, b in zip(drft, drftAns)))
        self.assertTrue(all(isClose(a[0], b[0]) for a, b in zip(diff, diffAns)))


if __name__ == "__main__":
    unittest.main()
