"""Unittesting for Ito"""

import helperFunctions as hf
import unittest
from ito import ito
from sde import SDE
from sympy import cos, sin, sqrt, symbols

class itoTest(unittest.TestCase):
    """Test Ito's formula."""

    def test_itoLemma(self):
        """Test Ito's Formula for single dimensional processes (Ito's Lemma)."""

        t, x, X, S, C = symbols('t x X S C')
        dX = SDE('X', 0, 1)
        f = x**3
        dfX = ito(f, [t, x], dX)
    
        dY = SDE('f(X)', 3.0*X, 3*X**2)

        dS = SDE('S', 0, 1)
        f = 5
        dfS = ito(f, [t, S], dS)

        dR = SDE('R', 0, 0)

        dC = SDE('C', sin(t), 5+C**2)
        f = t**2 + t*sin(x)
        dfC = ito(f, [t, x], dC)
        
        dA = SDE('R',
                 [-0.5*t*(C**2 + 5)**2*sin(C) + t*sin(t)*cos(C) + 2*t + sin(C)],
                 [[t*(C**2 + 5)*cos(C)]])
        
        self.assertEqual((dY.drift, dY.diffusion), (dfX.drift, dfX.diffusion))
        self.assertEqual((dR.drift, dR.diffusion), (dfS.drift, dfS.diffusion))
        self.assertEqual((dA.drift, dA.diffusion), (dfC.drift, dfC.diffusion))

    def test_itoFormula(self):
        """Test Ito's Formula for multidimensional processes."""
        
        dX = SDE(itoProcess=['B_1', 'B_2', 'B_3'],
                 drift=[0, 0, 0], diffusion = hf.identity(3))
        t, x, y, z = symbols('t, x, y, z')
        f = sqrt(x**2 + y**2 + z**2)
        fVars = [t, x, y, z]
        dfX = ito(f, fVars, dX)

        B_1, B_2, B_3 = symbols('B_1, B_2, B_3')
        dY = SDE(itoProcess = ['sqrt(B_1^2 + B_2^2 + B_3^2)'],
                      drift = [1.0/sqrt(B_1**2 + B_2**2 + B_3**2)],
                      diffusion = [[B_1/sqrt(B_1**2 + B_2**2 + B_3**2),
                                    B_2/sqrt(B_1**2 + B_2**2 + B_3**2),
                                    B_3/sqrt(B_1**2 + B_2**2 + B_3**2)]])
        
        self.assertEqual(dY, dfX)


if __name__ == "__main__":
    unittest.main()
