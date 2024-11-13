"""Generator Unittests"""

import unittest
from sympy import cos, exp, pi, sin, simplify, Symbol, symbols
import sys
sys.path.append('../StochCalc')
from generator import generator
from sde import SDE

class GeneratorTest(unittest.TestCase):
    """Class to test aspects of the generator function."""

    def test_general(self):
        """Test of a few classic examples."""

        #Standard 1-Dim Brownian motion
        dB = SDE('B', 0, 1)
        x, n = symbols('x n')
        f = x**n

        #For standard Brownian motion the generator is
        #1/2 the Laplace operator (of apprioprate dimension)
        f_xx = n*(n-1) * x**(n-2) 
        self.assertEqual(1/2 * f_xx, simplify(generator(dB, f, [x])))

        #Ornstein-Uhlenbeck process on R
        theta, mu, sigma, X_t = symbols('theta mu sigma X_t')
        dX_t = SDE('X_t', theta*(mu-X_t), sigma, var=[X_t])
        x = Symbol('x')
        n = 5
        f = x**n
        f_x, f_xx = n * x**(n-1), n*(n-1) * x**(n-2)
        gen_f = theta * (mu-x) * f_x + .5 * sigma**2 * f_xx

        self.assertEqual(gen_f, generator(dX_t, f, [x]))

        #Geometric Brownian motion on R
        alpha, r, X, w = symbols('alpha, r, X, w')
        dX = SDE('X', r*X, alpha*X, var=X)
        f = exp(w)

        self.assertEqual(r*w*exp(w) + .5*alpha**2 * w**2 * exp(w),
                         generator(dX, f, w))
        

    def test_multiDimProc(self):
        """Test the generator applied to systems of Ito Diffusions."""

        #Graph of Brownian Motion
        dY = SDE(['Y_1', 'Y_2'], [1, 0], [0, 1])
        t, x = symbols('t x')
        n = 27
        f = sin(t) + x**n
        gen_f = cos(t) + .5 * n*(n-1) * x**(n-2)
        self.assertEqual(gen_f, generator(dY, f, [t, x]))

        #Graph of Ornstein-Uhlenbeck process
        theta, mu, sigma, X_0, X = symbols('theta mu sigma X_0, X')
        dX = SDE(['X_0', 'X'], [1, theta*(mu-X)], [0, sigma], var=[X_0, X])
        t, x = symbols('t x')
        f = t**5 + x**6 + sin(t*x)

        ans = (0.5*sigma**2*(-t**2*sin(t*x) + 30*x**4)
               + 5*t**4 + x*cos(t*x)
               + theta*(mu - x)*(t*cos(t*x) + 6*x**5))
        
        self.assertEqual(ans, generator(dX, f, [t, x]))
        
    def test_position(self):
        """Test evalution of generator at fixed points."""
        
        dX = SDE(['X'], 1, 1)
        t, u, v = symbols('t u v')
        f = sin(u)*v + v
        self.assertEqual([0], generator(dX, f, [u, v], (0,0)))

        dY = SDE('Y', sin(t), cos(t))
        self.assertEqual([pi*sin(t)],
                         generator(dY, f, [u, v], position=[0, pi]))

        #Ornstein-Uhlenbeck process on R
        theta, mu, sigma, X_t = symbols('theta mu sigma X_t')
        dX_t = SDE('X_t', theta*(mu-X_t), sigma, var=[X_t])
        x = symbols('x')
        n = 5
        f = x**n
        f_x, f_xx = n * 7**(n-1), n*(n-1) * 7**(n-2)
        gen_f = theta * (mu-7) * f_x + .5 * sigma**2 * f_xx

        self.assertEqual([gen_f], generator(dX_t, f, [x], position=7))
        

if __name__ == "__main__":
    unittest.main()


