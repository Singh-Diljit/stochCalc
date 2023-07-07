"""Unittesting for helperFunctions.py"""

import unittest
from helperFunctions import changeForm, changeVar, evalFunc, makeProcess
from sympy import cos, pi, sin, symbols

class changeFormTest(unittest.TestCase):
    """Test of 'changeForm' function."""

    def test_FromScaler(self):
        """Test how 'changeForm' handles conversion when input is a scaler."""

        #Scaler -> Scaler
        self.assertEqual(0, changeForm(0, toScaler=True))
        self.assertEqual(0, changeForm(0, toMatrix=False))

        #Scaler -> Vector
        self.assertEqual([0], changeForm(0, toVector=True))
        self.assertEqual([1], changeForm(1, toVector=True))

        #Scaler -> Matrix
        self.assertEqual([[0]], changeForm(0))
        self.assertEqual([[2]], changeForm(2))
        self.assertEqual([[.10]], changeForm(.10, toMatrix=True))

    def test_FromVector(self):
        """Test how 'changeForm' handles conversion when input is a vector."""

        #Vector -> Scaler
        self.assertEqual(0, changeForm([], toScaler=True))
        self.assertEqual(0, changeForm([0], toScaler=True))
        self.assertEqual(1, changeForm([1, 2], toScaler=True))
        
        #Vector -> Vector
        self.assertEqual([0], changeForm([], toVector=True))
        self.assertEqual([1, 2], changeForm([1, 2], toVector=True))
        self.assertEqual([1, 2], changeForm([1, 2], toMatrix=False))

        #Vector -> Matrix
        self.assertEqual([[0]], changeForm([]))
        self.assertEqual([[1], [2]], changeForm([1, 2]))
        self.assertEqual([[1], [2], [3]], changeForm([1, 2, 3], toMatrix=True))

    def test_FromMatrix(self):
        """Test how 'changeForm' handles conversion when input is a matrix."""

        #Matrix -> Scaler
        self.assertEqual(0, changeForm([[]], toScaler=True))
        self.assertEqual(0, changeForm([[0]], toScaler=True))
        self.assertEqual(1, changeForm([[1, 2, 3]], toScaler=True))
        self.assertEqual(1, changeForm([[1], [2], [3]], toScaler=True))
        self.assertEqual(1, changeForm([[1, 2], [3, 4], [5, 6]],
                                         toScaler=True))

        #Matrix -> Vector
        self.assertEqual([0], changeForm([[]], toVector=True))
        self.assertEqual([2], changeForm([[2]], toVector=True))
        self.assertEqual([1, 2, 3],
                         changeForm([[1, 2, 3]], toVector=True))
        self.assertEqual([1, 2, 3],
                         changeForm([[1], [2], [3]], toVector=True))
        self.assertEqual([1, 3, 5],
                         changeForm([[1, 2], [3, 4], [5, 6]], toVector=True))

        #Matrix -> Matrix
        self.assertEqual([[0]], changeForm([[]]))
        self.assertEqual([[1, 2]], changeForm([[1, 2]]))
        self.assertEqual([[1, 2], [3, 4]],
                         changeForm([[1, 2], [3, 4]], toMatrix=False))

class changeVarTest(unittest.TestCase):
    """Test of 'changeVar' function."""

    def test_emptyVars(self):
        """Test if currVars and/or newVars are '[]'."""

        #Declare vairables and common functions
        t, x, y, z = symbols('t x y z')
        s = sin(t)
        f = [x**2 - t, y*s] #uses all symbols but 'z'
        g = x*y/z #uses all symbols but 't'
        h = [t**2-t, x*s] #f with x -> t
        
        #currVars = [], newVars != []
        self.assertEqual(f, changeVar(f, [], [t])) #swap with used var
        self.assertEqual(f, changeVar(f, [], [z])) #swap with unused var
        self.assertEqual([g], changeVar(g, [], [t])) #output is list
        self.assertEqual([4], changeVar(4, [], [z])) #output is list

        #currVars != [], newVars = []
        self.assertEqual(f, changeVar(f, [t], [])) #swap with used var
        self.assertEqual(f, changeVar(f, [z], [])) #swap unused var
        self.assertEqual([g], changeVar(g, [t], [])) #output is list
        self.assertEqual([4], changeVar(4, [z], [])) #output is list

        #currVars = [], newVars = []
        self.assertEqual(f, changeVar(f, [], []))
        self.assertEqual([g], changeVar(g, [], [])) #output is list
        self.assertEqual([4], changeVar(4, [], [])) #output is list

    def test_inputType(self):
        """Test varying input types pairings for currVars and newVars."""

        #Declare vairables and common functions
        t, x, y, z = symbols('t x y z')
        s = sin(t)
        f = [x**2 - t, y*s] #uses all symbols but 'z'
        g = x*y/z #uses all symbols but 't'
        h = [t**2-t, y*s] #f with x -> t

        #currVars is only non-listed input
        self.assertEqual(f, changeVar(f, t, []))
        self.assertEqual(h, changeVar(f, x, [t]))
        self.assertEqual([4], changeVar(4, x, [t]))
        
        #newVars is only non-listed input
        self.assertEqual(f, changeVar(f, [], t))
        self.assertEqual(h, changeVar(f, [x], t))
        self.assertEqual([4], changeVar(4, [x], t))

        #newVars and currVars are not lists
        self.assertEqual([t*y/z], changeVar(g, x, t))
        self.assertEqual([4], changeVar(4, x, t))

    def test_lenInput(self):
        """Test cases when len(currVars) != len(newVars)."""

        #Declare vairables and common functions
        t, x, y, z = symbols('t x y z')
        s = sin(t)
        f = [x**2 - t, y*s] #uses all symbols but 'z'
        h = [t**2-t, y*s] #f with x -> t

        #len(currVars) < len(newVars)
        self.assertEqual(h, changeVar(f, x, [t, z]))
        self.assertEqual(h, changeVar(f, x, [t, x, z]))
        self.assertEqual([4], changeVar(4, [x, y, x], [t]))

        #len(currVars) > len(newVars)
        self.assertEqual([z**2 - t, y*s], changeVar(f, [x, t], [z]))
        self.assertEqual(h, changeVar(f, [x, y, x], [t]))
        self.assertEqual([6], changeVar(6, [x, y, x], [t]))
        
    def test_numericSwap(self):
        """Test evaluation done by swapping vars with numerics."""
        
        #Declare vairables and common functions
        t, x, y, z = symbols('t x y z')
        s = sin(t)
        f = [x**2 - t, y*s] #uses all symbols but 'z'
        g = x*y/z #uses all symbols but 't'
        h = [t**2-t, y*s] #f with x -> t

        self.assertEqual([16 - t, y*s], changeVar(f, x, 4))
        self.assertEqual([.5], changeVar(g, [x, y, z], [1, 2, 4]))
        self.assertEqual([0], changeVar(s, t, pi))

    def test_swappingFuncs(self):
        """Test computation of composition of functions."""
        
        #Declare vairables and common functions
        t, x, y, z = symbols('t x y z')
        s = sin(t)
        f = [x**2 - t, y*s] #uses all symbols but 'z'
        g = x*y/z #uses all symbols but 't'
        h = [t**2-t, y*s] #f with x -> t

        self.assertEqual([sin(cos(t))], changeVar(s, t, cos(t)))
        self.assertEqual([s/2], changeVar(g, [x,y,z], [s, 2, 4]))
        
if __name__ == "__main__":
    unittest.main()
