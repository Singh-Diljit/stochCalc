# SDE Application

## Background

A stochastic differential equation (SDE) is a differential equation that describes the evolution of a random variable. SDEs are used to model a wide variety of phenomena, including the motion of particles in a fluid, the price of a stock, and the spread of a disease. A system of SDEs consists of a collection of SDEs which may or may not depend on one another.

## Features

This project provides tools for performing select symbolic and numerical calculations on SDEs with multidimensional Wiener processes (as well as a system of such SDEs). Namely, the the following features are included:

* Symbolic application of Ito's formula to SDEs 
* Symbolic and numerical application of the generator of an SDE on a function
* Numerical approximations to SDEs via both Euler-Maruyama and the Milstein method

Note: Ito's formula is a generalized version of Ito's lemma.

## Usage

To use this project, you will need to install the following dependencies:

* SymPy
* NumPy

Once you have installed the dependencies, you can import the project as follows:

```python
from sde import *
```

Alternatively, you can import the SDE class and any functions you are interested in using. The ‘heplerFunctions’ file contains many linear algebra functions which can be useful when dealing with complex transformations or initializing a system of SDEs. In addition, SymPy’s ‘Symbol’ class is critical if you want to perform symbolic calculations.  

## Examples

The following are some examples of how to use this project, for more in-depth use-cases refer to the proper functions documentation (examples can also be found in the appropriate unit tests).

* To define an SDE (or system of SDEs) use the SDE class

```python
#A single SDE with a 1-dimensional Wiener Process 
#Geometric Brownian motion on R:
>>> alpha, r, X = symbols('alpha, r, X')
>>> dX = SDE('X', r*X, alpha*X, var=X)

#A system of SDEs with at-most 1-dimensional Wiener Processes
#Graph of the Ornstein-Uhlenbeck process
>>> theta, mu, sigma, X_0, X, t, x = sp.symbols('theta mu sigma X_0 X t x')
>>> dX = SDE(itoProcess=['X_0', 'X'], drift=[1, theta*(mu-X)], diffusion=[0, sigma], var=[X_0, X])

#5-dimensional Brownian motion
>>> dB = SDE(B, 0, [1,1,1,1,1])
```

* To compute the action of the Graph of the Ornstein-Uhlenbeck process on f(t, x) = sin(t*x):

```python
>>> theta, mu, sigma, X_0, X, t, x = sp.symbols('theta mu sigma X_0 X t x')
>>> dX = SDE(itoProcess=['X_0', 'X'], drift=[1, theta*(mu-X)], diffusion=[0, sigma], var=[X_0, X])
>>> generator(dX, sp.sin(t*x), [t, x])
>>> -0.5*sigma**2*t**2*sin(t*x) + t*theta*(mu - x)*cos(t*x) + x*cos(t*x)
```

* To find the derivative of a 3-Dimensional Bessel Process:

```python
>>> dX = SDE(itoProcess=['B_1', 'B_2', 'B_3'], drift=[0, 0, 0], diffusion=identity(3))
>>> ito(sp.sqrt(x**2 + y**2 + z**2), [t, x, y, z], dX)
>>> d(sqrt(B_1^2 + B_2^2 + B_3^2))
            = [1.0/sqrt(B_1^2 + B_2^2 + B_3^2)]dt
            + [B_1/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_1)
            + [B_2/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_2)
            + [B_3/sqrt(B_1^2 + B_2^2 + B_3^2)]d(B_3)
```

* To approximate geometric Brownian motion on R:

```python
>>> alpha, r, X = symbols('alpha, r, X')
>>> dX = SDE('X', r*X, alpha*X, var=X)
>>> milstein(dX, 10**5, 10**5)
>>> [-0.0226723622013503*alpha + 0.505*r + 1]

#Alternatively:

>>> eulerMaruyama(dX, 10**5, 10**5)
>>> [-0.0332920181711253*alpha + 0.505*r + 1]
```