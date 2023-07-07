"""A collection of functions used to aide complicated processes."""

import sympy as sp
from numpy.random import normal

#Functions for performing linear algebra on lists
def transpose(M):
    """Return the transpose of a matrix.
    
    Parameters
    ----------
    M : list of lists
        Matrix to be transposed.

    Returns
    -------
    transpose(M) : list of lists
        The transpose of M.
        
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> transpose(A)
    >>> [[1, 4], [2, 5], [3, 6]]
    
    """
    return list(map(list, zip(*M)))

def dot(u, v):
    """Return the dot product of two vectors.
     
    Parameters
    ----------
    u, v : list
        Vectors whose dot-product will be taken.

    Returns
    -------
    dot(u, v) : sympy 'Symbol', sympy 'Function', numeric(float, int)
        The dot product of 'u' and 'v'.
        
    Notes
    -----
    Symbolic computations are supported, but to simplify the end 
    result sympy's 'sp.simplify' must be invoked. However simplfication
    is slow thus not blanketly implemented into 'dot(u, v)'.
        
        
    Example(s)
    ----------
    >>> a = [1, 2, 3]
    >>> b = [4, 5, 6]
    >>> dot(a, b)
    >>> 32
    
    >>> t = sp.Symbol('t')
    >>> a = [sp.sin(t), sp.cos(t)]
    >>> dot(a, a)
    >>> sin(t)**2 + cos(t)**2
    
    """
    return sum([u_i*v[i] for i, u_i in enumerate(u)])

def matrixMul(A, B):
    """Return the (standard) product of two matrices.
        
    Parameters
    ----------
    A, B : list of lists
        Matrices to be multiplied.

    Returns
    -------
    product : list of lists
        The matrix product of 'A' and 'B'.
        
    Notes
    -----
    For simplification of symbolic expression see documentation
    for 'dot' function.
        
    Example(s)
    ----------
    >>> t = sp.Symbol('t')
    >>> A = [[1, 2, 3], [4, 5, sp.sin(t)]]
    >>> B = [[1, 2], [3, 4], [5, 6]]
    >>> matrixMul(A, B)
    >>> [[22, 28], [5*sin(t) + 19, 6*sin(t) + 28]]

    """
    product = [[dot(rowA, rowB) for rowB in transpose(B)]
               for rowA in A]    
    return product

def hadamardMul(A, B):
    """Return the entrywise product of two matrices.
    
    Parameters
    ----------
    A, B : list of lists
        Matrices to be entrywise multiplied.

    Returns
    -------
    product: list of lists
        The Hadamard product of 'A' and 'B'.
    
    Notes
    -----
    Communative operation.
        
    Example(s)
    ----------
    >>> t = sp.Symbol('t')
    >>> A = [[1, 2, 3], [4, 5, sp.sin(t)]]
    >>> B = [[10, 20, 4], [1, 2, 4]]
    >>> hadamardMul(A, B)
    >>> [[10, 40, 12], [4, 10, 4*sin(t)]]
    
    """
    product = [[a*b for a, b in zip(rowA, rowB)]
               for rowA, rowB in zip(A, B)]
    return product

def vectorMul(u):
    """Return the product of all entries in a list.
    
    Parameters
    ----------
    u : list
        Vector whose entries will be multiplied.

    Returns
    -------
    vectorMul(u): sympy 'Symbol', sympy 'Function', numeric(float, int)
        The product of all entries in 'u'.
    
    Example(s)
    ----------
    >>> x, t = sp.symbols('x t')
    >>> u = [10, 2, x, sp.sin(t)]
    >>> vectorMul(u)
    >>> 20*x*sin(t)
    
    """
    product = 1
    for entry in u:
        product *= entry
        
    return product

def rowProduct(M):
    """Return the row-product vector of a matrix.

    The row product of a matrix reduces each row to
    just the product of its elements resulting in a
    column matrix.
        
    Parameters
    ----------
    M : list of lists
        Matrix whose row product will be computed.

    Returns
    -------
    rowProduct(M) : list of lists
        Column matrix with ith column being the product of entries in M[i].
        
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> rowProduct(A)
    >>> [[6], [120]]
    
    """
    return [vectorMul(row) for row in M]
    
def componentSum(M):
    """Return the sum of every element in a matrix.
    
    Parameters
    ----------
    M : list of lists
        Matrix whose components will be summed.

    Returns
    -------
    componentSum(M) : sympy 'Symbol', sympy 'Function', numeric(float, int)
        The sum of all entries in 'M'.
        
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> componentSum(M)
    >>> 21
    
    """
    return sum([sum(row) for row in M])

def gramMatrix(M):
    """Return M^T * M.
    
    Parameters
    ----------
    M : list of lists
        Matrix to be transformed by the Gram matrix procedure.

    Returns
    -------
    gramMatrix(M): list of lists
        The matrix given by 'transpose(M)' times M.
    
    Notes
    -----
    The Gram matrix of a matrix, 'A', is given by the product of A* by A
    where A* is the conjugate transpose of A. For this function the just
    the transpose is taken since entries are assumed to be real. 
        
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> gramMatrix(M)
    >>> [[17, 22, 27], [22, 29, 36] [27, 36, 45]]
    
    """
    return matrixMul(transpose(M), M)

def symMatrix(M):
    """Return M * M^T.
    
    Parameters
    ----------
    M : list of lists
        Matrix to be transformed by the symetric matrix procedure.

    Returns
    -------
    symMatrix(M): list of lists
        The matrix given by (M) times 'transpose(M)'.
    
    Notes
    -----
    This is the transpose of the Gram matrix transformation. 
    For square matrices 'symMatrix(M)' == 'gramMatrix(M)'.  
        
    Example(s)
    ----------
    >>> A = [[1, 2], [4, 5], [7, 8]]
    >>> symMatrix(M)
    >>> [[5, 14, 23], [14, 41, 68] [23, 68, 113]]
    
    """
    return matrixMul(M, transpose(M))
    
def identity(n):
    """Return the identity matrix of a given dimension.
    
    Parameters
    ----------
    n : int
        The number of rows (and cols) of the identity matrix.

    Returns
    -------
    identity(n) : list of lists
        The n by n indentity matrix.

    Example(s)
    ----------
    >>> n = 2
    >>> identity(2)
    >>> [[1, 0], [0, 1]]
    
    """
    return [[int(k == i) for k in range(n)] for i in range(n)]
    
def isMatrix(cand):
    """Return if the input is a matrix (list of lists).
    
    Parameters
    ----------
    cand : *
        Entry to be tested, verifying if it is a matrix (list of lists).

    Returns
    -------
    res : bool
        Returns if 'cand' is a matrix (list of lists).
        
    Notes
    -----
    Obvious faults in this function, but it is a placeholder until
    class 'Matrix' is implemented.
    
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> isMatrix(A)
    >>> True
    
    >>> A = [1, 2, 3]
    >>> isMatrix(A)
    >>> False
   
    """
    res = isinstance(cand, list)
    return res and all(isinstance(entry, list) for entry in cand)
    
def dim(cand):
    """Return the dimension of a vector or matrix.
    
    Parameters
    ----------
    cand : list, list of lists
        Vector or matrix whose dimensions will be returned.

    Returns
    -------
    dim(cand): tuple
        The dimension of 'cand'.
    
    Example(s)
    ----------
    >>> A = [[1, 2, 3], [4, 5, 6]]
    >>> dim(A)
    >>> (2, 3)
    
    >>> A = [1, 2, 3]
    >>> dim(A)
    >>> (1, 3)
    
    """
    if isMatrix(cand):
        rows, cols = len(cand), len(cand[0])
    else:
        rows, cols = 1, len(cand) 
    return rows, cols

def makeList(entry):
    """Convert an entry into a list.
    
    Parameters
    ----------
    entry : *
        The object to be converted into a list format
        or nested in a list).
        
    Returns
    -------
    res : list
        The entry converted into a list.
        
    Notes
    -----
    If order of output matters, input should be ordered iterable. 
    If the input is not an iterable, the outputs is the input
    nested in a list e.g. 4 -> [4] but also sp.sin(t) -> [sp.sin(t)].
         
    Example(s)
    ----------
    >>> n = 5
    >>> makeList(n)
    >>> [5]
    
    >>> n = (1, 2, 4)
    >>> makeList(n)
    >>> [1, 2, 3]
    
    >>> n = {1, 2, 3}
    >>> makeList(n)
    >>> [1, 2, 3]
    
    """
    #Deal with cases
    if isinstance(entry, str):
        if ',' or ' ' not in entry:
            res = [entry]
        else:
            entry = entry.replace(' ', ',')
            entry = entry.replace(',,', ',')
            res = entry.split(',') #seperate by spaces or commas
    else:
        try:
            iter(entry)
            res = list(entry)
        except TypeError:
            res = [entry]
        
    return res
    
def listify(*argv):
    """Convert any non-list inputs into lists.
    
    Used in makeDrift function to standardize input type
    for other functions.
    
    Parameters
    ----------
    *argv : *
        
    Returns
    -------
    res : list, list of lists
        A list is returned if one argument given. Otherwise 'makeList'
        is applied to each argument.
        
    Notes
    -----
    See 'makeList' documentation for treatment of input and ordered outputs.
         
    Example(s)
    ----------
    >>> toList = {10, 2}
    >>> makeList(toList)
    >>> [10, 2]
    
    >>> toList = ({10, 2}, 2)
    >>> makeList(toList)
    >>> [{10, 2}, 2]
    
    >>> toList = 10
    >>> makeList(toList)
    >>> [10]
    
    >>> toList = 's,a,s,s'
    >>> makeList(toList)
    >>> ['s', 'a', 's', 's']
    
    """
    res = [makeList(arg) for arg in argv]
    return res if len(res) > 1 else res[0]
    
def matrixify(arg):
    """Convert the input into a matrix (list of lists).

    Used in makeDiffusion function to standardize input type
    for other functions.
    
    Parameters
    ----------
    arg: *
        Input to be converted into a list of lists.
        
    Returns
    -------
    matrixify(arg) : list of lists
        A matrix (list of lists) with the contents of the input.
        
    Notes
    -----
    To be deprieciated when class: 'Matrix' is implemented as a subclass
    of linear maps. This function is used to essentially __init__ the
    'Matrix' class, furthermore in 'chnageForm' this function acts
    as a way to convert between matrices and vectors and scalers.
         
    Example(s)
    ----------
    >>> n = 5
    >>> matrixify(n)
    >>> [[5]]
    
    >>> [(1, 2, 3), (4, 5, 6)]
    >>> matrixify(n)
    >>> [[1, 2, 3], [4, 5, 6]]
    
    """
    if not isinstance(arg, list): #non-list -> list
        arg = [arg]      
    return [makeList(el) for el in arg]

def changeForm(x, toScaler=False, toVector=False, toMatrix=True):
    """Convert between a scalar, vector, and matrix.
    
    Parameters
    ----------
    x : list, list of lists, sympy 'Symbol', 
            sympy 'Function', numeric(float, int)
        Entry of to be converted into a different form.
        
    toScaler, toVector, toMatrix : bool, optional
        Describes what the target form is.
        
    Returns
    -------
    res : list, list of lists, sympy 'Symbol', 
            sympy 'Function', numeric(float, int)
        A matrix, vector, or scaler is returned.
         
    Notes
    -----
    Mapping 'upwards' (e.g. scaler -> vector or vector -> matrix) amounts 
    to wrapping the input in the required number of lists.
    
    Mapping downwards is done essentialy by the 'natural projection'. 
    That is to say a vector is converted to a scaler by returning the
    first entry only, a matrix is converted to a scaler the same way.
    Converting a matrix to a vector depends on the dimension of the matrix.
    All non-row matrices are converted to vectors by returning the first
    column of the matrix. For row matrices however the first row is taken.
    This was done to keep the function intuitive and to also make sure
    vector -> matrix -> vector has the same input and output vector. 
         
    Example(s)
    ----------
    (Row) Matrix to Vector:
    >>> n = [[1, 2, 3]]
    >>> changeForm(n, toVector=True)
    >>> [1, 2, 3]
    
    (Non-Row) Matrix to Vector
    >>> n = [[90, 1], [2, 2]]
    >>> changeForm(n, toVector=True)
    >>> [90, 2]
    
    """
    if x != 0: #Will check if x is [] or [[]]
        if not x: #Only occurs if x = []
            x = [0]
        elif x == [[]]:
            x = [[0]]
            
    if not (toScaler or toVector or toMatrix): #trivial case
        return x
        
    if toScaler or toVector:
        toMatrix = False
        
    res = matrixify(x)
    if toVector:
        res = res[0] if len(res) == 1 else [y[0] for y in res]
    elif not toMatrix: #desired output is a scalar
        res = res[0][0] if res[0] else 0 #note [[]] and [] -> 0
        
    return res

def simplifyVector(vector):
    """Symbolically simplify the components of a list.
    
    Parameters
    ----------
    vector : list
        The to-be-simplified vector.
        
    Returns
    -------
    vector : list
        Vector with simplified entried.
    
    Notes
    -----
    Simplifications are done with sympy's 'simplify' function.
    
    See Also
    --------
    docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html
         
    Example(s)
    ----------
    >>> t, x = sp.symbols('t x')
    >>> u = [sp.sin(t)**2 + sp.cos(t)**2, 1-2, x**2/x**3]
    >>> simplifyVector(u)
    >>> [1, -1, 1/x]
    
    """
    return [sp.simplify(component) for component in vector]

#Functions to standardize user input and initialize the SDE class
def strToSymbol(stringVar):
    """Convert an iterable of strings into sympy symbols.
    
    Parameters
    ----------
    stringVar : 'str', iterable (of 'str')
        Strings or strings to be converted to class: sp.Symbol.
        
    Returns
    -------
    strToSymbols(stringVar) : sp.Symbol, tuple (of sp.Symbol's)
        A single or tuple of entrie each of class sympy 'Symbol'.
    
    See Also
    --------
    docs.sympy.org/latest/modules/core.html?highlight=symbol#module-sympy.core.symbol
    
    Notes
    -----
    Multicharectored symbols are supported but must be given as one entry
    of an iterable (see example 1 vs last input of example 3 below).s
         
    Example(s)
    ----------
    >>> s = 'tsw'
    >>> strToSymbol(s)
    >>> (t, s, w)
    
    >>> s = 't'
    >>> strToSymbol(s)
    >>> t
    
    >>> s = ('a', 'b', 'cd')
    >>> strToSymbol(s)
    >>> (a, b , cd)
    
    """
    return sp.symbols(' '.join(stringVar))

def makeDrift(drift):
    """Standardize and simplify user-inputed drift.
    
    Parameters
    ----------
    drift : * (see 'listify)
        Input to be converted to list (for SDE's drift __init__).
        
    Returns
    -------
    makeDrift(drift) : list
        Drift in vector format.
    
    See Also
    --------
    For input see 'listify' and 'simplifyVector'.
    For output see SDE's __init__.
         
    Example(s)
    ----------
    >>> s = 10
    >>> makeDrift(s)
    >>> [10]
    
    >>> t = sp.Symbol('t')
    >>> s = (sp.sin(t)**2 + 2*sp.cos(t)**2, t+1)
    >>> makeDrift(s)
    >>> [cos(t)**2 + 1, t + 1]
    
    """
    drift = listify(drift) #Standardize format
    return simplifyVector(drift)

def makeDiffusion(diffusion):
    """Standardize and simplify the user-inputed diffusion.
        
    Parameters
    ----------
    diffusion : * (see 'matrixify')
        Input to be converted to list of lists (for SDE's diffusion __init__).
        
    Returns
    -------
    makeDiffusion(diffusion) : list of lists
        Diffusion in matrix format.
    
    See Also
    --------
    For input see: matrixify and simplifyVector.
    
    For affects of output see SDE __init__.
         
    Example(s)
    ----------
    >>> s = 10
    >>> makeDiffusion(s)
    >>> [[10]]
    
    """
    #Diffusion should be a matrix (list of lists)
    diffusion = matrixify(diffusion) #Standardize format
    return [simplifyVector(row) for row in diffusion]
    
def makeNames(names, dim):
    """Standardize the format of user-inputted processes names.
    
    Parameters
    ----------
    names : str, ordered iterable
        The name or names desired for the proccess.
    dim : int
        The dimension of the stochastic process.
        
    Returns
    -------
    names : list
         
    Notes
    -----
    Note input should not be empty, but if an empty string or iterable is
    given 'X' is used as the variable (or X_i if dim > 1).
    
    If the dimension does not match the number of names given, the last
    entered name is used as a variable with subscripts attached. This can
    look funny is the variable already has subscripts.
         
    Example(s)
    ----------
    >>> names = ['X', 'Y']
    >>> dim = 2
    >>> makeNames(names, dim)
    >>> ['X', 'Y']
    
    >>> names = ['A', 'C']
    >>> dim = 4
    >>> makeNames(names, dim)
    >>> ['X', 'Y_1', 'Y_2', 'Y_3']
    
    """
    if not names: #degenerate case
        return [f'X_{i}' for i in range(1, dim+1)]
    
    names = listify(names)
    if len(names) != dim:
        y = names.pop()
        names += [f'{y}_{i+1}' for i in range(dim-len(names))]

    return names

#Functions related to sympy class: functions
def changeVar(f, currVars, newVars):
    """Swap the variables in a given function.
    
    Parameters
    ----------
    f: sympy: 'Function', list of sympy: 'Function' 
        Function which swap of variables will occur on. 
    currVars: sympy: 'Symbol', iterable of sympy: 'Symbol'
        The current variable(s) to be swapped out with newVars.
    newVars : numeric(float, int), sympy: 'Symbol', 
                iterable of sympy: 'Symbol', sympy: 'Function',
                iterable of sympy:
        The new vairble(s).
        
    Returns
    -------
    newF : list of sympy: 'Function'
        Function 'f' written with respect to the new variables.
        
    Notes
    -----
    Ordered iterables should be used if swapping multiple variables.
    If len(currVars) != len(newVars): Variables are swapped only until
    one list is exhausted. Also this function can be used to evaluate
    at certian fixed points (by having numerics as variables being swapped
    in).
    
    See Also
    --------
    'evalFunc' for using this function to help evaluate at certian points.
    
    Example(s)
    ----------
    >>> t, x, y = sp.symbols('t x y')
    >>> f = [5, sp.tan(x)]
    >>> a, b = sp.symbols('a b') #can re-'symbol' var: t aswell
    >>> changeVar(f, [t, x, y], [t, a, b])
    >>> [5, tan(a)]
    
    >>> t, x, y = sp.symbols('t x y')
    >>> f = sp.cos(x**2+y) + sp.exp(7*x)
    >>> a = sp.Symbol('a')
    >>> changeVar(f, [y], a)
    >>> [exp(7*x) + cos(a + x**2)]

    """
    #Standardize format
    f, currentVar, newVar = listify(f, currVars, newVars)

    cutOff = min(len(currentVar), len(newVar))
    currentVar, newVar = currentVar[:cutOff], newVar[:cutOff]
        
    #Couple variables with their replacements
    replacements = [(currentVar[i], new)
                    for i, new in enumerate(newVar)]
    if isMatrix(f):
        newF = [[sp.simplify(f_).subs(replacements) for f_ in fi]
                 for fi in f]
    else:
        newF = [sp.simplify(f_).subs(replacements) for f_ in f]
        
    return newF
   
def evalFunc(f, varf, values, toScaler=False, toVec=False, toMat=False):
    """Evaluate a function and return in the desired form.
    
    Parameters
    ----------
    f : sympy: 'Function'
        The function who will have a change of variables.
    varf : sympy: 'Symbol', iterable of sympy: 'Symbol'
        The current variables to be swapped out with newVars.
    values : numeric(float, int), sympy: 'Symbol', 
                iterable of sympy: 'Symbol'
        The new vairble(s).
    toScaler, toVec, toMat : bool, optional
        If returned value should be in a special format.
    
    Returns
    -------
    res : numeric (float, int), sympy: 'Symbol', list, list of lists
        Function evaluated at the point. 
    
    Notes
    -----
    In addition to simplyfying the expression, if the dimension of
    the result can trivally have its form changed between a scaler,
    vector, and matrix the option can be done. In general if the
    form change causes loss of information (i.e. projects down into
    a space that is 'too small' then it is adviced to do the change
    independently with use of 'changeForm.'
    
    See Also
    --------
    'changeVar' for uses and limitations of this function (namely with symbolic
    fixed points).
    
    Example(s)
    ----------
    >>> t, x = sp.symbols('t x')
    >>> f = [sp.sin('t'), x**2]
    >>> evalFunc(f, [t, x], [sp.pi, 5])
    >>> [0, 25]
    
    >>> t, x = sp.symbols('t x')
    >>> f = sp.sin('t')
    >>> evalFunc(f, [t, x], sp.pi, toMat=True)
    >>> [[0]]
    
    """
    res = changeVar(f, varf, values)
    #If dimensionaly possible convert the result to the desired form
    row, col = dim(res)
    if (row, col) == (1,1): #res is a scaler, vec: [-], mat: [[-]]
        res = changeForm(res, toScaler, toVec, toMat)
    
    elif 1 in dim(res): #res is a vector, row matrix, or col matrix
        res = changeForm(res, toScaler=False, 
                         toVector=toVec, toMatrix=toMat)
        
    return res
        
def jacobian(f, variables):
    """Return the (variable restricted) Jacobian matrix.

    The Jacobian matrix of a function, f: R^n -> R^m, is
    the m by n matrix of all first-order partial derivatives.

    This function returns the matrix of first-order partial
    derivatives taken with respect to a possibly restricted
    collection of ordered variables.
        
    Parameters
    ----------
    f : sympy: 'Function', list of sympy: 'Function's.
        The function whose Jacobain is to be computed.
    variables : sympy: 'Symbol', iterable of sympy: 'Symbol'
        Each varaible a derivative will be taken with respect to.
        
    Returns
    -------
    jacobian(f, variables) : list of lists
        The Jacobian matrix.

    Example(s)
    ----------
    >>> t, x = sp.symbols('t x')
    >>> f = [t*sp.sin(x), x*sp.exp(y), x+t]
    >>> jacobian(f, [t, x])
    >>> [[sin(x), t*cos(x)], [0, exp(y)], [1, 1]]
    
    >>> t, x = sp.symbols('t x')
    >>> f = [t*sp.sin(x), x*sp.exp(y), x+t]
    >>> jacobian(f, t)
    >>> [[sin(x)], [0], [1]]
    
    >>> t = sp.Symbol('t')
    >>> jacobian(5, t)
    >>> [[0]]
    
    >>> t, x, y = sp.symbols('t x y')
    >>> f = [2, sp.exp(y), x+y]
    >>> jacobian(f, [t, x, y])
    >>> [[0, 0, 0], [0, 0, exp(y)], [1, 1, 0]]
    
    """
    f, variables = listify(f, variables) #Standardize format
    return [[sp.diff(f_, var) for var in variables] for f_ in f]

def gradient(f, variables):
    """Return the (variable restricted) gradient.

    The gradient vector of a scaler function is formed 
    by all the functions partial derivatives. It is
    equivalent to the first row of the row-matrix Jacobian
    matrix.
        
    Parameters
    ----------
    f : sympy: 'Function'
        The function whose gradient is to be computed.
    variables : sympy: 'Symbol', iterable of sympy: 'Symbol'
        Each varaible a derivative will be taken with respect to.
        
    Returns
    -------
    gradient(f, variables) : list
        The gradient vector.
        
    See Also
    --------
    'jacobian' function for computation of the gradiant.
    
    Example(s)
    ----------
    >>> t, x, y = sp.symbols('t x y')
    >>> f = sp.sin(t)**2 + x*t + y
    >>> jacobian(f, [t, x])
    >>> [x + 2*sin(t)*cos(t), t]
    """
    return jacobian(f, variables)[0]
    
def hessian(f, variables):
    """Return the (variable restricted) Hessian matrix.

    The Hessian matrix of a function, f: R^n -> R, is
    the n by n matrix of all second-order partial derivatives.

    This function returns the matrix of second-order partial
    derivatives taken with respect to a possibly restricted
    collection of ordered variables.
    
    Parameters
    ----------
    f : sympy: 'Function'
        The function whose Hessian is to be computed.
    variables : sympy: 'Symbol', iterable of sympy: 'Symbol'
        Each varaible derivatives will be taken with respect to.
        
    Returns
    -------
    Hf : list of lists
        The Hessian matrix.
    
    Example(s)
    ----------
    >>> t, x, y = sp.symbols('t x y')
    >>> f = t * sp.sin(x)
    >>> hessian(f, [t, x, y])
    >>> [[0, cos(x), 0], [cos(x), -t*sin(x), 0], [0, 0, 0]]
    
    >>> t, x, y = sp.symbols('t x y')
    >>> f = t*sp.sin(x) + sp.exp(y)
    >>> hessian([f], t)
    >>> [[0]]
    
    >>> t = sp.Symbol('t')
    >>> hessian(5, t)
    >>> [[0]]
    
    >>> t, x, y = sp.symbols('t x y')
    >>>> f = sp.exp(y) * (x+t)
    >>> hessian(f, [t, x, y])
    >>> [[0, 0, exp(y)], [0, 0, exp(y)], [exp(y), exp(y), (t + x)*exp(y)]]
    
    """
    #List formatting is only req'd for functions w/ image dim > 1
    if isinstance(f, list):
        f = f[0]
    variables = listify(variables) #Standardize format

    #Find second derivatives
    Hf = [[sp.diff(f, var1, var2) for var2 in variables]
          for var1 in variables]

    return Hf

def vectorHessian(f, variables):
    """Return the (variable restricted) Hessian matrix for
    a vector-valued function.

    The Hessian matrix of a vector valued function,
    f: R^n -> R^m, is a multidimensional array with m
    total components. Each component corresponds to
    the classic Hessian of f restricted to that dimension.
    
    Parameters
    ----------
    f : sympy: 'Function', list of sympy: 'Function's
        The function whose Hessian is to be computed.
    variables : sympy: 'Symbol', iterable of sympy: 'Symbol'
        Each varaible derivatives will be taken with respect to.
        
    Returns
    -------
    vectorHessian(f, variables) : list of lists
        The vector of Hessian matrices.
    
    Example(s)
    ----------
    >>> t, x, y = sp.symbols('t x y')
    >>> f = [t*sp.sin(x), sp.exp(y), x+t]
    >>> vectorHessian(f, [t, x])
    >>> [[[0, cos(x)], [cos(x), -t*sin(x)]], 
         [[0, 0], [0, 0]], 
         [[0, 0], [0, 0]]]
     
    >>> t, x, y = sp.symbols('t x y')
    >>> f = [t*sin(x), exp(y), x+t]
    >>> vectorHessian(f, t)
    >>> [[[0]], [[0]], [[0]]]
    
    """
    f, variables = listify(f, variables) #Standardize format
    return [hessian(f_, variables) for f_ in f]

#Functions used to print symbolic results
def rewrite(function):
    """Symbolically simplify and rewrite a function.
    
    Parameters
    ----------
    function : sympy: 'Function', list of sympy: 'Function'
        Input to be converted simplified and rewritten.
        
    Returns
    -------
    f : str
        The simplfied and rewritten version of the input.
              
    Example(s)
    ----------
    >>> t = sp.Symbol('t')
    >>> f = (5**(sp.sin(t) + sp.cos(t))) / 5**(sp.cos(t))
    >>> rewrite(f)
    >>> 5^sin(t)
    
    """
    strF = str(sp.simplify(function))
    f = strF.replace('**', '^')
    f = f.replace('*', '')
    
    return f

def showComponent(function, derivative):
    """Express a component of a SDE as a simplified string.
    
    Parameters
    ----------
    function : sympy: 'Function'
        Input to be converted simplified and rewritten.
    derivative : str
        The associated derivative, e.g. dB, dt, or dX_3.
    
    Returns
    -------
    component : str
        A simplfied and rewritten component of an SDE e.g. sin(t)dt
              
    Example(s)
    ----------
    >>> showComponent(0, 'dX') #output should be empty string
    >>>
    
    >>> showComponent(1, 'dX')
    >>> dX
    
    >>> t = sp.Symbol('t')
    >>> f = sp.sin(t**2)
    >>> showComponent(sp.sin(t**2), 'dX')
    >>> [sin(t^2)]dX

    """
    if function == 0:
        component = ''
    elif function == 1:
        component = derivative
    else: #f !=1 and f != 0
        component = f'[{rewrite(function)}]' + derivative
          
    return component

def makeProcess(process, drift, diffusion):
    """Express a SDE as a string.

    Parameters
    ----------
    process : str
        Names of the stochastic process (e.g. B for Brownian motion)
    drift : numeric(float, int), sympy: 'Function'
        The drift of the SDE (see: dX.drift).
    diffusion : list of (sympy: 'Function',
                        'Symbol', numerics(int, float))
        The diffusion of the SDE (see: dX.diffusion).
        
    Returns
    -------
    makeProcess(process, drift, diffusion) : str
        A SDE expressed as a string.
              
    Example(s)
    ----------
    >>> t, X = sp.symbols('t, X')
    >>> drift = sp.sin(t)**2
    >>> diffusion = [X + sp.cos(X*t), 0, 2*X - t]
    >>> makeProcess('X', drift, diffusion)
    >>> dX = [sin(t)^2]dt
            +[X + cos(Xt)]d(W_1)
            +[2X - t]d(W_3)
    
    """
    X = (f'({process})' if len(process) > 1 else f'{process}')
    dX = 'd' + X
    
    udt = showComponent(drift, 'dt')
    vdW = ''

    alignment = ' ' * (len(dX)+2) + '+'
    for i, vol in enumerate(diffusion):
        component = showComponent(vol, f'd(W_{i+1})')
        if component and not vdW:
            vdW += component
        elif component:
            vdW += '\n' + alignment + component

    if udt and vdW:
        RHS = udt + '\n' + alignment + vdW
        
    elif not (udt or vdW):
        RHS = '0'
        
    else:
        RHS = udt + vdW

    return f'{dX} = {RHS}'

#Other functions
def dW(variance, numberSamples):
    """Return a list of samples from N(0, sqrt(variance)).
        
    Parameters
    ----------
    variance : numeric(float, int)
        Variance of underlying normal distrution.
    numberSamples : int
        The number of samples to be taken.
        
    Returns
    -------
    dW_ : list
        A list of samples from a normally distrubed 
        (mean = 0, var = 'variance') distrubution.
    
    See Also
    --------
    numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
         
    Example(s)
    ----------
    >>> var, n = 1, 10
    >>> dW(var, n)
    >>> [-0.22088139375398147, 0.4230443424263282, -0.4810667189774699,
        1.6861934734865052, 2.1839407554276264, -2.5162186121902086, 
        -0.6980052900585519, 0.06057263528953575, 0.3917347012460837, 
        -2.0137654802069704]

    """
    dW = normal(0, sp.sqrt(variance), numberSamples)
    return list(dW)
