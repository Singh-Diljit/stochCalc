"""Implement systems of SDEs with m-dim. Wiener process."""

import helperFunctions as hf

class SDE:
    """This class implements SDEs."""
    
    def __init__(self, itoProcess, drift, diffusion,
                 timeInterval=[0,1], var=[], initialVal=[1]):

        #Initialize the name, drift vector, diffusion matrix
        self.drift = hf.makeDrift(drift)
        self.process = hf.makeNames(itoProcess, len(self.drift))
        self.diffusion = hf.makeDiffusion(diffusion)

        #Initialize the time interval, initial value
        #and variables used to define the SDE
        self.time = timeInterval
        self.var = hf.listify(var)
        self.initial = hf.listify(initialVal)

    @property
    def lenTime(self):
        """Duration of the Ito process."""
        return self.time[1] - self.time[0]
    
    @property
    def startTime(self):
        """Time the SDE begins at."""
        return self.time[0]

    @property
    def endTime(self):
        """Time the SDE ends at."""
        return self.time[1]

    @property
    def enumSystem(self):
        """Number of Ito process in the system of SDEs."""
        return len(self.drift)

    @property
    def dimBrownian(self):
        """Dimension of the brownian motion.

        Notes
        -----
        This is to be phased out in the next update in favor of
        'dimDiffusion'.
        
        """
        return len(self.diffusion[0])

    @property
    def dimDiffusion(self):
        """Dimension of the Wiener process."""
        return len(self.diffusion[0])

    def __repr__(self):
        """Representation of the class instance."""
        
        rep = f"""SDE(itoProcess = {self.process},
                      drift = {self.drift},
                      diffusion = {self.diffusion},
                      timeInterval = {self.time},
                      var = {self.var}
                      initialVal = {self.initial})"""

        return rep

    def __eq__(self, dY):
        """Return if dX and dY are equal (have same inputs).

        Notes
        -----
        This does not take into account systems that are identical
        but presented differently e.g. where a change of vars would
        show them being the same. This can be done with the
        'helperFunctions' file.
        
        """
        if self.__class__ != dY.__class__:
            return False

        return True if self.__dict__ == dY.__dict__ else False
    
    def __str__(self):
        """Return str(self).

        Example(s)
        ----------
        Single Ito Process
        >>> dX = SDE('X', cos(X)*sin(t**2), [sin(X), log(X+7), sin(X)/cos(X)])
        >>> print(dX)
        >>> dX = [sin(t^2)cos(X)]dt
               +[sin(X)]d(W_1)
               +[log(X + 7)]d(W_2)
               +[tan(X)]d(W_3)

        System of 3 Ito Process
        >>> drift = [sin(X_1)*t**2, log(X_2), 0]
        >>> diffusion = [[4,0,sin(t),X_1**4],
                        [cos(t*X_2),0,0,13/X_2],
                        [1,log(X_3),-5,0]]
        >>> dX = SDE(X, drift, diffusion)
        >>> print(dX)
        >>> d(X_1) = [t^2sin(X_1)]dt
                   +[4]d(W_1)
                   +[sin(t)]d(W_3)
                   +[X_1^4]d(W_4)
        >>> d(X_2) = [log(X_2)]dt
                   +[cos(X_2t)]d(W_1)
                   +[13/X_2]d(W_4)
        >>> d(X_3) = d(W_1)
                   +[log(X_3)]d(W_2)
                   +[-5]d(W_3)
                 
        """
        toShow = [hf.makeProcess(nameProcess,
                                 self.drift[i],
                                 self.diffusion[i])
                  for i, nameProcess in enumerate(self.process)]
        
        res = '\n'.join(toShow)

        return res
