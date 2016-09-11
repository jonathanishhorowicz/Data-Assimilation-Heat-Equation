import numpy as np

# Class to compute measurements
class measurementGetter:
    
    # Constructor
    def __init__(self, exactSol, omega, X, T, dx, dt, sigma):
        self.omega = omega
        self.T = T
        self.dx = dx
        self.dt = dt
        self.X = X
        self.sigma = sigma # Noise amplitude
        self.exactSol = exactSol # Lambda function
        self.getSignFromRandArray = np.vectorize(self.getElementSign)
        
    def getMeasMesh(self):
        # The full mesh details
        xVals = np.linspace(self.X[0], self.X[1], (self.X[1]-self.X[0])/self.dx +1)
        tVals = np.linspace(self.T[0], self.T[1], (self.T[1]-self.T[0])/self.dt +1)
        
        # Get the indicies where omega starts and ends on the full domain
        omegaIndicies = [None, None]
        omegaIndicies[0] = np.where(xVals==self.omega[0])[0]
        omegaIndicies[1] = np.where(xVals==self.omega[1])[0] + 1
        
        print "omega indicies:", omegaIndicies
    
        # Get the exact solution on omega
        xVals = xVals[omegaIndicies[0]:omegaIndicies[1]]
        Um = np.zeros((len(tVals), len(xVals)))
        vectorizedExactSol = np.vectorize(self.exactSol)
        for i in range (0, len(tVals)):
                Um[i,:] = vectorizedExactSol(xVals,tVals[i])
                # Add random noise to measurements
                if (self.sigma!=0.0):
                    sign = self.getPerturbSign(len(xVals)) # The sign of the perturbation
                    magnitude = np.random.rand(1,len(xVals))
                    noise = magnitude * sign * self.sigma + 1.0
                    Um[i,:] *= noise.reshape(len(xVals),)
                    
                    
                
        # Return the mesh of measurements
        measMesh = measuredMesh(self.omega, self.X, self.T, self.dx, self.dt, omegaIndicies)
        measMesh.mesh = Um
        return measMesh
    
    # The sign of the perturbation 
    def getPerturbSign(self, numberOfNodes):
        rand = np.random.randint(2, size=numberOfNodes)
        sign = self.getSignFromRandArray(rand)
        return sign
    
    # Get signs of perturbations for each position in x from array of random 0,1
    def getElementSign(self, val):
        if val==0:
            return -1
        if val==1:
            return 1

# Class to hold measurements and information about their location within the whole domain
class measuredMesh:
    # Constructor
    def __init__(self, omega, X, T, dx, dt, omegaIndicies):
        self.omega = omega
        self.T = T
        self.dx = dx
        self.dt = dt
        self.mesh = None
        self.omegaIndicies = omegaIndicies
    
    # For plotting - returns a full mesh with the solution in omega and zeros elsewhere
    def getFullMesh(self, X):
        xVals = np.linspace(X[0], X[1], (X[1]-X[0])/self.dx +1)
        tVals = np.linspace(self.T[0], self.T[1], (self.T[1]-self.T[0])/self.dt +1)
        fullMesh = np.zeros((len(tVals), len(xVals)))
        fullMesh[:,self.omegaIndicies[0]:self.omegaIndicies[1]] = self.mesh
        fullSol = fe_mesh(X, self.T, self.dx, self.dt)
        fullSol.mesh = fullMesh
        return fullSol
        
    # For direct indexing of the mesh
    def __getitem__(self, indexTuple):
        i, j = indexTuple
        return self.mesh[i,j]
        
    def __setitem__(self, index, value):
        self.mesh[index] = value
        
    def __str__(self):
        return str(self.mesh)
