import numpy as np

from scipy.sparse.linalg import cg

class adjointSolver:
    # Constructor
    def __init__(self, measMesh, X, T, dx, dt, gamma, gammaM, K, M, debug, cgtol):
        self.measMesh = measMesh
        self.X = X
        self.T = T
        self.dx = dx
        self.dt = dt
        self.gamma = gamma
        self.gammaM = gammaM
        self.nodePositions = np.linspace(self.X[0], self.X[1], (self.X[1]-self.X[0])/self.dx +1)
        self.numberOfNodes = len(self.nodePositions)
        self.M = M
        self.K = K
        self.omegaIndicies = self.measMesh.omegaIndicies
        self.omegaNodePositions = self.nodePositions[self.omegaIndicies[0]:self.omegaIndicies[1]]
        self.Momega = self.M[0:len(self.omegaNodePositions),0:len(self.omegaNodePositions)]
        self.Komega = self.K[0:len(self.omegaNodePositions),0:len(self.omegaNodePositions)]
        self.debug = debug
        self.cgtol = cgtol
        
        if self.debug:
            print "mass matrix for lagrange multiplier solver:", self.M
            print "stiffness matrix for lagrance multiplier solver", self.K
            print "nodes in omega are ", self.omegaNodePositions
            print "the mass and stiffness matrix for omega have shapes", self.Momega.shape, self.Komega.shape

    
    # Compute the lagrange multipliers    
    def solveForZ(self, solMesh):
        zMesh = np.zeros((solMesh.mesh.shape))
        for i in range(solMesh.mesh.shape[0]-1, 0, -1):
            omegaError = self.getE(solMesh, i)
            
            if self.debug:
                print "current time index (lagrange multiplier solver):", i
                print "difference in omega=", omegaError
                print "current row of z = ", zMesh[i,:]
                
            lhs = self.M + self.dt * self.K
            
            # Data assimilation term
            error = self.gammaM * self.dotInOmega(self.Momega, omegaError)
            
            # The regularisation terms
            # The second regularisation term has different forms depending on the timestep
            if i==solMesh.mesh.shape[0]-1: # From T to T-tau
                reg1 = self.gamma[1] * np.dot(self.K, solMesh[i,1:-1]-solMesh[i-1,1:-1]).reshape(self.numberOfNodes-2,1)
            elif i == 0: # From tau to 0
                reg1 = -1.0 * self.gamma[1] * np.dot(self.K, solMesh[i+1,1:-1] - solMesh[i,1:-1])
            else: # All intermediate time steps
                reg1 = self.gamma[1] * np.dot(self.K, -1.0 * solMesh[i+1,1:-1] + 2.0 * solMesh[i,1:-1] - 1.0 * solMesh[i-1,1:-1]).reshape(self.numberOfNodes-2,1)

            rhs =  np.dot(self.M, zMesh[i,1:-1]).reshape(self.M.shape[0],1) + self.dt * (error + reg1)
            
            nextZ, info = cg(lhs, rhs, tol=self.cgtol)
            #print info
            if self.debug:
                print "next row of z:", nextZ
            zMesh[i-1,1:-1] = nextZ
            
            if self.debug:
                print "updated zmesh = ", zMesh
            
        return zMesh
    
    # Multiply 2 arrays in omega only
    def dotInOmega(self, arr1, arr2):
        product =  np.zeros((self.numberOfNodes-2,1))
        product[self.omegaIndicies[0]-1:self.omegaIndicies[1]-1] += np.dot(arr1, arr2.reshape(arr2.shape[0],1))
        return product
        
    
    # Get the difference between the computed and calculated values in omega at a given time
    def getE(self, solMesh, timeIndex):
        if self.debug:
            print "Getting E vector"
            print "measurement = ", self.measMesh.mesh[timeIndex,:]
            
        # The difference between the measured and calculated values in omega
        difference = self.measMesh.mesh[timeIndex,:] - solMesh.mesh[timeIndex,self.omegaIndicies[0]:self.omegaIndicies[1]]
        if self.debug:
            print "difference between measured and computed solution in omega = ", difference
            
        return difference

        