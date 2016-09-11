import numpy as np
import scipy as sp
from scipy.sparse.linalg import cg
from scipy.integrate import quad

# A class for the solution mesh
class sol_mesh:
    
    # Constructor. dx, dt are the mesh size
    def __init__ (self, X, T, dx, dt):
        self.dx = dx
        self.dt = dt
        self.X = X
        self.T = T
        self.mesh = np.zeros([(self.T[1]-self.T[0])/self.dt +1, (self.X[1]-self.X[0])/self.dx +1])
        self.nodePositions = np.linspace(self.X[0], self.X[1], (self.X[1]-self.X[0])/self.dx +1)
        self.numberOfNodes = len(self.nodePositions)

    # Apply the initial condition ic. ic can be a lambda function or an array
    def applyIC(self, ic):
        # If the initial condition is an array
        if type(ic) is np.ndarray:
            self.mesh[0,:] = ic.reshape(ic.shape[0],)
        
        # If the initial condition is a lambda function
        else:
            vecic = np.vectorize(ic)
            self.mesh[0,:] = vecic(self.nodePositions)

         
    # For direct indexing of the mesh
    def __getitem__(self, indexTuple):
        i, j = indexTuple
        return self.mesh[i,j]
        
    def __setitem__(self, index, value):
        self.mesh[index] = value
        
    def __str__(self):
        return str(self.mesh)
        
    def shape(self):
        return self.mesh.shape


# Solve the heat equation using finite element method for x and backward Euler finite difference scheme in time     
class heat_solver:
    
    # Constructor
    def __init__(self, solMesh, f, cgtol):
        # Get the dimensions from the mesh
        self.elementLength = solMesh.dx
        self.timeStep = solMesh.dt
        self.solMesh = solMesh
        self.f = f # A lambda function for the source term
        self.cgtol = cgtol
        
        # Compute the stiffness and mass matrices and the load vector
        self.M = self.buildMassMatrix()
        self.K = self.buildStiffnessMatrix()
        self.F = self.getLoadVector()
        
        
    # Compute the global mass matrix
    def buildMassMatrix(self):          
        firstCol = np.zeros((self.solMesh.numberOfNodes-2, 1))
        firstRow = np.zeros((self.solMesh.numberOfNodes-2, 1))
        firstCol[0] = 4.0
        firstCol[1] = 1.0
        firstRow[0] = 4.0
        firstRow[1] = 1.0
        
        return self.elementLength/6.0 * sp.linalg.toeplitz(firstCol, firstRow)
        
                    
    # Compute the global stiffness matrix
    def buildStiffnessMatrix(self):
        firstCol = np.zeros((self.solMesh.numberOfNodes-2, 1))
        firstRow = np.zeros((self.solMesh.numberOfNodes-2, 1))
        firstCol[0] = 2.0
        firstCol[1] = -1.0
        firstRow[0] = 2.0
        firstRow[1] = -1.0
        
        return 1.0/self.elementLength * sp.linalg.toeplitz(firstCol, firstRow)
                
            
    # Compute the load vector
    def getLoadVector(self):
        fVec = np.zeros([self.solMesh.numberOfNodes, 1])
        fLoc = np.zeros([2,1])
        # Loop over elements
        for i in range (0, self.solMesh.numberOfNodes-1):
            # Get local load vector for this element
            node1 = self.solMesh.nodePositions[i]
            node2 = self.solMesh.nodePositions[i+1]
            fLoc[0] = quad(lambda x: (node2 - x)/self.elementLength * self.f(x), node1, node2)[0]
            fLoc[1] = quad(lambda x: (x - node1)/self.elementLength * self.f(x), node1, node2)[0]
            fVec[i:i+2] += fLoc
        
        return fVec[1:-1]
    
    # Use a backward Euler finite difference method to discretise in time
    def backwardEuler(self):
        # March forward in time
        for i in range(1, self.solMesh.mesh.shape[0]):        
            prevSol = np.reshape(self.solMesh[i-1,1:-1], (self.solMesh[i-1,1:-1].shape[0], 1))
            A = self.M + self.timeStep * self.K
            #F = self.getLoadVector(tVals[i])
            b = self.timeStep * self.F + np.dot(self.M, prevSol)
            x = cg(A, b, tol=self.cgtol)[0]
            self.solMesh[i,1:-1] = x

        
