import numpy as np
import scipy as sp

from scipy.sparse.linalg import cg
from numpy.linalg import norm

from forwardSolver import forwardSolver
from adjointSolver import adjointSolver


class dataAssimilator:
    # Constructor
    def __init__(self, P, zeta, u00, X, T, dx, dt, f, measMesh, gamma, gammaM, maxIter, cgtol, debug, printnorms, iterinc, ztolerance):
        self.P = P
        self.zeta = zeta
        self.u00 = u00
        self.X = X
        self.T = T
        self.dx = dx
        self.dt = dt
        self.numberOfNodes = (self.X[1]-self.X[0])/self.dx + 1
        self.nodePositions = np.linspace(self.X[0], self.X[1], (self.X[1]-self.X[0])/self.dx + 1)
        self.f = f
        self.M = self.buildMMatrix()
        self.K = self.buildKMatrix()
        self.PM = np.dot(self.P, self.M)
        self.measMesh = measMesh
        self.gamma = gamma
        self.gammaM = gammaM
        self.maxIter = maxIter
        self.dgfcgtol = cgtol[2]
        self.debug = debug
        self.printnorms = printnorms
        self.iterinc = iterinc
        self.ztolerance = ztolerance
        
        self.iterationData = []
        self.z0Data = []
        
        self.prevZ0 = None
        self.UmMinZ0 = [u00, 100.0]
        
        # Class to solve for the lagrange multipliers
        self.adjSolver = adjointSolver(self.measMesh, self.X, self.T, self.dx, self.dt, self.gamma, self.gammaM, self.K, self.M, self.debug, cgtol[1])
        
        # To solve forward problem
        self.forwardsolver = forwardSolver(self.X, self.T, self.dx, self.dt, self.f, cgtol[0])
        
        if self.debug:
            print "Solving using gradient flow with pseudo-timestep", zeta, "gamma", gamma
            print "Starting initial condition", u00
            print "X = ", X, "T = ", T, "dx = ", dx, "dt = ", dt
            print "The measurements are", measMesh, "\n"
        
    
    # Solve using iterative data assimilation algorithm
    def solve(self):
        Um = self.u00
        Umplus1 = None
        
        # Loop until iteration limit reached
        for i in range (0, self.maxIter):

            # Calculate the next initial condition
            Umplus1, cont = self.getUmplus1(Um, i)
                  
            # If has converged
            if not cont:
                comment = "converged after " + str(i) + " iterations"
                print comment
                return (self.UmMinZ0[0], comment)
            
            # If not calculate the next Um
            else:
                Um = Umplus1
            
            if self.debug:
                print "\n"
            
        comment = "did not converge in " + str(self.maxIter) + " iterations"
        print comment
        return (self.UmMinZ0[0], comment)

            
    # One iteration of discrete gradient flow
    def getUmplus1(self, Um, iterationNo):
        cont = False # Whether or not to continue the iterations
        
        if self.debug:
            print "solving using the initial condition", Um
        
        # On first itertation initial condition will be a vector function
        if type(Um) is not np.ndarray:
            Umfunc = np.vectorize(Um)
            Um = Umfunc(self.nodePositions)
        
        # Solve the forward problem
        solMesh = self.forwardsolver.solve(Um.reshape(Um.shape[0],))
        
        if self.debug:
            print "solution is ", solMesh
            print "getting lagrange multipliers"
        
        # Solve the adjoint problem
        z = self.adjSolver.solveForZ(solMesh)
        
        normz0 = norm(z[0,1:-1])
        
        # Get next initial condition using discrete gradient flow
        temp1 = np.dot(self.PM - self.gamma[0] * self.zeta * self.dx**2.0 * self.K, Um[1:-1]).reshape(self.K.shape[0],1)
        temp2 = np.dot(self.M, z[0,1:-1].reshape(z[0,1:-1].shape[0],1))
        rhs =  temp1 + temp2 * self.zeta
        Umplus1 = np.zeros((self.numberOfNodes, 1))
        Umplus1[1:-1] = cg(self.PM, rhs, tol = self.dgfcgtol)[0].reshape(Umplus1.shape[0]-2,1)
        
        if self.debug:
            print " got um+1 = ", Umplus1
            
        self.prevZ = z
        
        if self.printnorms:
            if np.mod(iterationNo, self.iterinc)==0:
                print "Iteration number", iterationNo
                print "norm(z0)=", normz0
                print "norm(uM+1)=", np.linalg.norm(Umplus1)
                print "relative change in norm(z0):", -(normz0-self.UmMinZ0[1])/self.UmMinZ0[1], "\n"
        
        
        # Store the norm of z0 and iteration number
        self.z0Data.append((iterationNo, normz0))

        
        # Store the inital condition with the lowest norm(z0)
        if((self.UmMinZ0[1]-normz0)/self.UmMinZ0[1] > self.ztolerance):
            self.UmMinZ0[0] = Umplus1
            self.UmMinZ0[1] = normz0
            cont = True
         
        return [Umplus1,cont]
    
    # Returns the mass matrix
    def buildMMatrix(self):
        firstCol = np.zeros((self.numberOfNodes-2, 1))
        firstCol[0] = 4.0
        firstCol[1] = 1.0
        
        M = sp.linalg.toeplitz(firstCol, firstCol)

        M *= self.dx/6.0
        
        return M
    
    # Returns the stiffness matrix
    def buildKMatrix(self):
        firstCol = np.zeros((self.numberOfNodes-2, 1))
        firstCol[0] = 2.0
        firstCol[1] = -1.0
        
        K = sp.linalg.toeplitz(firstCol, firstCol)

        K *= 1.0/self.dx
        
        return K