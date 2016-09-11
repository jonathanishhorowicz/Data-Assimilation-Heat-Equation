import numpy as np
import scipy.io 

from fe_mesh import sol_mesh
from forwardSolver import forwardSolver
from measurementGetter import measurementGetter
from dataAssimilator import dataAssimilator

### Set to true to output variables at each iteration
debug = False


##### Computation settings/parameter values

### Domain and discretisation

# Domain size
X = (0.0, 1.0)
T = (0.0, 0.1)

# Mesh size
dx = 1e-2
dt = 1e-3

# Arrays of node positions and time levels
nodePositions = np.linspace(X[0], X[1], (X[1]-X[0])/dx +1)
tVals = np.linspace(T[0], T[1], (T[1]-T[0])/dt +1)


### Discrete gradient flow settings

# Maximum number of iterations for the whole algorithm
maxIter = 1000

# Psudeo timestep
zeta = 1e-1

# P operator matrix
P = np.eye(len(nodePositions)-2)

# Starting guess of the initial condition
u00 = lambda x: 2.0
u00vec = np.vectorize(u00)

# If the relative change in norm(z0) is below this value the algorithm will terminate
ztol = 1e-1

# Choose whether to print the norms of the solution and z0 (at every iterIncth iteration)
printnorms = True
iterinc = 10



### Measurements

# Measurement domain size - unresolved bug means certain values will not work when dx=0.1. These include 0.4,0.6
omega = (0.2, 0.8)

# Exact solution as vectorized lambda function
exactSol = lambda x,t: np.exp(-np.pi**2.0 * t) * np.sin(np.pi * x)
exactVecFunc = np.vectorize(exactSol)

# Noise maximum amplitude
sigma = 1e-1

# This class returns an instance of measMesh which is an array of the measurements with information about their location in the whole domain
measGetter = measurementGetter(exactSol, omega, X, T, dx, dt, sigma)
measMesh = measGetter.getMeasMesh()


### Other parameters

# Source term. f=f(x) only
f = lambda x: 0.0

# Regularisation parameters
gammaM = 1.0
gamma = (1.0, 1.0)

# Conjugate gradient tolerances
cgtol = (1e-8, 1e-12, 1e-8) # [0] for foward problem, [1] for adjoint problem, [2] for discrete gradient flow


### Mesh save options
saveAsNumpyArray = True
saveAsMatlabFile = True
filename = "test" # Change this for some filename generator 



### Run computations here

# To give idea of how long to convergence
exactIC = exactVecFunc(nodePositions,T[0])
print "exact ic norm = ", np.linalg.norm(exactIC)

# Reconstruct the solution from the measurements
da = dataAssimilator(P, zeta, u00, X, T, dx, dt, f, measMesh, gamma, gammaM, maxIter, cgtol, debug, printnorms, iterinc, ztol)
uM, comment = da.solve()

# Solve using the reconstructed initial condition that minimises z0. Data stored as instance of solMesh class - to access numpy array directly use solMesh.mesh
constructedSolver = forwardSolver(X, T, dx, dt, f, cgtol[0])
constructedMesh = constructedSolver.solve(da.UmMinZ0[0])

# Save the reconstructed solution in either Numpy or Matlab format
if saveAsNumpyArray:
    np.save(filename, constructedMesh.mesh)
if saveAsMatlabFile:
    scipy.io.savemat(filename, {'u':constructedMesh.mesh})
