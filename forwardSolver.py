from fe_mesh import *

# This class solves the forward problem (the heat equation)
class forwardSolver:
    
    # Constructor
    def __init__(self, X, T, dx, dt, f, forcgtol):
        self.X = X
        self.T = T
        self.dx = dx
        self.dt = dt
        self.f = f
        
        self.newmesh = sol_mesh(self.X, self.T, self.dx, self.dt)
        self.solver = heat_solver(self.newmesh, self.f, forcgtol)
    
    # Returns the whole solution
    def solve(self, ic):
        self.newmesh.applyIC(ic)
        self.solver.backwardEuler()
        return self.newmesh
        