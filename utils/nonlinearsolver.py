import scipy.optimize
import numpy as np
from utils.equationsolver import directSolver

__all__=['nonlinearsolver','Iteration','NewtonIteration','fsolve','fzero']

TOL=1e-8
MAXITE=10000

class nonlinearsolver():
    def __init__(self,x0,fun,jac=None):
        self._x0=x0
        self._fun=fun
        self._jac=jac
    def __call__(self):
        return self.solve()
    def solve(self):
        raise NotImplementedError()

class Iteration():
    def __init__(self,x0,phi):
        self._x0=x0
        self._phi=phi
    def __call__(self):
        x=self._x0
        for _ in range(MAXITE):
            x=self._phi(x)
        return x

class NewtonIteration(nonlinearsolver):
    def __init__(self,x0,fun,jac):
        super().__init__(x0,fun,jac)
        self._solver=directSolver()
    def solve(self,observe=None):
        x=self._x0
        for i in range(MAXITE):
            #print(x)
            if(i==observe):
                print(x)
            dx=self._solver(self._jac(x),-self._fun(x))
            #print(self._jac(x),'--------',-self._fun(x))
            if(np.linalg.norm(dx)<TOL):
                break
            x=x+dx
        return x

class fzero(nonlinearsolver):
    def solve(self):
        return scipy.optimize.root(self._fun,self._x0,jac=self._jac)

class fsolve(nonlinearsolver):
    def solve(self):
        return scipy.optimize.fsolve(self._fun,self._x0,fprime=self._jac)