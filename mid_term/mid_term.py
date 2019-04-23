import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.function import *
from utils.odesolver import *
from utils.equationsolver import *
from utils.nonlinearsolver import *
from scipy.optimize import fmin,minimize,root
import matplotlib.pyplot as plt
import numpy as np

def p1():
    class function1(function):
        def get(self, x):
            print(x)
            if(x==0.):
                return 0.
            if(x==0.5):
                return 0.4159
            if (x == 1):
                return 1.3272
            if (x == 1.5):
                return 2.7294
            if (x == 2.):
                return 4.6184
    int=trapezia_integrator(0.,2.,function1(),5)
    print(int.calc())

def p1_2():
    int = lagrangian_interpolator(x=[0.,1.,2.], y=[0.,1.3272,4.6184], fun=None)
    print(int.get_fun())

def p2():
    class fun1(function2d):
        def get(self, x, y):
            return y + 3 * x
    solver=improved_euler_ode(fun1(),0.,[1.],0.,1.,0.1)
    solver.calc()
    print(solver.get_y())

def p3(n):
    A=np.zeros((n,n),dtype=np.float)
    for i in range(n):
        if(i!=0):
            A[i][i-1]=1.
            A[i-1][i]=1.
        A[i][i]=2.
    b=np.ones(n,dtype=np.float)
    print(np.linalg.cond(A))
    gs=gaussSedeilSolver()
    gs.solve(A,b,np.zeros(n,dtype=np.float),10)

def p5():
    class fun:
        def get(self, x):
            return 2.*x[0]*x[0]+2.*x[1]*x[1]+3.*x[0]*x[1]-4.*x[0]-8.*x[1]
        def __call__(self, x):
            return self.get(x)
    print(minimize(fun(),np.asarray([0.01,0.01]),method='CG',tol=0.01,options={"disp":True}))

def p6():
    class fun:
        def __call__(self,x):
            x=np.asarray(x)
            return np.cos(np.exp(3./(x+1)))*np.sin(2.*x)
    class fun1:
        def __call__(self, x):
            x=np.asarray(x)
            ret=[]
            for t in x:
                int=quad_integrator(0,t,fun(),1000)
                x1,x2=int.calc()
                ret.append(x1-0.54)
            return np.asarray(ret)

    class jac1:
        def __call__(self, x):
            f=fun()
            return [f(x)]
    funx=fun1()
    spl=isometry_sampler(2.2,10,1000)
    xnew=spl.getAll()
    plt.plot(xnew, funx(xnew), '-')
    plt.grid(True)
    plt.show()
    ns=NewtonIteration([2.],fun1(),jac1())
    print(ns.solve())


p6()