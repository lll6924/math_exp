import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.nonlinearsolver import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class p3_1_fun1:
    def __call__(self,x):
        y=1./(1.+x)
        return (1.-np.power(y,180))/150.
class p3_1_fun2:
    def __call__(self,x):
        y=x/(x+1.)
        return np.asarray(150. - (1. - np.power(y, 180)) *x).reshape((1))

class p3_1_jac2:
    def __call__(self, x):
        x2=x*x
        y=x/(x+1.)
        return -np.asarray(1.-np.power(y,180)+x*(1.-180.*np.power(y,179)/np.square(1+x))).reshape((1, 1))

class p3_2_fun1:
    def __call__(self,x):
        y=x/(x+1.)
        return np.asarray(5000./45. - (1. - np.power(y, 180)) *x).reshape((1))

class p3_2_jac1:
    def __call__(self, x):
        x2=x*x
        y=x/(x+1.)
        return -np.asarray(1.-np.power(y,180)+x*(1.-180.*np.power(y,179)/np.square(1+x))).reshape((1, 1))

class p3_2_fun2:
    def __call__(self,x):
        y=x/(x+1.)
        return np.asarray(500./45. - (1. - np.power(y, 20)) *x).reshape((1))

class p3_2_jac2:
    def __call__(self, x):
        x2=x*x
        y=x/(x+1.)
        return -np.asarray(1.-np.power(y,20)+x*(1.-20.*np.power(y,19)/np.square(1+x))).reshape((1, 1))


def p3_1():
    s1=Iteration(0.5,p3_1_fun1())
    print(s1())
    s=p3_1_fun2()
    print(15./(150.-s(1./s1())[0]))
    s2=fzero(0.5,p3_1_fun2(),p3_1_jac2())
    print(1./s2()['x'])
    print(15./(150.-s(s2()['x'])[0]))


def p3_2():
    s1=fzero(0.5,p3_2_fun1(),p3_2_jac1())
    print(1./s1()['x']*12)
    s2=fzero(0.5,p3_2_fun2(),p3_2_jac2())
    print(1./s2()['x'])

p3_2()

def p5():
    gamma=1.4
    S=np.pi*0.04*0.04
    C=10000.*np.power(S*0.5,gamma)
    print(C)
    F=25.
    a=0.8
    b=0.25
    class fun():
        def __call__(self,x):
            return np.asarray(F*a*b/np.sqrt(b*b+x*x)-C/np.power(S*(0.5-x),gamma)*S*b).reshape((1))
    class jac():
        def __call__(self,x):
            return np.asarray(-0.5*F*a*b*np.power(x*x+b*b,-1.5)-1.4*C*b*np.power(S,-0.4)*np.power(0.5-x,-2.4)).reshape((1,1))

    solver=fzero(0.25,fun(),jac())
    x=solver()['x'][0]
    print(x)
    print(np.arctan(x/b))
    print(np.arctan(x / b)/np.pi*180)

p5()

def padd(lamb,n=10):
    h=1./n
    n1=n+1
    n2=(n+1)*(n+1)
    class fun():
        def __call__(self,X):
            ret=np.empty(shape=(n2),dtype=np.float)
            for i in range(n2):
                x=i//n1
                y=i%n1
                if(x==0 or x==n or y==0 or y==n):
                    ret[i]=X[i]
                    continue
                ret[i]=n*n*(X[i+1]+X[i-1]+X[i+n1]+X[i-n1]-4*X[i])+lamb*np.exp(X[i])
            return ret

    class jac():
        def __call__(self,X):
            ret=np.zeros(shape=(n2,n2),dtype=np.float)
            for i in range(n2):
                x=i//n1
                y=i%n1
                if(x==0 or x==n or y==0 or y==n):
                    ret[i][i]=1
                    continue
                ret[i][i+1]=ret[i][i-1]=ret[i][i+n1]=ret[i][i-n1]=n*n
                ret[i][i]=lamb*np.exp(X[i])-4.*n*n
            return ret

    solver=fsolve(np.zeros(shape=(n2),dtype=np.float),fun(),jac())
    ans=solver()
    #print(solver())
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 1 + 1. / n, 1. / n)
    Y = np.arange(0, 1 + 1. / n, 1. / n)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    for i in range(n + 1):
        for j in range(n + 1):
            Z[i][j] = ans[i * n1 + j]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

padd(0,20)