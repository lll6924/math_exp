import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils import *
import numpy as np
from matplotlib import pyplot as plt


class fun1(function2d):
    def get(self,x,y):
        return y+2*x

class truefun1(function):
    def get(self,x):
        #print(x)
        return 3*np.exp(x)-2*x-2

class fun2(function2d):
    def get(self,x,y):
        y2=(0.25-x*x)*y[0]/x/x-y[1]/x
        return np.asarray([(0.25-x*x)*y[0]/x-y2*x,(0.25-x*x)*y[0]/x/x-y[1]/x])

class truefun2(function):
    def get(self,x):
        return np.sin(x)*np.sqrt(2*np.pi/x)

class fun3(function2d):
    def get(self,x,y):
        return (32000.-y*y*0.4)/(1400.-18.*x)-9.8

class fun3_2(function2d):
    def get(self,x,y):
        return (- y * y * 0.4) / 320. - 9.8
class fun9(function2d):
    def __init__(self,r1,r2,n1,n2,s1,s2):
        self._r1=r1
        self._r2=r2
        self._n1=n1
        self._n2=n2
        self._s1=s1
        self._s2=s2
    def get(self,x,y):
        x=self._r1*y[0]*(1.-y[0]/self._n1-self._s1*y[1]/self._n2)
        y=self._r2*y[1]*(1.-self._s2*y[0]/self._n1-y[1]/self._n2)
        return np.asarray([x,y])

def p_2_1():
    samples_x = np.arange(0.,1.,0.01)
    func = truefun1()
    solver = rk45(fun1(),0.,[1.],0.,1.,0.01,samples_x,func(samples_x))
    solver.plot(1)
    ys=solver.get_y()
    print(np.abs(func(1.)-ys[0][-1]))

def p_2_3():
    samples_x = np.arange(np.pi / 2., np.pi, 0.01)
    func = truefun2()
    solver = rk45(fun2(), np.pi / 2., [2,-2./np.pi], np.pi / 2., np.pi, 0.01, samples_x, func(samples_x))
    #solver.plot(1)
    ys=solver.get_y()
    print(np.abs(func(np.pi)-ys[0][-1]))

def p_3():
    f3=fun3()
    solver = rk45(f3,0.,[0.],0.,60.,0.01)
    solver.plot(1)
    xs=solver.get_x()
    ys=solver.get_y()[0]
    ans_x=[]
    ans=0
    for y in ys:
        ans+=y
        ans_x.append(ans/100.)
    print(ans/100)
    print(ys[-1])
    print(f3(xs[-1],ys[-1]))
    plt.plot(xs, f3(xs,ys), '-')
    plt.legend('acceleration', loc='best')
    plt.show()
    plt.plot(xs, ans_x, '-')
    plt.legend('height', loc='best')
    plt.show()

    f32 = fun3_2()
    solver2 = rk45(f32,60.,[ys[-1]],60.,85,0.01)
    ans_y = ans_x
    ans_x = xs.tolist()
    ans_y_1 = ys.tolist()
    ans_y_2 = f3(xs,ys).tolist()
    xs=solver2.get_x()
    ys=solver2.get_y()[0]

    for i in range(len(xs)):
        if(ys[i]<=0):
            break
        ans += ys[i]
        ans_y.append(ans / 100.)
        ans_y_1.append(ys[i])
        ans_x.append(xs[i])
        ans_y_2.append(f32(xs[i],ys[i]))
    ans_x = np.asarray(ans_x)
    ans_y = np.asarray(ans_y)
    ans_y_1 = np.asarray(ans_y_1)

    print(ans_x[-1])
    print(f32(ans_x[-1],ans_y_1[-1]))
    print(ans_y_1[-1])
    print(ans_y[-1])

    plt.plot(ans_x, ans_y_2, '-')
    plt.legend('acceleration', loc='best')
    plt.show()
    plt.plot(ans_x, ans_y_1, '-')
    plt.legend('velocity', loc='best')
    plt.show()
    plt.plot(ans_x, ans_y, '-')
    plt.legend('height', loc='best')
    plt.show()


def p_9():
    f9=fun9(r1=1.,r2=1.0,n1=100,n2=100,s1=1.5,s2=1.7)
    solver = rk45(f9, 0., [10,10], 0., 60., 0.01)
    solver.plot(2,['x','y'])


#p_2_3()
#for i in range(1000):
#    p_2_3()
#p_3()
p_9()