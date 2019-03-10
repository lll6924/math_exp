from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt

__all__ = ['isdigit', 'function', 'sampler','isometry_sampler','interpolator','lagrangian_interpolator',
           'linear_interpolator','spline_interpolator']

PLOT_SAMPLES=10000

def isdigit(x):
    #fake function
    return type(x)==int or type(x)==float or type(x)==np.float32 or type(x)==np.int32

class function:
    def __call__(self,x):
        return self.get(x)
    def get(self,x):
        raise NotImplementedError()

class sampler:
    def get(self,x):
        raise NotImplementedError()
    def getAll(self):
        ret=[]
        for i in range(self._samples):
            ret.append(self.get(i))
        ret.sort()
        return ret

class isometry_sampler(sampler):
    def __init__(self,left=0.,right=1.,samples=2):
        assert(isdigit(left))
        assert(isdigit(right))
        assert(type(samples)==int)
        assert(samples>=2)
        left=float(left)
        right=float(right)
        self._left=left
        self._right=right
        self._samples=samples
    def get(self,x):
        assert(type(x)==int)
        return float(x)/(self._samples-1.)*(self._right-self._left)+self._left

class interpolator:
    def feed(self,x,y,fun):
        assert(len(x)==len(y))
        for i in range(len(x)):
            assert(isdigit(x[i]))
            assert(isdigit(y[i]))
        self._x=x
        self._y=y
        self._n=len(x)
        self._fun = fun
        self._left=x[0]
        self._right=x[self._n-1]

    def get(self,x):
        raise NotImplementedError()

class lagrangian_interpolator(interpolator):
    def feed(self,x,y,fun):
        super().feed(x,y,fun)
        self._f = self.get
    def get(self,x):
        res=0.
        for i in range(self._n):
            multiplier=1.
            for j in range(self._n):
                if(i!=j):
                    multiplier=multiplier*(x-self._x[j])/(self._x[i]-self._x[j])
            res+=multiplier*self._y[i]
        return res
    def plot(self):
        xnew = np.linspace(self._left, self._right, num=PLOT_SAMPLES, endpoint=True)
        plt.plot(self._x, self._y, 'o',xnew, self._fun(xnew), '-', xnew, self._f(xnew), '--')
        plt.legend(['data', 'function', 'lagrangian'], loc='best')
        plt.show()


class linear_interpolator(interpolator):
    def feed(self,x,y,fun):
        super().feed(x,y,fun)
        self._f = interp1d(self._x,self._y)
    def get(self,x):
        return self._f(x)
    def plot(self):
        xnew = np.linspace(self._left, self._right, num=PLOT_SAMPLES, endpoint=True)
        plt.plot(self._x, self._y, 'o',xnew, self._fun(xnew), '-', xnew, self._f(xnew), '--')
        plt.legend(['data', 'function', 'linear'], loc='best')
        plt.show()

class spline_interpolator(interpolator):
    def feed(self,x,y,fun):
        super().feed(x,y,fun)
        self._f = interp1d(self._x,self._y,kind='cubic')
    def get(self,x):
        return self._f(x)
    def plot(self):
        xnew = np.linspace(self._left, self._right, num=PLOT_SAMPLES, endpoint=True)
        plt.plot(self._x, self._y, 'o',xnew, self._fun(xnew), '-', xnew, self._f(xnew), '--')
        plt.legend(['data', 'function', 'spline'], loc='best')
        plt.show()
