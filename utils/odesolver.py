from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np

__all__ = ["odesolver", "euler_ode", "improved_euler_ode", "classic_rk", "rk23", "rk45"]

class odesolver:
    def __init__(self, fun, x0, y0, left, right, h, expected_x=None, expected_y=None):
        self._fun = fun
        self._x0 = float(x0)
        self._y0 = np.asarray(y0,dtype=np.float)
        self._h = float(h)
        self._left = left
        self._right = right
        self._expected_x = expected_x
        self._expected_y = expected_y
        self._x_cache=self._y_cache=None
    def get_y(self):
        if self._y_cache is None or self._x_cache is None:
            self.calc()
        return self._y_cache
    def get_x(self):
        if self._y_cache is None or self._x_cache is None:
            self.calc()
        self._x_cache=np.asarray(self._x_cache)
        return self._x_cache
    def calc(self):
        raise NotImplementedError()
    def plot(self,number, name=None):
        if self._y_cache is None or self._x_cache is None:
            self.calc()
        cnt=0
        names=[]
        if(name is not None):
            names+=name
        for ys in self._y_cache:
            cnt+=1
            plt.plot(self._x_cache, ys, '-')
            if name is None:
                names.append('solver'+str(cnt))
            if(cnt>=number):
                break
        if self._expected_x is not None and self._expected_y is not None:
            plt.plot(self._expected_x, self._expected_y, '--')
            names.append('expected')
        plt.legend(names, loc='best')
        plt.show()

class euler_ode(odesolver):
    # forward euler method
    def __init__(self, fun, x0, y0, left, right, h, expected_x=None, expected_y=None):
        super().__init__(fun, x0, y0, left, right, h, expected_x, expected_y)
    def calc(self):
        self._y_cache=[]
        self._x_cache=[]
        x_now=self._x0
        y_now=self._y0
        while(x_now>=self._left and x_now<=self._right):
            self._x_cache.append(x_now)
            self._y_cache.append(y_now)
            toadd=self._h * self._fun(x_now, y_now)
            x_now+=self._h
            y_now=y_now+toadd
        self._y_cache = np.transpose(self._y_cache,[1,0])
        #print(self._y_cache)

class improved_euler_ode(odesolver):
    def __init__(self, fun, x0, y0, left, right, h, expected_x=None, expected_y=None):
        super().__init__(fun, x0, y0, left, right, h, expected_x, expected_y)

    def calc(self):
        self._y_cache = []
        self._x_cache = []
        x_now = self._x0
        y_now = self._y0
        while (x_now >= self._left and x_now <= self._right):
            self._x_cache.append(x_now)
            self._y_cache.append(y_now)
            fake_y = y_now + self._h * self._fun(x_now, y_now)
            toadd = self._h / 2. * (self._fun(x_now, y_now) + self._fun(x_now + self._h, fake_y))
            x_now += self._h
            y_now = y_now+ toadd
        self._y_cache = np.transpose(self._y_cache,[1,0])



class classic_rk(odesolver):
    def __init__(self, fun, x0, y0, left, right, h, expected_x=None, expected_y=None):
        super().__init__(fun, x0, y0, left, right, h, expected_x, expected_y)
    def calc(self):
        self._y_cache=[]
        self._x_cache=[]
        x_now=self._x0
        y_now=self._y0
        while(x_now>=self._left and x_now<=self._right):
            self._x_cache.append(x_now)
            self._y_cache.append(y_now)
            K1=self._fun(x_now, y_now)
            K2=self._fun(x_now + self._h/2., y_now + K1*self._h/2.)
            K3=self._fun(x_now + self._h/2., y_now + K2*self._h/2.)
            K4=self._fun(x_now + self._h, y_now + K3*self._h)
            toadd=self._h / 6. * (K1 + 2*K2 + 2*K3 + K4)
            x_now+=self._h
            y_now=y_now+toadd
        self._y_cache = np.transpose(self._y_cache,[1,0])


class rk23(odesolver):
    def __init__(self, fun, x0, y0, left, right, h=None, expected_x=None, expected_y=None):
        super().__init__(fun, x0, y0, left, right, h, expected_x, expected_y)
    def calc(self):
        self._y_cache=[]
        self._x_cache=[]
        x_now=self._x0
        while(x_now>=self._left and x_now<=self._right):
            self._x_cache.append(x_now)
            x_now+=self._h
        if(self._h>0):
            self._rk = solve_ivp(self._fun, (self._x0, self._right), self._y0, method="RK23", t_eval=self._x_cache)
        else:
            self._rk = solve_ivp(self._fun, (self._x0, self._left), self._y0, method="RK23", t_eval=self._x_cache)
        self._y_cache=self._rk.y

class rk45(odesolver):
    def __init__(self, fun, x0, y0, left, right, h=None, expected_x=None, expected_y=None):
        super().__init__(fun, x0, y0, left, right, h, expected_x, expected_y)
    def calc(self):
        self._y_cache=[]
        self._x_cache=[]
        x_now=self._x0
        while(x_now>=self._left and x_now<=self._right):
            self._x_cache.append(x_now)
            x_now+=self._h
        if(self._h>0):
            self._rk = solve_ivp(self._fun, (self._x0, self._right), self._y0, method="RK45", t_eval=self._x_cache)
        else:
            self._rk = solve_ivp(self._fun, (self._x0, self._left), self._y0, method="RK45", t_eval=self._x_cache)
        self._y_cache=self._rk.y