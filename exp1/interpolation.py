import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from utils import *
import math

class function1(function):
    def get(self,x):
        x=np.asarray(x,dtype=np.float32)
        return 1./(1+x*x)

class strange_sampler(sampler):
    def __init__(self, width=10., samples=2):
        assert(isdigit(width))
        assert(samples>=2)
        width=float(width)
        self._width=width
        self._samples=samples
    def get(self,x):
        return self._width*math.cos((2.*x+1.)/(2.*self._samples)*math.pi)

int=linear_interpolator()
samp=strange_sampler(10.,11)
samp=isometry_sampler(-10.,10.,11)
x=samp.getAll()
func=function1()
int.feed(x=x,y=func.get(x),fun=func)
int.plot()