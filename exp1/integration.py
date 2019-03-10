import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from utils import *
import math
from scipy.stats import norm

SIMPSON_M=1000
SIMPSON_LEFT=-100

class function2(function):
    def get(self,x):
        x=np.asarray(x,dtype=np.float32)
        return np.exp(-x*x/2.)/math.sqrt(math.pi*2)

def simpson_integration(x):
    int=simpson_integrator(SIMPSON_LEFT,x,function2(),SIMPSON_M)
    (res,err)=int.calc()
    return res

def quad_integration(x):
    int=quad_integrator(SIMPSON_LEFT,x,function2(),2*SIMPSON_M)
    (res,err)=int.calc()
    return (res,err)

def normal_cdf(x):
    res=norm.cdf(x)
    return res

def get(x):
    res1=simpson_integration(x)
    print(round(res1,15))
    (res2,err)=quad_integration(x)
    print(round(res2,15))
    print(round(err,15))
    print(round(normal_cdf(x),15))
    return (res2,err)

get(5)