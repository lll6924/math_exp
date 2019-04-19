import scipy.optimize
import numpy as np


def p6_1():
    c=np.asarray([-4.3,-2.7,-2.5,-2.2,-4.5],dtype=np.float)
    A_ub=np.asarray([[2.,2.,1.,1.,5.],[9.,15.,4.,3.,2.],[0,-1.,-1.,-1.,0]],dtype=np.float)
    b_ub=np.asarray([1400.,5000.,-400.],dtype=np.float)
    A_eq=np.asarray([[1.,1.,1.,1.,1.]],dtype=np.float)
    b_eq=np.asarray([1000.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub,A_eq,b_eq))


def p6_2():
    c=np.asarray([-4.3,-2.7,-2.5,-2.2,-4.5,2.75],dtype=np.float)
    A_ub=np.asarray([[2.,2.,1.,1.,5.,-1.4],[9.,15.,4.,3.,2.,-5.],[0,-1.,-1.,-1.,0,0],[0,0,0,0,0,1.]],dtype=np.float)
    b_ub=np.asarray([1400.,5000.,-400.,100.],dtype=np.float)
    A_eq=np.asarray([[1.,1.,1.,1.,1.,-1.]],dtype=np.float)
    b_eq=np.asarray([1000.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub,A_eq,b_eq))

def p6_3_1():
    c=np.asarray([-4.5,-2.7,-2.5,-2.2,-4.5],dtype=np.float)
    A_ub=np.asarray([[2.,2.,1.,1.,5.],[9.,15.,4.,3.,2.],[0,-1.,-1.,-1.,0]],dtype=np.float)
    b_ub=np.asarray([1400.,5000.,-400.],dtype=np.float)
    A_eq=np.asarray([[1.,1.,1.,1.,1.]],dtype=np.float)
    b_eq=np.asarray([1000.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub,A_eq,b_eq))

def p6_3_2():
    c=np.asarray([-4.3,-2.7,-2.4,-2.2,-4.5],dtype=np.float)
    A_ub=np.asarray([[2.,2.,1.,1.,5.],[9.,15.,4.,3.,2.],[0,-1.,-1.,-1.,0]],dtype=np.float)
    b_ub=np.asarray([1400.,5000.,-400.],dtype=np.float)
    A_eq=np.asarray([[1.,1.,1.,1.,1.]],dtype=np.float)
    b_eq=np.asarray([1000.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub,A_eq,b_eq))

def p8():
    c=np.asarray([1.8,3.5,0.4,1.],dtype=np.float)
    A_ub=np.asarray([[-0.5,-1.,-2.,-6.],[-2.,-4.,-0.5,-1.],[-5.,-2.,-1.,-2.5]],dtype=np.float)
    b_ub=np.asarray([-40.,-20.,-45.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub))

def p10_1():
    c=np.asarray([5.,5.,5.],dtype=np.float)
    A_ub=np.asarray([[-5.,0,0],[-4.5,-5,0],[-2.7,-3.,-5.]],dtype=np.float)
    b_ub=np.asarray([-300.,-470.,-132.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub))

def p10_2():
    c=np.asarray([5.,5.,5.],dtype=np.float)
    A_ub=np.asarray([[-4.5,0,0]],dtype=np.float)
    b_ub=np.asarray([-170.],dtype=np.float)
    print(scipy.optimize.linprog(c,A_ub,b_ub))

p10_2()