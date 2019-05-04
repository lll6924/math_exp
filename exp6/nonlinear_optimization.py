import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def p2_1():
    class fun:
        def get(self, x):
            return np.square(x[0]*x[1])*np.square(1.-x[0])*np.square(1.-x[0]-x[1]*np.power(1.-x[0],5))
        def __call__(self, x):
            return self.get(x)
    class jac:
        def __call__(self, x):
            return self.get(x)
        def get(self, x):
            j1=2.*x[0]*np.square(x[1])*np.square(1.-x[0])*np.square(1.-x[0]-x[1]*np.power(1.-x[0],5)) \
                -np.square(x[0]*x[1])*(1.-x[0])*np.square(1.-x[0]-x[1]*np.power(1.-x[0],5)) \
                +np.square(x[0]*x[1])*np.square(1.-x[0])*(1.-x[0]-x[1]*np.power(1.-x[0],5))*(-1.+5*x[1]*np.power(1.-x[0],4))
            j2=2.*x[1]*np.square(x[0])*np.square(1.-x[0])*np.square(1.-x[0]-x[1]*np.power(1.-x[0],5)) \
                -np.square(x[0]*x[1])*np.square(1.-x[0])*(1.-x[0]-x[1]*np.power(1.-x[0],5))*np.power(1.-x[0],5)
            return np.asarray([j1,j2])
    xs=[]
    ys=[]
    nits=[]
    for i in range(101):
        for j in range(101):
            x=i/25.-2.
            y=j/25.-2.
            res=minimize(fun(),np.asarray([x,y]),method='BFGS',jac=jac())
            if(np.abs(res['fun'])>1e-5):
                continue
            nits.append(res['nit'])
            xs.append(res['x'][0])
            ys.append(res['x'][1])
    print(round(np.mean(nits),5))
    plt.plot(xs,ys,'.')
    plt.show()
    #print(minimize(fun(),np.asarray([0.50,0.50]),method='CG',jac=jac(),options={"disp":True}))
def p2_2():
    class fun:
        c1=0.7
        c2=0.73
        a1=[4.,4.]
        a2=[2.5,3.8]
        def get(self, x):
            n1=np.linalg.norm(x-self.a1)
            n2=np.linalg.norm(x-self.a2)
            return -(1./(np.square(n1)+self.c1)+1./(np.square(n2)+self.c2))
        def __call__(self, x):
            return self.get(x)
    class jac:
        c1=0.7
        c2=0.73
        a1=[4.,4.]
        a2=[2.5,3.8]
        def __call__(self, x):
            return self.get(x)
        def get(self, x):
            n1=np.linalg.norm(x-self.a1)
            n2=np.linalg.norm(x-self.a2)
            j1=2.*(x[0]-self.a1[0])/np.square(np.square(n1)+self.c1)+2.*(x[0]-self.a2[0])/np.square(np.square(n2)+self.c2)
            j2=2.*(x[1]-self.a1[1])/np.square(np.square(n1)+self.c1)+2.*(x[1]-self.a2[1])/np.square(np.square(n2)+self.c2)
            return np.asarray([j1,j2])
    toplot=[]
    nits=[]
    for i in range(21):
        #print(i)
        for j in range(21):
            x=i/2.5
            y=j/2.5
            res=minimize(fun(),np.asarray([x,y]),method='BFGS')#,jac=jac())
            nits.append(res['nit'])
            if(res['x'][0]<3):
                label="2.607,3.814"
            else:
                label="3.906,3.987"
            toplot.append({"x0":x,"y0":y,"result":label})
    toplot=pd.DataFrame(toplot)
    print(round(np.mean(nits),5))
    #sns.scatterplot(x='x0',y='y0',hue='result',data=toplot)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    #plt.plot([2.607], [3.814], "x", color=sns.color_palette("muted")[0], markersize=12)
    #plt.plot([3.906], [3.987], "x", color=sns.color_palette("muted")[1], markersize=12)
    #plt.tight_layout()
    #plt.show()


def p3():
    class fun:
        def get(self, x):
            x1=x[0]
            x2=x[1]
            x3=x[2]
            x4=x[3]
            return 100*np.square(x2-np.square(x1))+np.square(1-x1)+90*np.square(x4-np.square(x3))+np.square(1-x3)\
                    +10.1*(np.square(1-x2)+np.square(1-x4))+19.8*(x2-1)*(x4-1)
        def __call__(self, x):
            return self.get(x)

    class jac:
        def __call__(self, x):
            return self.get(x)
        def get(self, x):
            x1=x[0]
            x2=x[1]
            x3=x[2]
            x4=x[3]
            j1=-400*x1*(x2-np.square(x1))-2*(1-x1)
            j2 = 200 * (x2 - np.square(x1)) + 10.1 * (-2.*(1 - x2)) + 19.8 * (x4 - 1)
            j3 = -360*x3 * (x4 - np.square(x3)) - 2*(1 - x3)
            j4 = 180 * (x4 - np.square(x3))+ 10.1 * (-2.*(1 - x4)) + 19.8 * (x2 - 1)
            return np.asarray([j1,j2,j3,j4])

    x01=np.asarray([-3.,-1.,-3.,-1.])
    x02=np.asarray([3.,1.,3.,1.])
    x03=np.asarray([2,2,2,2])
    cons = ({'type': 'ineq', 'fun': lambda x: -x[0]*x[1]+x[0]+x[1]-1.5},
            {'type': 'ineq', 'fun': lambda x: x[0]*x[1]+10},
            {'type': 'ineq', 'fun': lambda x: -x[0]*x[1]*x[2]*x[3]+100},
            {'type': 'ineq', 'fun': lambda x: x[0] * x[1] * x[2] * x[3] + 100},
            #{'type': 'eq', 'fun': lambda x: x[0] + x[1]},
            #{'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] * x[3] - 16}
            )
    res = minimize(fun(), x03, method='SLSQP',jac=jac(),bounds=[(-10.,10.),(-10.,10.),(-10.,10.),(-10.,10.)],constraints=cons)
    print(np.round(res['x'],5))
    print(np.round(res['fun'],5))


A=[1.300,1.103,1.216,0.954,0.929,1.056,1.038,1.089,1.090,1.083,1.035,1.176]
B=[1.225,1.290,1.216,0.728,1.144,1.107,1.321,1.305,1.195,1.390,0.928,1.715]
C=[1.149,1.260,1.419,0.922,1.169,0.965,1.133,1.732,1.021,1.131,1.006,1.908]
A=np.asarray(A)-1
B=np.asarray(B)-1
C=np.asarray(C)-1
cABC=np.cov([A,B,C])
covAB=cABC[0][1]
covBC = cABC[1][2]
covCA = cABC[2][0]
vA=cABC[0][0]
vB=cABC[1][1]
vC = cABC[2][2]
mA=np.mean(A)
mB=np.mean(B)
mC=np.mean(C)
print(cABC)
print(mA)
print(mB)
print(mC)

def _p8(bar,allow_risk_free=False,is_question_3=False):
    allow_risk_free=int(allow_risk_free)
    is_question_3=int(is_question_3)

    class fun:
        def get(self, x):
            A=x[0]
            B=x[1]
            C=x[2]
            return A*A*vA+B*B*vB+C*C*vC+2*A*B*covAB+2*B*C*covBC+2*C*A*covCA
        def __call__(self, x):
            return self.get(x)
    class jac:
        def get(self, x):
            A=x[0]
            B=x[1]
            C=x[2]
            j1=2*A*vA+2*B*covAB+2*C*covCA
            j2=2*B*vB+2*A*covAB+2*C*covBC
            j3=2*C*vC+2*B*covBC+2*A*covCA
            return np.asarray([j1,j2,j3,0])
        def __call__(self, x):
            return self.get(x)

    cons = ({'type': 'ineq', 'fun': lambda x: x[0]*mA+x[1]*mB+x[2]*mC+x[3]*0.05*allow_risk_free-bar
                                              -is_question_3*np.sum(np.abs(x-[0.5,0.35,0.15,0])/100)},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]*allow_risk_free-1.
                                            +is_question_3*np.sum(np.abs(x-[0.5,0.35,0.15,0])/100)}
            )
    res = minimize(fun(), [0.3,0.3,0.4,0.],jac=jac(), method='SLSQP',bounds=[(0.,1.),(0.,1.),(0.,1.),(0.,1.)],constraints=cons)
    return(res['x'],res['fun'])

def p8_1():
    toplot=[]
    toplot2=[]
    for i in range(1001):
        rate=i/1000*0.3
        x,std=_p8(bar=rate)
        toplot.append({"rate":rate,"fraction":x[0],"share":"A"})
        toplot.append({"rate":rate,"fraction":x[1],"share":"B"})
        toplot.append({"rate":rate,"fraction":x[2],"share":"C"})
        toplot2.append({"rate":rate,"std":std,"share|std":"std"})
    toplot=pd.DataFrame(toplot)
    toplot2=pd.DataFrame(toplot2)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 4), sharex=True)
    sns.lineplot(x='rate',y='fraction',hue='share',data=toplot,ax=ax2)
    sns.lineplot(x='rate',y='std',data=toplot2,ax=ax1)

    plt.show()

def p8_3():
    print(_p8(bar=0.15,is_question_3=True))
print(_p8(bar=0.15,allow_risk_free=True))
p8_3()
