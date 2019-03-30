import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.equationsolver import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def p1_solver(A,b):
    print()
    n=len(b)
    ds = directSolver()
    js = jacobiSolver()
    gs = gaussSedeilSolver()
    ps = pcgSolver()
    dsa=ds(A,b).reshape(n)
    jsa=js(A,b,x0=np.full((n,1),0.5)).reshape(n)
    gsa=gs(A,b,x0=np.full((n,1),0.5)).reshape(n)
    psa=ps(A,b,x0=np.full((n,1),0.5))
    print("directSolver=", dsa)
    print("jacobiSolver=",jsa)
    print("gaussSedeilSolver=",gsa)
    print("pcgSolver=",psa)
    return dsa

def p1(n,epsilon):
    xs = np.arange(1,n*0.1+1.,0.1)
    multiplier = np.ones(n)
    A1=[]
    for _ in range(n):
        A1.append(multiplier)
        multiplier = multiplier*xs
    A1 = np.transpose(A1,[1,0])
    b1 = np.sum(A1,axis=1).reshape((n,1))
    A2=1./(np.arange(0,n)+np.arange(1,n+1).reshape((n,1)))
    b2 = np.sum(A2,axis=1).reshape((n,1))

    print("solving A1...")
    condA1=np.linalg.cond(A1)
    print("cond(A1)=",condA1)
    x0=p1_solver(A1,b1)
    A0=A1.copy()
    A1[n-1][n-1]+=epsilon
    xa=p1_solver(A1,b1)
    DA_A = np.linalg.norm(A0-A1)/np.linalg.norm(A0)
    A1[n-1][n-1]-=epsilon
    b0=b1.copy()
    b1[n-1]+=epsilon
    xb=p1_solver(A1,b1)
    db_b = np.linalg.norm(b1-b0)/np.linalg.norm(b0)
    print("|△x|/x(disturb A1)=",np.linalg.norm(xa-x0)/np.linalg.norm(x0)," estimation:",condA1/(1-condA1*DA_A)*DA_A)
    print("|△x|/x(disturb b1)=",np.linalg.norm(xb-x0)/np.linalg.norm(x0)," estimation:",condA1*db_b)

    print("solving A2...")
    condA2 = np.linalg.cond(A2)
    print("cond(A2)=", condA2)
    x0 = p1_solver(A2, b2)
    A0 = A2.copy()
    A2[n - 1][n - 1] += epsilon
    xa = p1_solver(A2, b2)
    DA_A = np.linalg.norm(A0 - A2) / np.linalg.norm(A0)
    A2[n - 1][n - 1] -= epsilon
    b0 = b2.copy()
    b2[n - 1] += epsilon
    xb = p1_solver(A2, b2)
    db_b = np.linalg.norm(b2 - b0) / np.linalg.norm(b0)
    print("|△x|/x(disturb A2)=", np.linalg.norm(xa - x0) / np.linalg.norm(x0), " estimation:",
          condA2 / (1 - condA2 * DA_A) * DA_A)
    print("|△x|/x(disturb b2)=", np.linalg.norm(xb - x0) / np.linalg.norm(x0), " estimation:", condA2 * db_b)

#p1(9,1e-8)

def p3(n):
    A=np.diag(np.full(n,3)).astype(np.float)
    for i in range(n-1):
        A[i][i+1]=-0.5
        A[i+1][i]=-0.5
        if(i<n-2):
            A[i][i+2]=-0.25
            A[i+2][i]=-0.25
    b1=np.sum(A,axis=1).reshape((n,1))
    b2=np.full(n,1).reshape((n,1))
    b3=np.matmul(A,np.arange(0,n).reshape((n,1)))
    x01=np.zeros(n).reshape((n,1))
    x02=np.full(n,1000).reshape((n,1))
    ds=directSolver()
    js=jacobiSolver()
    gs=gaussSedeilSolver()

    dsa=ds.solve(A,b1)
    print(dsa)
    for i in range(10):
        print()
        print(i)
        jsa=js.solve(A,b1,x01)
        gsa=gs.solve(A,b1,x01)
        A += np.diag(np.full(n, 3)).astype(np.float)
    #print(jsa)
    #print(gsa)

#p3(20)

def get(x,y):
    return 2*np.pi*np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)
def gettrue(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)

def pn_2(n=100):
    h=1./n
    A=np.full((n+1)*(n+1)*(n+1)*(n+1),0).reshape((n+1)*(n+1),(n+1)*(n+1))
    A=np.asarray(A,dtype=np.float)
    b=np.full((n+1)*(n+1),0).reshape(((n+1)*(n+1),1))
    b=np.asarray(b,dtype=np.float)
    for i in range(n+1):
        for j in range(n+1):
            id=i*(n+1)+j
            if(i==0 or j==0 or i==n or j==n):
                A[id][id]=1
                b[id][0]=0
                continue
            A[id][id-1]=-n*n
            A[id][id+1]=-n*n
            A[id][id-n-1]=-n*n
            A[id][id+n+1]=-n*n
            A[id][id]=n*n*4.
            b[id][0]=get(float(i)/n,float(j)/n)
    solver=pcgSolver()
    dsa=solver(A,b)
    print(dsa)
    div=0.
    maxdiv=0.
    for i in range(n+1):
        for j in range(n+1):
            if(np.abs(dsa[i*(n+1)+j])<1e-5):
                continue
            tem=np.abs(dsa[i*(n+1)+j]-gettrue(float(i)/n,float(j)/n))/gettrue(float(i)/n,float(j)/n)
            div+=tem
            if(tem>maxdiv):
                maxdiv=tem
    print(div/(n+1)/(n+1))
    print(maxdiv)
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0,1+1./n,1./n)
    Y = np.arange(0,1+1./n,1./n)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    for i in range(n+1):
        for j in range(n+1):
            Z[i][j]=dsa[i*(n+1)+j]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

#pn_2(10)

def get_1(x):
    return 5.-6.*x+7.*x*x

def get_true_1(x):
    return x*x-x*x*x+0.5*x*x*x*x+x*x*np.log(x)

def pn_1(n):
    h=1./n
    A=np.full((n+1)*(n+1),0).reshape((n+1,n+1))
    A=np.asarray(A,np.float)
    b=np.full(n+1,0).reshape((n+1,1))
    b=np.asarray(b,np.float)
    for i in range(n+1):
        x=1.+float(i)/n
        if(i==0):
            A[i][i]=1
            b[i][0]=0.5
            continue
        if(i==n):
            A[i][i]=1
            b[i][0]=4.+4.*np.log(2)
            continue
        A[i][i]=-2.*n*n-6./x/x
        A[i][i-1]=float(n)*n-float(n)/x
        A[i][i+1]=float(n)*n+float(n)/x
        b[i][0]=get_1(x)

    #print(A)
    #print(b)

    solver=gaussSedeilSolver()
    psa=solver(A,b,x0=np.zeros(n+1).reshape((n+1,1)))

    div=0.
    maxdiv=0.
    for i in range(n+1):
        tem=np.abs(psa[i]-get_true_1(1.+float(i)/n))/np.abs(get_true_1(1.+float(i)/n))
        div+=tem
        if(tem>maxdiv):
            maxdiv=tem
    print(div/(n+1))
    print(maxdiv)


    xs=np.arange(1.,2.+h,h)
    #print(xs)
    #print(get_true_1(xs))
    #print(psa)
    plt.plot(xs, get_true_1(xs), '--', xs, psa, '-')
    plt.legend(['real f(x)','answer'], loc='best')
    plt.show()

pn_2(100)