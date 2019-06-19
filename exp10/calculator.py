import numpy as np
import random
import math
from scipy.stats import t
import time
N=20
ret=0
tt=math.sqrt(3)/2
delta=0.1
epsilon=0.1
alpha=0.05
def raw_calculator(A):
    A=np.asarray(A)
    n=len(A)
    assert(A.shape==(n,n))
    B=np.arange(n)
    def count(k):
        global ret
        if k==n:
            ret=ret+1
            return
        for i in range(n-k):
            if A[k][B[i]]==1:
                t=B[i]
                B[i]=B[n-k-1]
                count(k+1)
                B[i]=t
    count(0)
    return ret

def ryser_calculator(A):
    A=np.asarray(A)
    n=len(A)
    assert(A.shape==(n,n))
    last=0
    B=list(np.zeros(n))
    sums=list(np.zeros(n))
    size=0
    ans=0
    for i in range(1 << n):
        current = (i ^ (i >> 1))
        difference=current^last
        if difference==0:
            continue
        place=0
        while(difference>1):
            place+=1
            difference/=2
        if B[place]==1:
            B[place]=0
            for j in range(n):
                if A[j][place]==1:
                    sums[j]-=1
            size-=1
        else:
            B[place]=1
            for j in range(n):
                if A[j][place]==1:
                    sums[j]+=1
            size+=1
        toadd=1
        for j in range(n):
            toadd=toadd*sums[j]
        if (size%2)==0:
            ans+=toadd
        else:
            ans-=toadd
        last=current
    if (n % 2) == 1:
        ans=-ans
    return int(ans)

def gg_calculator(A,number=20000):
    A=np.asarray(A)
    n=len(A)
    assert(A.shape==(n,n))
    ans=[]
    B = np.zeros(A.shape)
    for _ in range(number):
        for i in range(n):
            for j in range(n):
                if A[i][j]==1:
                    if random.randint(0,1)==0:
                        B[i][j]=-1
                    else:
                        B[i][j]=1
                else:
                    B[i][j]=0
        ans.append(np.square(np.linalg.det(B)))
    s=np.std(ans)
    mean=np.mean(ans)
    rg=t.ppf(1 - 0.5 * alpha, number)*s/math.sqrt(number)
    return np.mean(ans),rg*2

def kklll_calculator(A,number=20000):
    A = np.asarray(A)
    n = len(A)
    assert (A.shape == (n, n))
    ans = []
    B = np.zeros(A.shape,dtype=np.complex)
    for T in range(number):
        for i in range(n):
            for j in range(n):
                if A[i][j] == 1:
                    r = random.randint(0, 2)
                    if r == 0:
                        B[i][j] = complex(-0.5, tt)
                    elif r == 1:
                        B[i][j] = complex(-0.5, -tt)
                    else:
                        B[i][j] = 1
                else:
                    B[i][j] = 0
        det = np.linalg.det(B)
        ans.append((det * det.conjugate()).real)
    s = np.std(ans)
    mean = np.mean(ans)
    rg = t.ppf(1 - 0.5 * alpha, number) * s / math.sqrt(number)
    return np.mean(ans),rg*2

def normal_calculator(A,number=20000):
    A = np.asarray(A)
    n = len(A)
    assert (A.shape == (n, n))
    # print(number)
    ans = []
    B = np.zeros(A.shape)
    for T in range(number):
        for i in range(n):
            for j in range(n):
                if A[i][j] == 1:
                    B[i][j]=np.random.normal()
                else:
                    B[i][j] = 0
        det = np.linalg.det(B)
        ans.append(np.square(det))
    s = np.std(ans)
    mean = np.mean(ans)
    rg = t.ppf(1 - 0.5 * alpha, number) * s / math.sqrt(number)
    return np.mean(ans),rg*2



if __name__=="__main__":
    np.random.seed(19971222)
    random.seed(19971222)
    A=np.random.randint(0,2,(N,N))
    t1=time.time()
    res,length=kklll_calculator(A)
    t2=time.time()
    print(t2-t1)
    print(res)
    print(abs(res-4461838126392)/4461838126392)
    print(length/res)
    #t2=ryser_calculator(A)
    #print(abs(t1-t2)/t2)
    #print(raw_calculator(A))