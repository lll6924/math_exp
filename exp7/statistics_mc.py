import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def p1_1():
    SAMPLES=100000
    samples=np.random.randint(0,2,(SAMPLES,200))
    print(samples)
    num_of_1=np.sum(samples,axis=1)
    bins=list(range(num_of_1.min(),num_of_1.max()+1))
    bins.append(np.inf)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(num_of_1,cumulative=True,bins=bins,histtype='step',density=True,label='Empirical')
    X=np.arange(0,201)
    Y=[]
    acu=0
    for x in X:
        res=0.
        for i in range(1,201):
            res+=np.log(i)
        for i in range(1,x+1):
            res-=np.log(i)
        for i in range(1,200-x+1):
            res-=np.log(i)
        res-=200*np.log(2)
        Y.append(acu+np.exp(res))
        acu+=np.exp(res)
    n80=n90=n100=n110=n120=0
    for t in num_of_1:
        if t<=80:
            n80+=1
        if t<=90:
            n90+=1
        if t<=100:
            n100+=1
        if t<=110:
            n110+=1
        if t<=120:
            n120+=1
    print(n80/SAMPLES,Y[80])
    print(n90 / SAMPLES, Y[90])
    print(n100 / SAMPLES, Y[100])
    print(n110 / SAMPLES, Y[110])
    print(n120 / SAMPLES, Y[120])
    ax.plot(X, Y, 'k--', linewidth=1.5, label='Theoretical')
    ax.set_title(str(SAMPLES))
    plt.legend()
    plt.show()

def p1_2():
    SAMPLES=100000
    samples=np.random.randint(0,2,(SAMPLES,200))
    print(samples)
    toplot=[]
    for s in samples:
        lst=-1
        len=0
        res=1
        for t in s:
            if t!=lst:
                lst=t
                len=0
            if t==lst:
                len+=1
            if len>res:
                res=len
        toplot.append(res)
    toplot=np.asarray(toplot)
    bins=list(range(toplot.min(),toplot.max()+1))
    bins.append(np.inf)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(toplot,cumulative=True,bins=bins,histtype='step',density=True,label='Empirical')
    X = np.arange(1, 31)
    Y = []
    for x in X:
        m=x+1
        res=1.
        for j in range(1,199//m+1):
            mid=np.log(0.5+(200-j*m)/j*0.5)+np.log(0.5)*(j*m+j-1)
            a=199-j*m
            b=j-1
            for i in range(1,a+1):
                mid+=np.log(i)
            for i in range(1,b+1):
                mid-=np.log(i)
            for i in range(1,a-b+1):
                mid-=np.log(i)
            if j%2==0:
                res+=np.exp(mid)
            else:
                res-=np.exp(mid)
        Y.append(res)
    n80=n90=n100=n110=n120=0
    for t in toplot:
        if t<=6:
            n80+=1
        if t<=7:
            n90+=1
        if t<=8:
            n100+=1
        if t<=9:
            n110+=1
        if t<=10:
            n120+=1
    print(n80/SAMPLES,Y[4])
    print(n90 / SAMPLES, Y[5])
    print(n100 / SAMPLES, Y[6])
    print(n110 / SAMPLES, Y[7])
    print(n120 / SAMPLES, Y[8])
    ax.plot(X+1., Y, 'k--', linewidth=1.5, label='Theoretical')

    ax.set_title(str(SAMPLES))
    plt.legend()
    plt.show()

def p1_3(t=5):
    SAMPLES=100000
    samples=np.random.randint(0,2,(SAMPLES,200))
    print(samples)
    toplot5=[]
    toplot6=[]
    toplot7=[]
    for s in samples:
        lst=-1
        len=0
        cnt5=0
        cnt6=0
        cnt7=0
        for t in s:
            if t!=lst:
                lst=t
                len=0
            if t==lst:
                len+=1
            if len==5:
                cnt5+=1
            if len==6:
                cnt6+=1
            if len>=7:
                cnt7+=1
        toplot5.append(cnt5)
        toplot6.append(cnt6)
        toplot7.append(cnt7)
    toplot=np.asarray([toplot5,toplot6,toplot7])
    bins=list(range(toplot.min(),toplot.max()+1))
    bins.append(np.inf)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(toplot5,cumulative=True,bins=bins,histtype='step',density=True,label='5')
    ax.hist(toplot6,cumulative=True,bins=bins,histtype='step',density=True,label='6')
    ax.hist(toplot7,cumulative=True,bins=bins,histtype='step',density=True,label='>=7')

    ax.set_title(str(SAMPLES))
    plt.legend()
    plt.show()

def p2():
    SAMPLES=200000
    samples=np.random.normal(size=(SAMPLES,100))
    samples=np.sort(samples,axis=1)
    s1=samples[:,99]
    s2=samples[:,98]
    s3=samples[:,97]
    datas=[]
    for s in s1:
        datas.append({"sample":s,'label':'X1'})
    for s in s2:
        datas.append({"sample": s, 'label': 'X2'})
    for s in s3:
        datas.append({"sample": s, 'label': 'X3'})
    datas=pd.DataFrame(datas)
    _, bins = np.histogram(datas['sample'], bins=40)
    g = sns.FacetGrid(datas, hue="label", hue_order=['X1', 'X2', 'X3'])
    g = g.map(sns.distplot, "sample", bins=bins)

    plt.title("Empirical "+str(SAMPLES))
    plt.legend(['X1', 'X2','X3'], loc='right', ncol=1,
               frameon=False)
    plt.tight_layout()
    print(np.mean(s1),np.std(s1)*np.sqrt(SAMPLES/(SAMPLES-1)))
    print(np.mean(s2),np.std(s2)*np.sqrt(SAMPLES/(SAMPLES-1)))
    print(np.mean(s3),np.std(s3)*np.sqrt(SAMPLES/(SAMPLES-1)))
    plt.show()

def p5():
    SAMPLES=1000000
    sigma1=80
    sigma2=50
    r=0.4
    cxy=r*sigma1*sigma2
    samples=np.random.multivariate_normal(mean=[0,0],cov=[[sigma1*sigma1,cxy],[cxy,sigma2*sigma2]],size=(SAMPLES))
    cnt=0
    for s in samples:
        if np.linalg.norm(s)<=100:
            cnt+=1
    print(cnt/SAMPLES)

def p6():
    SAMPLES=1000000
    x=np.random.random((SAMPLES,1))*2-1
    y=np.random.random((SAMPLES,1))*2-1
    z=np.random.random((SAMPLES,1)) * 2
    samples=np.concatenate([x,y,z],axis=1)
    cnt=0
    for s in samples:
        norm=np.linalg.norm([s[0],s[1]])
        if norm<=1.:
            z_lower=norm
            z_upper=np.sqrt(1.-s[0]*s[0]-s[1]*s[1])+1.
            if s[2]>z_lower and s[2]<z_upper:
                cnt+=1
    print(cnt/SAMPLES*8)


p6()