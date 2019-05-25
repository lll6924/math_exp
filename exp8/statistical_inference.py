from data import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t,kstest
from scipy.optimize import root
import math

ALPHA=0.05

def p6():
    data=get6()
    data=pd.DataFrame(data)
    T=t.ppf(1-ALPHA/2.,99)
    print(kstest(list(data['height']),"norm"))
    print(kstest(list(data['weight']),"norm"))

    mean1=np.mean(data['height'])
    std1=(np.std(data['height']))
    print(mean1,mean1-T*std1/10,mean1+T*std1/10)
    mean2=(np.mean(data['weight']))
    std2=(np.std(data['weight']))
    print(mean2,mean2-T*std2/10,mean2+T*std2/10)
    sns.distplot(data['height'])
    plt.show()
    sns.distplot(data['weight'])
    plt.show()

N1=200000/2/12
N2=200000/2/12

def p10(alpha=ALPHA):
    print(N1)
    print(N2)
    data=get10()
    data=pd.DataFrame(data)
    sns.lineplot(x='age',y='mean',hue='gender',style='country',data=data)
    chineseboy=data[data['country']=='china'][data['gender']=='boy']
    chinesegirl=data[data['country']=='china'][data['gender']=='girl']
    japaneseboy=data[data['country']=='japan'][data['gender']=='boy']
    japanesegirl=data[data['country']=='japan'][data['gender']=='girl']
    print("Checking boy...")
    for age in range(7,19):
        mean1=float(chineseboy[chineseboy['age']==age]['mean'])
        std1=float(chineseboy[chineseboy['age']==age]['std'])
        mean2=float(japaneseboy[japaneseboy['age']==age]['mean'])
        std2=float(japaneseboy[japaneseboy['age']==age]['std'])
        s2=((N1-1)*std1*std1+(N2-1)*std2*std2)/(N1+N2-2)
        z=(mean1-mean2)/np.sqrt(s2/N1+s2/N2)
        acrate=t.ppf(1-alpha/2,N1+N2-2)
        print(acrate)
        print('age=',age," z=",z,' Accepted' if np.abs(z)<acrate else ' Rejected')
    print("Checking girl...")
    for age in range(7,19):
        mean1=float(chinesegirl[chinesegirl['age']==age]['mean'])
        std1=float(chinesegirl[chinesegirl['age']==age]['std'])
        mean2=float(japanesegirl[japanesegirl['age']==age]['mean'])
        std2=float(japanesegirl[japanesegirl['age']==age]['std'])
        s2=((N1-1)*std1*std1+(N2-1)*std2*std2)/(N1+N2-2)
        z=(mean1-mean2)/np.sqrt(s2/N1+s2/N2)
        acrate=t.ppf(1-alpha/2,N1+N2-2)
        print('age=',age," z=",z,' Accepted' if np.abs(z)<acrate else ' Rejected')
    plt.show()

LAMBDA=0.001

def p3_1(T=500):
    samples=np.random.random((1000,100))
    samples=-np.log(samples)/LAMBDA
    estimations=[]
    for sample in samples:
        cnt=0
        for s in sample:
            if s<T:
                cnt+=1
        estimations.append(-T/np.log(1-cnt/100))
    ac = t.ppf(1 - ALPHA / 2., 999)
    mean=np.mean(estimations)
    std=np.std(estimations)
    print(round(mean,3),'[', round(mean - ac * std / np.sqrt(1000),3),',', round(mean + ac * std / np.sqrt(1000),3),']')
    sns.distplot(estimations)
    #plt.show()

def p3_2(T=500):
    samples=np.random.random((1000,100))
    samples=-np.log(samples)/LAMBDA
    estimations=[]
    class functioner:
        def __init__(self,K):
            self.K=K
        def __call__(self,x):
            inter=np.exp(-x*T)
            return (-T*inter-1/x*inter+1/x)/(1-inter)-self.K
    for sample in samples:
        cnt=0
        acu=0
        for s in sample:
            if s<T:
                cnt+=1
                acu+=s
        acu/=cnt
        fun=functioner(acu)
        got=root(fun,0.001)
        estimations.append(1/got['x'])
    ac = t.ppf(1 - ALPHA / 2., 999)
    mean=np.mean(estimations)
    std=np.std(estimations)
    print(round(mean,3),'[', round(mean - ac * std / np.sqrt(1000),3),',', round(mean + ac * std / np.sqrt(1000),3),']')
    sns.distplot(estimations)
    #plt.show()

def p3_3(T=500):
    samples=np.random.random((1000,100))
    samples=-np.log(samples)/LAMBDA
    estimations=[]
    class functioner:
        def __init__(self,K):
            self.K=K
        def __call__(self,x):
            inter=np.exp(-x*T)
            return -1/x*inter+1/x-self.K
    for sample in samples:
        cnt=0
        acu=0
        for s in sample:
            if s<T:
                cnt+=1
                acu+=s
            else:
                cnt+=1
                acu+=T
        acu/=cnt
        fun=functioner(acu)
        estimations.append(1/root(fun,0.001)['x'])
    ac = t.ppf(1 - ALPHA / 2., 999)
    mean=np.mean(estimations)
    std=np.std(estimations)
    print(round(mean,3),'[', round(mean - ac * std / np.sqrt(1000),3),',', round(mean + ac * std / np.sqrt(1000),3),']')
    sns.distplot(estimations)
    #plt.show()

p3_3(1500)