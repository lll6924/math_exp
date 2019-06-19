from calculator import *
import math
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time

def getn_kklll(delta,epsilon):
    for n in range(1,1000):
        if math.pow(math.sqrt(2),n)>=n*n/epsilon/epsilon*math.log(1/delta):
            return n

def getn_gg(delta,epsilon):
    for n in range(1,1000):
        if math.pow(math.sqrt(4/3),n)>=n*n/epsilon/epsilon*math.log(1/delta):
            return n

def generate(n,beta=0.5):
    ret=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if random.random()<beta:
                ret[i][j]=1
    return ret

def getnums_kklll():
    deltas=[0.2,0.1,0.05,0.01]
    epsilons=[0.2,0.1,0.05,0.01]
    with open("kklll.csv",mode='w')as f:
        s=''
        for epsilon in epsilons:
            s+=','+str(epsilon)
        f.write(s+'\n')
        for delta in deltas:
            s=str(delta)
            for epsilon in epsilons:
                s+=','+str(getn_kklll(delta,epsilon))
            f.write(s+'\n')

def getnums_gg():
    deltas=[0.2,0.1,0.05,0.01]
    epsilons=[0.2,0.1,0.05,0.01]
    with open("gg.csv",mode='w')as f:
        s=''
        for epsilon in epsilons:
            s+=','+str(epsilon)
        f.write(s+'\n')
        for delta in deltas:
            s=str(delta)
            for epsilon in epsilons:
                s+=','+str(getn_gg(delta,epsilon))
            f.write(s+'\n')

def estimate_error():
    datas=[]
    for method in ['gg','kklll','normal']:
        for i in range(1,20):
            A = generate(i,0.5)
            actual=ryser_calculator(A)
            while actual==0:
                A = generate(i, 0.5)
                actual = ryser_calculator(A)
            if method=='gg':
                estimated,length=gg_calculator(A)
            elif method=='kklll':
                estimated, length = kklll_calculator(A)
            else:
                estimated, length = normal_calculator(A)
            datas.append({"n":i,"ratio":length/estimated,"data":"length of CI/estimated",'method':method})
            datas.append({"n": i, "ratio": abs(actual-estimated)/actual, "data": "relative error",'method':method})
            print(i,length/actual,abs(actual-estimated)/actual)
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='n',y='ratio',style='data',hue='method',data=datas)
    plt.title("Comparison among three MC methods")
    plt.show()

def estimate_error_beta():
    datas = []
    N=15
    for method in ['gg','kklll','normal']:
        for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            A = generate(N, beta)
            actual = ryser_calculator(A)
            if method == 'gg':
                estimated, length = gg_calculator(A)
            elif method == 'kklll':
                estimated, length = kklll_calculator(A)
            else:
                estimated, length = normal_calculator(A)
            if actual==0:
                error=0
            else:
                error=abs(actual-estimated)/actual
            datas.append({"beta": beta, "error": error, 'method':method})
            print(method,beta,error)
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='error',hue='method',data=datas)
    plt.title("Relative error of different MC methods with different beta(n=15)")
    plt.show()

def estimate_time_gg():
    datas = []
    for N in [5,10,15,20]:
        for error in [0.1,0.3,0.5,0.7,0.9]:
            A=generate(N,error)
            for _ in range(5):
                t1=time.time()
                gg_calculator(A)
                t2=time.time()
                print(N,error,t2-t1)
                datas.append({'beta':error,'N':N,'time':t2-t1})
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='time',hue='N',data=datas)
    plt.title("Time consumption of GG estimator")
    plt.show()

def estimate_time_kklll():
    datas = []
    for N in [5,10,15,20]:
        for error in [0.1,0.3,0.5,0.7,0.9]:
            A=generate(N,error)
            for _ in range(5):
                t1=time.time()
                kklll_calculator(A)
                t2=time.time()
                datas.append({'beta':error,'N':N,'time':t2-t1})
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='time',hue='N',data=datas)
    plt.title("Time consumption of KKLLL estimator")
    plt.show()

def estimate_time_ryser():
    datas = []
    for N in [5,10,15,20]:
        for error in [0.1,0.3,0.5,0.7,0.9]:
            A=generate(N,error)
            for _ in range(5):
                t1=time.time()
                ryser_calculator(A)
                t2=time.time()
                datas.append({'beta':error,'N':N,'time':t2-t1})
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='time',hue='N',data=datas)
    plt.title("Time consumption of Ryser algorithm")
    plt.show()

def estimate_time_normal():
    datas = []
    for N in [5,10,15,20]:
        for error in [0.1,0.3,0.5,0.7,0.9]:
            A=generate(N,error)
            for _ in range(5):
                t1=time.time()
                normal_calculator(A)
                t2=time.time()
                datas.append({'beta':error,'N':N,'time':t2-t1})
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='time',hue='N',data=datas)
    plt.title("Time consumption of Normal estimator")
    plt.show()

def estimate_time_naive():
    datas = []
    for N in [5,8,11]:
        for error in [0.1,0.3,0.5,0.7,0.9]:
            print(int(N))
            A=generate(int(N),error)
            for _ in range(5):
                t1=time.time()
                raw_calculator(A)
                t2=time.time()
                datas.append({'beta':error,'N':N,'time':t2-t1})
    datas=pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='beta',y='time',hue='N',data=datas)
    plt.title("Time consumption of naive algorithm")
    plt.legend([5,8,11])
    plt.show()

def estimate_N():
    datas = []
    beta=0.5
    for method in ['gg', 'kklll', 'normal']:
        for N in [4000,8000,12000,16000,20000]:
            A = generate(15, beta)
            actual = ryser_calculator(A)
            if method == 'gg':
                estimated, length = gg_calculator(A,N)
            elif method == 'kklll':
                estimated, length = kklll_calculator(A,N)
            else:
                estimated, length = normal_calculator(A,N)
            if actual == 0:
                error = 0
                l_esti=0
            else:
                error = abs(actual - estimated) / actual
                l_esti=length/estimated
            datas.append({"N":N,"ratio":l_esti,"data":"length of CI/estimated",'method':method})
            datas.append({"N": N, "ratio": error,"data":'relative error', 'method': method})
    datas = pd.DataFrame(datas)
    plt.clf()
    sns.lineplot(x='N', y='ratio', hue='method',style='data', data=datas)
    plt.title("Comparison of different methods at different N(beta=0.5,n=15)")
    plt.show()

def estimate_time_all():
    datas = []
    for method in ['ryser','gg', 'kklll', 'normal']:
        times=[]
        for _ in range(10):
            A=generate(20,0.5)
            t1=time.time()
            if method=='ryser':
                estimated=ryser_calculator(A)
            elif method == 'gg':
                estimated, length = gg_calculator(A)
            elif method == 'kklll':
                estimated, length = kklll_calculator(A)
            else:
                estimated, length = normal_calculator(A)
            t2=time.time()
            times.append(t2-t1)
        print(method,np.mean(times))

def main():
    #getnums_gg()
    #getnums_kklll()
    estimate_time_all()

if __name__=='__main__':
    main()