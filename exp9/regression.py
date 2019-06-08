import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.linear_model import LinearRegression
from scipy.stats import t
from data import *
import seaborn as sns
import matplotlib.pyplot as plt

ALPHA=0.05

def p5():
    data=pd.DataFrame(get5())
    regressor=LinearRegression()
    X=[list(data['x1']),list(data['x2']),list(data['x3']),]
    X=np.asarray(X)
    X=np.resize(X,X.shape+(1,))
    X=np.hstack(X)
    Y=list(data['y'])
    regressor.fit(X,Y)
    #regressor.
    print(regressor.coef_)
    print(regressor.intercept_)
    beta=np.asarray(regressor.coef_)
    inter=regressor.intercept_
    y_predict=np.sum(X*beta,axis=1)+inter
    Q=np.sum(np.square(y_predict-Y))
    s=np.sqrt(Q/(len(X)-4))
    print(s)
    X_center=np.mean(X,axis=0,keepdims=True)
    X_centrialize=X-X_center
    matrix=np.linalg.inv(np.matmul(np.transpose(X_centrialize),X_centrialize))
    #print(matrix)
    #print(s)
    T=t.ppf(1-0.5*ALPHA,len(X)-2)
    for i in range(3):
        delta=T*s*np.sqrt(matrix[i][i])
        print(beta[i]-delta,beta[i]+delta)

def p5_1():
    data=pd.DataFrame(get5())
    data=data.drop(index=[7,19])
    regressor=LinearRegression()
    X=[list(data['x1']),list(data['x2'])]
    X=np.asarray(X)
    X=np.resize(X,X.shape+(1,))
    X=np.hstack(X)
    Y=list(data['y'])
    regressor.fit(X,Y)
    #regressor.
    print(regressor.coef_)
    print(regressor.intercept_)
    beta=np.asarray(regressor.coef_)
    inter=regressor.intercept_
    y_predict=np.sum(X*beta,axis=1)+inter
    Q=np.sum(np.square(y_predict-Y))
    s=np.sqrt(Q/(len(X)-3))
    print(s)
    X_center_=np.mean(X,axis=0)
    X_center=np.mean(X,axis=0,keepdims=True)
    X_centrialize=X-X_center
    matrix=np.linalg.inv(np.matmul(np.transpose(X_centrialize),X_centrialize))
    #print(matrix)
    #print(s)
    T=t.ppf(1-0.5*ALPHA,len(X)-2)
    for i in range(2):
        delta=T*s*np.sqrt(matrix[i][i])
        print(beta[i]-delta,beta[i]+delta)
    d0=np.sqrt(1./len(X)+np.matmul(np.matmul(np.transpose(X_center_),matrix),X_center_))*s*T
    print(inter-d0,inter+d0)
    error=y_predict-Y
    plt.plot(range(len(X)),error,'o')
    for i in range(len(X)):
        plt.vlines(i,error[i]-s*2,error[i]+s*2)
    plt.hlines(0,0,len(X),linestyles='dotted')
    plt.show()

def p10():
    data = pd.DataFrame(get10())
    #data = data.drop(index=[7, 19])
    regressor = LinearRegression()
    x1=np.asarray(list(data['x1']))
    x2=np.asarray(list(data['x2']))
    X = [x1, x1*x1,x2]
    num=len(X)
    X = np.asarray(X)
    X = np.resize(X, X.shape + (1,))
    X = np.hstack(X)
    Y = list(data['y'])
    regressor.fit(X, Y)
    # regressor.
    print(regressor.coef_)
    print(regressor.intercept_)
    beta = np.asarray(regressor.coef_)
    inter = regressor.intercept_
    y_predict = np.sum(X * beta, axis=1) + inter
    Q = np.sum(np.square(y_predict - Y))
    s = np.sqrt(Q / (len(X) - num-1))
    print(s)
    X_center_ = np.mean(X, axis=0)

    X_center = np.mean(X, axis=0, keepdims=True)
    X_centrialize = X - X_center
    matrix = np.linalg.inv(np.matmul(np.transpose(X_centrialize), X_centrialize))
    # print(matrix)
    # print(s)
    T = t.ppf(1 - 0.5 * ALPHA, len(X) - 2)
    for i in range(num):
        delta = T * s * np.sqrt(matrix[i][i])
        print(beta[i] - delta, beta[i] + delta)
    d0=np.sqrt(1./len(X)+np.matmul(np.matmul(np.transpose(X_center_),matrix),X_center_))*s*T
    print(inter-d0,inter+d0)
    error = y_predict - Y
    plt.plot(X[:,0], error, 'o')
    for i in range(len(X)):
        plt.vlines(X[i][0], error[i] - s * 2, error[i] + s * 2)
    plt.hlines(0, 0, plt.xlim()[1], linestyles='dotted')
    plt.show()

def p11():
    data = pd.DataFrame(get11())
    # data = data.drop(index=[7, 19])
    regressor = LinearRegression()
    x1 = np.asarray(list(data['x1']))
    x2 = np.asarray(list(data['x2']))
    x3 = np.asarray(list(data['x3']))
    X = [x1,x1*x1,x3,x1*x2,x1*x1*x3]
    num=len(X)
    X = np.asarray(X)
    X = np.resize(X, X.shape + (1,))
    X = np.hstack(X)
    Y = list(data['y'])
    regressor.fit(X, Y)
    # regressor.
    print(regressor.coef_)
    print(regressor.intercept_)
    beta = np.asarray(regressor.coef_)
    inter = regressor.intercept_
    y_predict = np.sum(X * beta, axis=1) + inter
    Q = np.sum(np.square(y_predict - Y))
    s = np.sqrt(Q / (len(X) - num-1))
    print(s)
    X_center_ = np.mean(X, axis=0)

    X_center = np.mean(X, axis=0, keepdims=True)
    X_centrialize = X - X_center
    matrix = np.linalg.inv(np.matmul(np.transpose(X_centrialize), X_centrialize))
    # print(matrix)
    # print(s)
    T = t.ppf(1 - 0.5 * ALPHA, len(X) - 2)
    for i in range(num):
        delta = T * s * np.sqrt(matrix[i][i])
        print(beta[i] - delta, beta[i] + delta)
    d0 = np.sqrt(1. / len(X) + np.matmul(np.matmul(np.transpose(X_center_), matrix), X_center_)) * s * T
    print(inter - d0, inter + d0)
    error = y_predict - Y
    #plt.plot(x1+x3*40,error, 'o')
    #for i in range(len(X)):
      #  plt.vlines(i, error[i] - s * 2, error[i] + s * 2)
    #plt.hlines(0, 0, plt.xlim()[1], linestyles='dotted')
    #plt.show()

p11()