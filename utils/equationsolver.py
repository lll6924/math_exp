import scipy.sparse.linalg
import numpy as np

__all__ = ["equationSolver", "directSolver", "jacobiSolver","gaussSedeilSolver", "pcgSolver"]

class equationSolver:
    def __call__(self, x, y, n=None, x0=None):
        return self.solve(x, y, x0)
    def solve(self, x, y, x0):
        raise NotImplementedError()

class directSolver(equationSolver):
    def solve(self, x, y, x0=None):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.matmul(np.linalg.inv(x),y)

class jacobiSolver(equationSolver):
    def solve(self, x, y, x0):
        X = x0
        D = np.diag(np.diag(x))
        U = -np.triu(x)+D
        L = -np.tril(x)+D
        _D = np.linalg.inv(D)
        B = np.matmul(_D,(L+U))
        eig,_ =np.linalg.eig(B)
        if(np.max(eig)>1.0 or np.min(eig)<-1.0):
            print("failed, maxeig=",np.max(eig),", mineig=",np.min(eig))
            return x0
        F = np.matmul(_D,y)
        for i in range(10000000):
            LX=X.copy()
            X = np.matmul(B,X) + F
            if(np.linalg.norm(LX-X,np.inf)<1e-5):
                print("times:", i)
                break
        return X

class gaussSedeilSolver(equationSolver):
    def solve(self, x, y, x0, observe=-1):
        A=x
        X = x0
        D = np.diag(np.diag(x))
        U = -np.triu(x)+D
        L = -np.tril(x)+D
        _D_L = np.linalg.inv(D-L)
        B = np.matmul(_D_L,U)
        eig, _ = np.linalg.eig(B)
        print("maxeig=",np.max(eig),", mineig=",np.min(eig))
        if(np.max(eig)>1.0 or np.min(eig)<-1.0):
            print("failed")
            return x0
        F = np.matmul(_D_L,y)
        for i in range(10000000):
            if(i==observe):
                for i,x in zip(range(len(X)),X):
                    print(i,x)
                print(np.linalg.norm(np.matmul(A,X)-y,ord=1))
            LX = X.copy()
            X = np.matmul(B,X) + F
            if (np.linalg.norm(LX - X, np.inf) < 1e-5):
                print("times:", i)
                break
        return X

class pcgSolver(equationSolver):
    def solve(self, x, y, x0):
        X, info = scipy.sparse.linalg.cg(A=x, b=y, x0=x0)
        return X


#solver = jacobiSolver()
#solver(np.arange(9).reshape((3,3)),None)

