import numpy as np
from ..kernels import RBF

        
class Ridge:
    
    def __init__(self, alpha = 0.01, kernel = None, gamma = 1.):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
    
    def fit(self,X,y):
        X = np.array(X, copy=False)

        # kernel pass 1
        # create self.kernel_
        if self.kernel is None:
            pass
        elif type(self.kernel) is str:
            if self.kernel == "rbf":
                self.kernel_ = RBF( length_scale=np.sqrt(1./(2*self.gamma)) )
            else:
                raise ValueError("uknown kernel %s"%self.kernel)
        else:
            self.kernel_ = self.kernel_
        
        # create new X
        if self.kernel is not None:
            self.X_train = X
            X = self.kernel_(X, X)

        # alias
        lamb = float(self.alpha)

        Xmeans = X.mean(axis = 0)
        ymeans = y.mean(axis = 0)
        Xp = X - Xmeans
        yp = y - ymeans

        if self.kernel is not None:
            w  = self.fitKernel(Xp, yp, lamb)
        else:
            w  = self.fitSVD(Xp, yp, lamb)
            
        b =  ymeans - w.dot(Xmeans)
        self.coef_ = w
        self.intercept_ = b

    def fitKernel(self, Xp, yp, lamb):
        A = Xp + np.identity(Xp.shape[0])*lamb
        w,_,_,_ = np.linalg.lstsq(A, yp, rcond=None)
        return w

    def fitSVD(self, Xp, yp, lamb):

        # U[N,r]
        # D[r,r]
        # VT[r,M]
        U,D,VT = np.linalg.svd(Xp, full_matrices=False)
        invD = D/(np.square(D) + lamb) 
        w = VT.T.dot((invD*U.T.dot(yp)))
        return w
                        
    def predict(self, X):
        X = np.array(X, copy=False)        
        assert X.ndim == 2

        if self.kernel is not None:
            X = self.kernel_(X, self.X_train)

        return np.dot(X, self.coef_) + self.intercept_

class LinearRegressor(Ridge):
    def __init__(self, *args, **kargs):
        super().__init__(1E-10, *args, **kargs)

class Lasso:
    
    def __init__(self,alpha = 0.01):
        self.alpha_ = float(alpha)
        pass
    
    def fit(self,X,y, max_iters=100):
        assert X.ndim == 2
        
        X = np.array(X)
        Xmeans = X.mean(axis = 0)
        ymeans = y.mean(axis = 0)
        X = X - Xmeans
        y = y - ymeans
        
        nfeatures = X.shape[1]
        
        W = np.zeros(nfeatures)
        

        B = X.T.dot(X)
        C = y.dot(X)
                
        lamb = 2*X.shape[0]*self.alpha_
        for it in range(max_iters):
            for k in range(nfeatures):
                
                Ak = B[k,k]
                
                Fk = B[:,k].dot(W) - Ak*W[k] - C[k]
                                
                if Fk < -lamb/2: # weight > 0
                    weight = -(Fk + lamb/2)/Ak
                elif Fk > lamb/2: # weight < 0
                    weight = -(Fk - lamb/2)/Ak
                else:
                    weight = 0.0

                W[k] = weight
                
            
        self.coef_ = W
        self.intercept_ = ymeans - W.dot(Xmeans)
                            
    def predict(self, X):
        assert X.ndim == 2            
        return np.dot(X, self.coef_) + self.intercept_

