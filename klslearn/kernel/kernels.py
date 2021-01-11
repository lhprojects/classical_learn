import numpy as np

class RBF:
    '''
    kij = exp(-(d(xi, xj))^2/(2 l^2))
    '''
    def __init__(self, length_scale = 1.0, sigma = 1.):
        self.length_scale = length_scale
        self.sigma = sigma
        
    def __call__(self, X, Y=None):
        '''
        (X.shape[0], X.shape[1])
        '''
        if Y is None:
            Y = X
            
        distance2 = np.square(X[:,np.newaxis,:] - Y[np.newaxis,:,:]).sum(axis=2)
        distance2 /= 2*self.length_scale**2
        K = np.exp(-distance2)
        K *= self.sigma
        return K

