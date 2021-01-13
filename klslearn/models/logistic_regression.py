import numpy as np
from ..kernels import RBF

def sigmoid(z):
    # [n_examples, 1] -> [n_exmaples, 1]
    max_ = np.maximum(0, z)
    z_ = z - max_
    zero_ = 0 - max_
    expz = np.exp(z_)
    return expz / (np.exp(zero_) +expz)


def softmax(z):
    # [n_examples, n_labels] -> [n_exmaples, n_labels]
    maxz = np.amax(z, axis=1, keepdims=True)
    z = z - maxz
    expz = np.exp(z)
    norm = np.sum(expz, axis=1, keepdims=True)
    return expz / norm

# gradient descent
def optimize_wb(use_sigmoid, use_kernel, w, b, X, y_onehot, learning_rate, lambda_, tol, max_iter):
    # X[n_examples, n_features]
    # b[n_labels]
    # w[n_labels, n_features]
    # y_onehot[n_examples]
    n_examples = X.shape[0]
    n_labels = y_onehot.shape[1]

    for _ in range(max_iter):
        # calculate sigmoid
        # z[n_examples, n_labels]
        # h[n_examples, n_labels]
        z = np.dot(X, w.T) + b
        h = sigmoid(z) if use_sigmoid else softmax(z)

        # calculate gradients
        # r[n_examples, n_labels]
        # grad[n_labels, n_features]
        r = h - y_onehot
        grad = r.T.dot(X) / n_examples
        grad += lambda_ * (np.dot(w, X.T) if use_kernel else w)
        grad_b = r.mean(axis=0)

        # update for weights and bias
        w += -learning_rate * grad
        b += -learning_rate * grad_b

        err = np.sqrt(np.tensordot(grad, grad, axes=((0, 1), (0, 1))) + grad_b.dot(grad_b))\
                / (n_labels + 1)
        if err < tol:
            break
            
    
class LogisticRegressor:

    def __init__(self, C=10, tol=1E-5, max_iter = 200, learning_rate=0.1,random_state=910715,\
                 sigmoid=None, kernel=None, gamma=1.):
        '''
        sigmoid: bool
            True:  sigmoid for binary labels
                  failed for multi(>2) labels
            False: softmax for binary or multi(>2) labels
            None: determine from data
            
        C: floating
            l2 regularization = 1/C
            
        learning_rate: floating
            learning_rate for gradient descent
            
        random_state: integer
            seed for internal random generator
            （not used indeed）
            
        tol: floating
            stop criterior for gradient
            
        max_iter: integer
            max iteration
        '''


        self.C = C
        self.tol = tol
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.sigmoid = sigmoid
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y):
        
        learning_rate = self.learning_rate
        lambda_ = 1 / self.C
        tol = self.tol
        max_iter_ = self.max_iter_

        if self.kernel is None:
            pass
        elif type(self.kernel) is str:
            if self.kernel == "rbf":
                self.kernel_ = RBF(np.sqrt( 1/(2*self.gamma) ))
            else:
                raise ValueError("Unkonw kernel %s"%self.kernel)
        else:
            self.kernel_ = self.kernel
            
        use_kernel = self.kernel is not None
        
        if use_kernel:
            # For a particular label:
            # loss(y) = loss(phi(X) \cdot phi(X) + b)  = loss( K \cdot alpha + b)
            # alpha is like weights
            # K is like features
            # Note the L2 regualization
            #       |phi|^2 =  alpha \cdot  K \cdot alpha
            #  So the contribution to gradient is K \cdot alpha, instead of alpha 
            # We need ot convert the concept shown here to code
            X = self.kernel_(X, X)

        n_examples = X.shape[0]
        n_features = X.shape[1]
        n_labels = len(self.labels)

        if self.sigmoid or (len(self.labels) == 2 and self.sigmoid is None):
            self.sigmoid_ = True
        else:
            self.sigmoid_ = False

        if self.sigmoid_:
            self.w = np.zeros((1, n_features))
            self.b = np.zeros((1, ))
        else:
            self.w = np.zeros((n_labels, n_features))
            self.b = np.zeros((n_labels, ))
        b = self.b
        w = self.w


        if self.sigmoid_:
            # fake an onehot representation
            y_onehot = y.reshape(-1,1)
            use_sigmoid = True
        else:
            # construct onehot representation
            y_onehot = np.zeros((n_examples, n_labels))
            y_onehot[np.arange(n_examples), y] = 1
            use_sigmoid = False




        optimize_wb(use_sigmoid=use_sigmoid, use_kernel=use_kernel,
                    w = w, b = b, X = X, y_onehot = y_onehot,\
            learning_rate = learning_rate, lambda_ = lambda_, tol = tol, max_iter = max_iter_)
            

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert X.ndim == 2, X.shape
        assert y.ndim == 1, y.shape
        assert X.shape[0] == y.shape[0], "X.shape[0]:%s, y.shape[0]:%s"%(X.shape[0], y.shape[0])

        self.max_iter_ = int(self.max_iter)
        
        # if
        #   y = ["a", "a", "b"]
        # then
        #   labels = ["a", "b"]
        #   unique_inverse = [1,1,2]
        labels, unique_inverse = np.unique(y, return_inverse=True)        
        y = unique_inverse
        self.labels = labels

        self.train(X, y)

        self.intercept_ = self.b
        self.coef_ = self.w

    def predict(self, X):
        assert X.ndim == 2, X.shape
        w = self.w
        b = self.b

        if self.kernel is not None and self.kernel_ is not None:
            X = self.kernel_(X)

        z = np.dot(X, w.T) + b
        if self.sigmoid_:
            h = sigmoid(z)
            yhat_label_indices = h > 0.5
            # boolean array cause problem, cast it to int
            yhat_label_indices = np.array(yhat_label_indices.reshape(-1), dtype=np.int_)
        else:
            h = softmax(z)
            yhat_label_indices = np.argmax(h, axis=1)
        yhat = self.labels[yhat_label_indices]

        return yhat

        