import numpy as np
import matplotlib.pyplot as plt

def generate_scatter_labeled(center = [[0,0]], n = 200, random_state = 0):

    '''
    center: [[floating]]
        center of each class
        dimension infered from center
    n: int or [int]
        number of examples of each class
    '''

    rnd = np.random.RandomState(random_state)

    if isinstance(n, int):
        Xs = [rnd.randn(n, 2) + c for c in center]
        ys = [np.full(n, i) for i in range(len(center))]
    else:
        assert len(n) == len(center), n
        Xs = [rnd.randn(n_, 2) + c for c, n_ in zip(center, n)]
        ys = [np.full(n_, i) for i, n_ in enumerate(n)]

    X = np.vstack(Xs)
    y = np.hstack(ys)
    pert = np.random.permutation(X.shape[0])
    X = X[pert]
    y = y[pert]

    return X, y


def plot_scatter_2D_labeled(X, y, ax=None):
    ax_ = ax
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        
    y_unique, unique_inverse = np.unique(y, return_inverse=True)

    n_classes = len(y_unique)
    for i in range(n_classes):
        ax.scatter(X[unique_inverse == i, 0], X[unique_inverse == i, 1], label = "%s"%y_unique[i])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    
    if ax_ is None:
        plt.show()