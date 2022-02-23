import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from random import seed
np.random.seed(1)
seed(1)

def rank_R_decomp(X, rank = 3):
    X_t = tl.tensor(X)
    _, factors = parafac(X_t, int(rank))
    fact_np = [tl.to_numpy(f) for f in factors]
    return fact_np


def inner_prod_decomp(Ai, Aj):
    s = 0.0
    R = Ai[0].shape[1]
    for p in range(R):
        for q in range(R):
            prod = 1.0
            for ai, aj in zip(Ai, Aj):  
                prod *= np.dot(ai[:, p], aj[:, q])
            s += prod
    return s

def make_kernel(data_decomp):
    K = np.zeros((len(data_decomp), len(data_decomp)))
    for i in range(len(data_decomp)):
        for j in range(i+1):
            K[i, j] = inner_prod_decomp(data_decomp[i], data_decomp[j])
            K[j, i] = K[i, j]
    return K

def construct_W_from_mat(data_decomp, l, eps=1e-100):
    R = data_decomp[0][1].shape[1]
    W = tl.zeros([data_decomp[0][i].shape[0] for i in range(len(data_decomp[0]))])
    for i, flag in enumerate((np.abs(l) > eps)):
        if flag:
            W += l[i]*tl.cp_to_tensor((np.ones(R), data_decomp[i]))
    return tl.to_numpy(W)


def data_decomp(x, rank = 3):
    xnew = np.array([])
    for X in x:
        X_t = tl.tensor(X)
        _, factors = parafac(X_t, int(rank))
        facts = []
        facts.extend(tl.to_numpy(f) for f in factors)
        Xnew = np.vstack(facts)
        Xnew = Xnew.reshape((1,*Xnew.shape))
        if len(xnew) == 0:
            xnew = Xnew
        else:
            xnew = np.vstack([xnew,Xnew])
    return xnew , X.shape

def construct_W_from_decomp(W_decomp, shape):
    R = W_decomp.shape[1]
    shape = [0] + list(shape)
    facts = [W_decomp[shape[i-1]:shape[i-1]+shape[i]] for i in range(1,len(shape))]
    W = tl.cp_to_tensor((np.ones(R),facts))
    return W
