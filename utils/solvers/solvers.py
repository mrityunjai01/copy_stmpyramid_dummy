# TODO give option to constrain the weights from min to max OR 0 to max OR -max to max etc
# TODO ask if the max/min of the constraints is AFTER or BEFORE decomposition
# TODO code up primal form of SHTM (is the 'w' on the decomposed matrices of X or on the reconstructed X after the outerproduct summation)
# TODO SHTM is not working (error that expressions with dimensions more than 2 are not supported)
# TODO MCTM is giving mediocre results and doesn't get better with height or C so maybe some bug
from utils.solvers.tensor import make_kernel,rank_R_decomp,construct_W_from_mat
from utils.solvers.vector import inner_prod,construct_W_from_vec,inner_prod_cp
from utils.solvers.centroid import centroid
from sklearn.utils import shuffle
from constants import verbose_sgd,verbose_solver

import cvxpy as cp
import numpy as np
from random import seed

np.random.seed(1)
seed(1)
import warnings

warnings.filterwarnings("ignore")


def SHTM(X, y, C=1.0, rank=3, xa=None, xb=None, constrain='lax', wnorm='L1', wconst='maxmax', margin='soft'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    constrain = M if constrain == 'lax' else 1
    rank = 3 if rank is None else rank

    data_fact = [rank_R_decomp(x, rank) for x in X]
    K = make_kernel(data_fact)

    l = cp.Variable(M)
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    objfun = C * cp.sum(q)
    if wnorm == 'L1':
        objfun += cp.sum(cp.abs(l @ X)) + cp.abs(wa) + cp.abs(wb)
    elif wnorm == 'L2':
        objfun += 1 / 2 * (cp.sum(cp.square(l @ X)) + cp.square(wa) + cp.square(wb))
    constraints = []
    if wconst == 'maxmax':
        constraints.append(l >= -1 / M * constrain)
        constraints.append(l <= 1 / M * constrain)
    elif wconst == 'minmax':
        constraints.append(l >= 0)
        constraints.append(l <= 1 / M * constrain)
    constraints.append(q >= 0)
    for i in range(M):
        constraints.append(
            y[i] * (cp.sum(cp.multiply(l, K[:, i])) + b + cp.multiply(wa, xa[i]) + cp.multiply(wb, xb[i])) + q[i] >= 1)

    problem = cp.Problem(cp.Minimize(objfun), constraints)
    problem.solve()

    W = construct_W_from_mat(data_fact, l.value, 1e-9)

    if verbose_solver:
        tots = q.value
        tots[tots < 1e-9] = 0
        print(f"SHTM done, q = {np.sum(np.sign(tots))}")

    return W, b.value, wa.value, wb.value


def STM(X, y, C=1.0, rank=3, xa=None, xb=None, constrain='lax', wnorm='L1', wconst='maxmax', margin='soft'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    wshape = X.shape[1:]
    w = cp.Variable(len(X[0].reshape(-1)))
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    if (margin == 'soft'):
        objfun = C * cp.sum(q)
    else:
        objfun = 0
    if wnorm == 'L1':
        objfun += cp.sum(cp.abs(w)) + cp.abs(wa) + cp.abs(wb)
    elif wnorm == 'L2':
        objfun += 1 / 2 * (cp.sum(cp.square(w)) + cp.square(wa) + cp.square(wb))
    constraints = []
    maxes = np.max(X, axis=0).reshape(-1)
    mines = np.min(X, axis=0).reshape(-1)
    if (margin == 'soft'):
        if wconst == 'maxmax':
            abmaxes = np.maximum(np.abs(maxes), np.abs(mines))
            constraints.append(w <= abmaxes)
            constraints.append(w >= -abmaxes)
        elif wconst == 'minmax':
            constraints.append(w <= maxes)
            constraints.append(w >= mines)
        constraints.append(q >= 0)
    for i in range(M):
        if (margin == 'soft'):
            constraints.append((y[i]*(inner_prod_cp(w,X[i].reshape(-1))+b+cp.multiply(wa,xa[i])+cp.multiply(wb,xb[i]))+q[i]) >= 1.0)
        else:
            constraints.append((y[i]*(inner_prod_cp(w,X[i].reshape(-1))+b+cp.multiply(wa,xa[i])+cp.multiply(wb,xb[i]))) >= 1.0)
    constraints.append(wa >= 0)
    constraints.append(wb <= 0)
    problem = cp.Problem(cp.Minimize(objfun), constraints)
    problem.solve()

    W = construct_W_from_vec(w.value, wshape)
    if((verbose_solver==True)&(margin=='soft')):
        tots = q.value
        tots[tots < 1e-9] = 0
        print(f"STM done, q = {np.sum(np.sign(tots))}")

    return W, b.value, wa.value, wb.value


def MCM(X, y, C=1.0, rank=3, xa=None, xb=None, constrain='lax', wnorm='L1', wconst='maxmax', margin='soft'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)

    wshape = X.shape[1:]

    w = cp.Variable(len(X[0].reshape(-1)))
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    h = cp.Variable()
    objfun = h + C * cp.sum(q)
    constraints = []
    maxes = np.max(X, axis=0).reshape(-1)
    mines = np.min(X, axis=0).reshape(-1)
    if wconst == 'maxmax':
        abmaxes = np.maximum(np.abs(maxes), np.abs(mines))
        constraints.append(w <= abmaxes)
        constraints.append(w >= -abmaxes)
    elif wconst == 'minmax':
        constraints.append(w <= maxes)
        constraints.append(w >= mines)
    constraints.append(q >= 0)
    for i in range(M):
        constraints.append(
            y[i] * (inner_prod_cp(w, X[i].reshape(-1)) + b + cp.multiply(wa, xa[i]) + cp.multiply(wb, xb[i])) + q[
                i] >= 1.0)
        constraints.append(
            y[i] * (inner_prod_cp(w, X[i].reshape(-1)) + b + cp.multiply(wa, xa[i]) + cp.multiply(wb, xb[i])) + q[
                i] <= h)

    problem = cp.Problem(cp.Minimize(objfun), constraints)
    problem.solve()

    W = construct_W_from_vec(w.value, wshape)

    if verbose_solver:
        tots = q.value.copy()
        tots[tots < 1e-9] = 0
        ## The below shows accuracy can be 100% even with value of q!=0 (just needs to be q<1)
        # tots2 = q.value.copy()
        # tots2[tots2<1] = 0
        # print(f"MCM done, q = {np.sum(np.sign(tots))}, q2 = {np.sum(np.sign(tots2))}")
        print(f"MCM done, q = {np.sum(np.sign(tots))}")

    return W, b.value, wa.value, wb.value


def MCTM(X, y, C=1.0, rank=3, xa=None, xb=None, constrain='lax', wnorm='L1', wconst='maxmax', margin='soft'):
    '''
    If solver doesn't work, then hyperparameters chosen are faulty.
    '''
    M = len(X)

    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    rank = 3 if rank is None else rank
    constrain = M if constrain == 'lax' else 1
    data_fact = [rank_R_decomp(x, rank) for x in X]
    K = make_kernel(data_fact)

    h = cp.Variable()
    b = cp.Variable()
    q = cp.Variable(M)
    l = cp.Variable(M)
    wa = cp.Variable()
    wb = cp.Variable()

    obj = h + C * cp.sum(q)

    constraints = []
    for i in range(M):
        constraints.append(
            h >= y[i] * (cp.sum(cp.multiply(l, K[:, i])) + b + cp.multiply(wa, xa[i]) + cp.multiply(wb, xb[i])) + q[i])
        constraints.append(
            y[i] * (cp.sum(cp.multiply(l, K[:, i])) + b + cp.multiply(wa, xa[i]) + cp.multiply(wb, xb[i])) + q[i] >= 1)
    constraints.append(q >= 0)
    if wconst == 'maxmax':
        constraints.append(l >= -1 / M * constrain)
        constraints.append(l <= 1 / M * constrain)
    elif wconst == 'minmax':
        constraints.append(l >= 0)
        constraints.append(l <= 1 / M * constrain)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    W = construct_W_from_mat(data_fact, l.value, 1e-9)

    if verbose_solver:
        tots = q.value
        tots[tots < 1e-9] = 0
        print(f"MCTM done, q = {np.sum(np.sign(tots))}")

    return W, b.value, wa.value, wb.value


def cost_function(X, y, C, W, b, xa, xb, wa, wb, wnorm='L1'):
    M = len(X)
    cost = np.sum(np.max(1 - y * (inner_prod(W, X) + b + wa * xa + wb * xb), 0), axis=-1) * C / M
    if wnorm == 'L1':
        cost += (np.sum(np.abs(W)) + np.abs(b) + np.abs(wa) + np.abs(wb))
    elif wnorm == 'L2':
        cost += 1 / 2 * (np.sum(np.square(W)) + np.square(b) + np.square(wa) + np.square(wb))
    return cost


def derivative_cost_function(X, y, C, W, b, xa, xb, wa, wb, wnorm='L1'):
    M = len(X)
    tempy = y.copy()
    dist = 1 - y * (inner_prod(W, X) + b + wa * xa + wb * xb)
    tempy[dist < 0] = 0
    if wnorm == 'L2':
        # cost_derivative_w = np.sum(W - C*np.dot(tempy,X),axis=-1)/M
        cost_derivative_w = np.sum(W - C * tempy * X, axis=-1) / M
        cost_derivative_b = np.sum(b - C * tempy) / M
        cost_derivative_wa = np.sum(wa - C * (xa * tempy)) / M
        cost_derivative_wb = np.sum(wb - C * (xb * tempy)) / M
    elif wnorm == 'L1':
        # cost_derivative_w = np.sum(np.sign(W) - C*np.dot(tempy,X),axis=-1)/M
        cost_derivative_w = np.sum(np.sign(W) - C * tempy * X, axis=-1) / M
        cost_derivative_b = np.sum(np.sign(b) - C * tempy) / M
        cost_derivative_wa = np.sum(np.sign(wa) - C * (xa * tempy)) / M
        cost_derivative_wb = np.sum(np.sign(wb) - C * (xb * tempy)) / M
    return cost_derivative_w, cost_derivative_b, cost_derivative_wa, cost_derivative_wb


def SGD_STM(X, y, C=1.0, rank=3, xa=None, xb=None, constrain='lax', wnorm='L1', max_epoch=2, lr=0.2, wconst='maxmax',
            margin='soft'):
    if verbose_solver:
        print('Reached SGD_STM')
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)

    wshape = X.shape[1:]
    W = np.random.randn(*wshape)
    b = 0
    wa = np.random.randn()
    wb = np.random.randn()
    for epoch in range(max_epoch):
        X, y = shuffle(X, y)
        for i, x in enumerate(X):
            wgrad, bgrad, wagrad, wbgrad = derivative_cost_function(x, y[i], C, W, b, xa[i], xb[i], wa, wb, wnorm)
            W -= wgrad * lr
            b -= bgrad * lr
            wa -= wagrad * lr
            wb -= wbgrad * lr
            maxes = np.max(X, axis=0)
            mines = np.min(X, axis=0)
            if wconst == 'maxmax':
                abmaxes = np.maximum(np.abs(maxes), np.abs(mines))
                W = np.clip(W, -abmaxes, abmaxes)
            elif wconst == 'minmax':
                W = np.clip(W, -mines, maxes)
        if verbose_sgd == True:
            print(f"Epoch number {epoch + 1} is done")
    return W, b, wa, wb


def getHyperPlaneFromTwoPoints(xp, xn):
    if verbose_solver:
        print('Reached centroid')
    x1 = centroid(xp)
    x2 = centroid(xn)
    w = (2) * (x2 - x1) / (np.linalg.norm(x1 - x2) ** 2)
    b = -1 * inner_prod(w, (0.5 * (x1 + x2)))
    return -w, -b
