from utils.model import Node
import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def crossvalidator(Xtrain, ytrain, solver, k = 5, onlyonce = False, h = 3, C = 1.0, rank = 3, xa = None, xb = None, constrain = 'lax', wnorm = 'L1'):
    n = Xtrain.shape[0]
    i = 0
    r = int(n/k)
    k = 1 if onlyonce is True else k
    total_accuracy = 0
    while(i < k):
        print(f"{i} fold")
        if ((i+1)*r>n):
            print("lower the test %")
            break
        Xtra1 = Xtrain[0 : r*i]
        ytra1 = ytrain[0 : r*i]
        Xtra2 = Xtrain[r*(i+1) : ]
        ytra2 = ytrain[r*(i+1) : ]
        Xtra = np.concatenate((Xtra1, Xtra2), axis=0)
        ytra = np.concatenate((ytra1, ytra2), axis=0)
        Xtest = Xtrain[r*i : r*(i+1)]
        ytest = ytrain[r*i : r*(i+1)]
        model = Node(Xtra.shape[1:], solver, C, rank, xa, xb, constrain, wnorm)
        model.recursive(Xtra, ytra, h)
        model.fine_tune_weights()
        print('Train Accuracy:')
        print(model.accuracy(Xtra, ytra))
        acc = model.accuracy(Xtest, ytest)
        total_accuracy += acc
        print('Test Accuracy:')
        print(acc)
        i += 1
        print('\n')
    print(f"Total test accuracy={total_accuracy/k}\n\n")