# TODO need to pass a model here only
from utils.accuracy import accuracy
import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def crossvalidator(Xtrain, ytrain, train, k = 5, onlyonce = False):
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
        Ytra1 = ytrain[0 : r*i]
        Xtra2 = Xtrain[r*(i+1) : ]
        Ytra2 = ytrain[r*(i+1) : ]
        Xtra = np.concatenate((Xtra1, Xtra2), axis=0)
        Ytra = np.concatenate((Ytra1, Ytra2), axis=0)
        Xtest = Xtrain[r*i : r*(i+1)]
        Ytest = ytrain[r*i : r*(i+1)]
        #####################
        model = train(Xtra,Ytra,1,100,'L1','L1')
        ypred = model.forward(Xtest)
        acc = accuracy(ypred,Ytest)
        ####################
        total_accuracy += acc
        print('Test Accuracy:')
        print(acc)
        i += 1
        print('\n')
    print(f"Total test accuracy={total_accuracy/k}\n\n")