import numpy as np
from sklearn.preprocessing import StandardScaler
import mnist
from random import seed
np.random.seed(1)
seed(1)
def bin_mnist(class1 = 3, class2 = 8, totalsamp = None):
    XALL = mnist.train_images()[:totalsamp]/255
    yALL = mnist.train_labels().reshape(-1,1)[:totalsamp].astype(np.int8)
    y0 = (yALL == class1)
    y2 = (yALL == class2)
    ytot = (y0 + y2).reshape(-1)
    Xtrain = XALL[ytot]
    ytrain = np.where(yALL[ytot] == class1, -1, 1)
    return Xtrain, ytrain

def normer(Xtrain):
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain.reshape(-1, Xtrain.shape[-1])).reshape(Xtrain.shape)
    return Xtrain, scaler