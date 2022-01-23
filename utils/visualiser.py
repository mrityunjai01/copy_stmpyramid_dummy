import matplotlib.pyplot as plt
from constants import visuals_folder, visualiser
import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def visualise(w,shape):
    wsq=w.copy()
    wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()


def visualise_pos(w,shape):
    
    wsq=w.copy()
    wsq=np.maximum(wsq, 0)
    wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()

def visualise_neg(w,shape):
    
    wsq=w.copy()
    wsq=np.minimum(wsq, 0)
    wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()