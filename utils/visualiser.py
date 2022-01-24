import matplotlib.pyplot as plt
from constants import visuals_folder, visualiser, normalised
import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def visualise(w,shape):
    if visualiser == False:
        return
    wsq=w.copy()
    if normalised == False:
        wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    '''if len(shape) == 3:
        wsq = np.average(wsq, axis = -1)'''
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()


def visualise_pos(w,shape):
    if visualiser == False:
        return
    wsq=w.copy()
    wsq=np.maximum(wsq, 0)
    if normalised == False:
        wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    '''if len(shape) == 3:
        wsq = np.average(wsq, axis = -1)'''
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()

def visualise_neg(w,shape):
    if visualiser == False:
        return
    wsq=w.copy()
    wsq=np.minimum(wsq, 0)
    if normalised == False:
        wsq = (wsq - np.min(wsq)) / (np.max(wsq) - np.min(wsq))
    wsq=np.reshape(wsq,shape)
    '''if len(shape) == 3:
        wsq = np.average(wsq, axis = -1)'''
    plt.imshow(wsq, interpolation='nearest',cmap='gray')
    plt.show()
