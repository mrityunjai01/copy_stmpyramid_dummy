import numpy as np
from random import seed
np.random.seed(1)
seed(1)
def centroid(points):
    c = np.zeros(points.shape[1:])
    for i in range(points.shape[0]):
        c = c + points[i]
    return c/points.shape[0]