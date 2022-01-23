# TODO make a way to save output data rather than print it out (especially for images)
dataset = 'mnist'                   # Current options are mnist, custom

# For mnist
class1 = 3                          # {0,1,2,3,4,5,6,7,8,9}
class2 = 8                          # {0,1,2,3,4,5,6,7,8,9}
totalsamp = 2000                    # int within [50,60000] or None (for max)

# For custom
Xtrain_file = '/path/to/data.npy'   # (Xtrain.shape[0] = total samples and Xtrain is numpy array, no other constraint)
ytrain_file = '/path/to/label.npy'  # (ytrain.shape = (Total Samples, 1) and ytrain is numpy array, no other constraint)

normalised = True                   # normalised data? {True, False}
solver = 'MCTM'                     # Current options are STM, SHTM, MCM, MCTM
k = 5                               # positive int
onlyonce = True                     # {True, False}
h = 2                               # non negative int
C = 100                             # positive float
rank = 3                            # positive int
constrain = 'lax'                   # lax constrain on W or not {'lax', #ANYTHING} using MCM / MCTM
wnorm = 'L1'                        # {'L1', 'L2'} for L1 or L2 norm using STM / SHTM

visualiser = True
visuals_folder = '/path/to/folder'  # folder path to save files to