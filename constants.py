# TODO make a way to save output data rather than print it out (especially for images)
dataset = 'cifar10'                  # {mnist, cifar10, custom}

# For mnist 
if dataset == 'mnist':
    class1 = 3                       # {0,1,2,3,4,5,6,7,8,9}
    class2 = 8                       # {0,1,2,3,4,5,6,7,8,9}
    totalsamp = 2000                 # int within [50,60000] or None (for max)

# For cifar10
if dataset == 'cifar10':
    class1 = 3                      # {0,1,2,3,4,5,6,7,8,9}
    class2 = 8                      # {0,1,2,3,4,5,6,7,8,9}
    totalsamp = 3000                # int within [50,10000] or None (for max)


# For custom
Xtrain_file = '/path/to/data.npy'   # (Xtrain.shape[0] = total samples and Xtrain is numpy array, no other constraint)
ytrain_file = '/path/to/label.npy'  # (ytrain.shape = (Total Samples, 1) and ytrain is numpy array, no other constraint)

wconst = 'maxmax'                   # {'minmax', 'maxmax', #ANYTHING} #ANYTHING for no constraint 
normalised = False                  # normalised data? {True, False}
solver = 'MCTM'                      # Current options are STM, SHTM, MCM, MCTM
k = 5                               # positive int
onlyonce = True                     # {True, False}
h = 3                               # non negative int
C = 10                             # positive float
rank = 3                            # positive int
constrain = '!lax'                   # lax constrain on W {'lax', #ANYTHING} using MCM / MCTM #ANYTHING for not lax
wnorm = 'L1'                        # {'L1', 'L2'} for L1 or L2 norm using STM / SHTM

visualiser = False
visuals_folder = '/path/to/folder'  # folder path to save files to

