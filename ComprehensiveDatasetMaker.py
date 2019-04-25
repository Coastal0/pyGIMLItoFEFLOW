# -*- coding: utf-8 -*-
"""
Comprehensive ERT Scheme generator (from Florian)
"""

import pybert as pb
import numpy as np
import itertools

def abmn(n):
    """
    Construct all possible four-point configurations for a given
    number of sensors after Noel and Xu (1991).
    """
    combs = np.array(list(itertools.combinations(list(range(1, n+1)), 4)))
    perms = np.empty((int(n*(n-3)*(n-2)*(n-1)/8), 4), 'int')
    print(("Comprehensive data set: %d configurations." % len(perms)))
    for i in range(np.size(combs, 0)):
        perms[0+i*3, :] = combs[i,:] # ABMN
        perms[1+i*3, :] = (combs[i, 0], combs[i, 2], combs[i, 3], combs[i, 1]) #AMNB
        perms[2+i*3, :] = (combs[i, 0], combs[i, 2], combs[i, 1], combs[i, 3]) #AMBN

    return perms - 1

# Create empty DataContainer
data = pb.DataContainerERT()

# Add electrodes
n = 44
spacing = 10
for i in range(n):
    data.createSensor([i * spacing, 0.0]) # 2D, no topography

# Add configurations
cfgs = abmn(n) # create all possible 4P cgfs for n electrodes
for i, cfg in enumerate(cfgs):
    data.createFourPointData(i, *map(int, cfg)) # (We have to look into this: Mapping of int necessary since he doesn't like np.int64?)

# Optional: Save in unified data format for use with command line apps
data.save("data.shm", "a b m n")
