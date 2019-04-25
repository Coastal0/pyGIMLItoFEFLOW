# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:10:57 2018

@author: 264401k
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.spatial as scp
import numpy as np

fIn = r"K:/FePest/QR_Quad_Fepest_NM3_output_mass_stable.fem_HydCondValues_asCentre.dat"
data = pd.read_table(fIn, delim_whitespace = True)


x = data['X'].values
y = data['Y'].values
z = data['COND'].values

xy = np.stack([x,y]).T

test = tri.Triangulation(x,y)

init_mask_frac = 0.0
min_circle_ratio = .01

mask = tri.TriAnalyzer(test).get_flat_tri_mask(min_circle_ratio)
test.set_mask(mask)



plt.triplot(x,y)


flat_tri = tri.Triangulation(x, y)
flat_tri.set_mask(~mask)
plt.triplot(flat_tri, color='red')
