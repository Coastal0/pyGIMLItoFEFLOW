# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 08:33:29 2018

@author: 264401k
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

import pygimli as pg
import numpy as np

#%% Setup data
n = 5
mesh = pg.createGrid(range(n), range(n))

# Negative log-scaled data
data1 = -1*np.logspace(-2, 3, (n-1)**2)
# Positive log-scaled data
data2 = np.logspace(-2, 3, (n-1)**2)
# Combined data
data = np.concatenate([data1,data2])
# Sort data
data = np.sort(data)
# Setup norm
norm = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=-1000.0, vmax=1000.0)

#%% Using pg.show() with kwargs
#ax, cbar = pg.show(mesh, data, logScale=True, orientation="vertical", label="Logarithmic data", hold=True, nLevs=6, norm = norm)

#%% Using drawMPLTri
norm = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=-1000.0, vmax=1000.0)
fig, ax = plt.subplots()
im = pg.mplviewer.drawMPLTri(ax, mesh, data=data, cMin=None, cMax=None, cmap='RdBu', logScale=True, norm=norm)
cbar = fig.colorbar(im, ax=ax, extend='both')

#%% Colorbar Customization



def logformat(cbar):
    new_labels = []
    for label in cbar.ax.yaxis.get_ticklabels():
        print(label)
        new_labels.append()
        new_labels.append("10$^{%d}$" % np.log10(float(label.get_text())))
    cbar.ax.yaxis.set_ticklabels(new_labels)

#logformat(cbar)