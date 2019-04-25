# -*- coding: utf-8 -*-
import pygimli as pg
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import matplotlib.colors as colors
import glob
import re
import os

def getFinalVector(invPath):
    vpath = glob.glob(invPath+"\model_*.vector")
    vectors=list()
    for path in vpath:
        vectors.append(os.path.basename(path))
    v=np.zeros(len(vectors))
    for s in enumerate(vectors): v[s[0]] = int(re.split(r'[_.]', s[1])[1])
    finalVector = invPath+"\\"+vectors[np.where(v == v.max())[0][0]]
    return finalVector
#%% Compare Results against models
# To-Do: read inverted models (see viewVectors.py) and compare them to forward model.
# Thoughts: Nodes probably won't be similar. Will need to interpolate somehow.

#%% Load up the right files
# Define first inversion model
path1 = r'G:\BERT\data\Constraints\Regions\GR\SeawaterWedgeModel\result190418_1343'
mesh1 = pg.load(path1+'\mesh\meshParaDomain.bms')
data1 = pg.load(path1+'\\resistivity.vector')
print('Model 1 #Cells = '+str(mesh1.cellCount()))
print('Model 1 Vector, #Elements = ' + str(data1.size()))

# Define second inversion model
path2 = r'G:\BERT\data\Constraints\Regions\GR\SeawaterWedgeModel\result190418_1309'
mesh2 = pg.load(path2+'\mesh\meshParaDomain.bms')
data2 = pg.load(path2+'\\resistivity.vector')
print('Model 2 #Cells = '+str(mesh2.cellCount()))
print('Model 2 Vector, #Elements = '+str(data2.size()))

#%% Compare the forward model against the inversion results
# Interpolate inversion results onto forward model mesh
dCompare = pg.interpolate(mesh2, data2, mesh1.cellCenter())
len(dCompare)
# Show interpolated mesh
ax, cb = pg.show(mesh1,dCompare, colorBar=True, cmap='jet_r', cMax = 1000, cMin = 1) # I've forgotten how to logscale it.
ax.minorticks_on()
#ertScheme = pg.load('ertScheme')
#pg.mplviewer.drawSensors(ax, ertScheme.sensorPositions(),Facecolor = '0.75', edgeColor = 'k', diam = 1)
ax.set_xlim(left=-10, right=500)
ax.set_ylim(top=40, bottom=-60)

# Calculate difference between the two
diffs = ((data1 - dCompare)/dCompare)*100
diffs = diffs.array()
print(diffs)


fig, ax = plt.subplots(nrows = 3, ncols =1, sharex=True, sharey=True)
pg.show(mesh1,data1, ax = ax[0], cmap = 'jet_r', colorBar = True, orientation = 'vertical', cMin = 1, cMax = 2000, label = '$\Omega$ m')
pg.show(mesh2,data2, ax = ax[1], cmap = 'jet_r', colorBar = True, orientation = 'vertical', cMin = 1, cMax = 2000, label = '$\Omega$ m')
pg.show(mesh1,diffs, ax = ax[2], cmap='RdBu', colorBar = True, orientation = 'vertical', cMin = -50, cMax = 50, label = '% Difference')
ax[0].set_xlim(left=-30, right=300)
ax[0].set_ylim(top=40, bottom=-50)

for a in ax:
    a.minorticks_on()
    a.set_ylabel('Elevation (mASL)')
    a.set_xlabel('Distance (m)')
    a.tick_params(which = 'both', direction = 'in')
    a.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    a.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

fig = plt.gcf()
fig.tight_layout()
fig.savefig('Comparison.png', bbox_inches='tight', dpi=600)
# Histogram
#fig = plt.figure(figsize = (17,5))
#plt.hist(diffValsa,bins=100,density=True, range=[-100,100])

#%%  Show the forward, inverted, and difference plots
from mpl_toolkits.axes_grid1 import AxesGrid

fig = plt.figure(figsize=(10,14))
grid = AxesGrid(fig, 111,
                nrows_ncols = (2,1),
                axes_pad = 0.5,
                share_all = True,
                cbar_mode = 'single',
                cbar_location = 'bottom',
                cbar_pad = 0.3,
                cbar_size="20%")
plt.suptitle('Forward Model vs Inversion Results',fontsize=14, fontweight='bold')


im = pg.mplviewer.drawModel(grid[0], fwdMesh, fwdData, cmap = 'jet_r')
#im.cmap.set_under(color = '0.75')
im.set_clim(1,1000)
grid[0].minorticks_on()
grid[0].set_title('Forward Model')
grid[0].set_xlim(left=-10, right=500)
grid[0].set_ylim(top=25, bottom=-50)
grid[0].set_ylabel('Elevation (mASL)')
grid[0].set_xlabel('Distance (m)')

im = pg.mplviewer.drawModel(grid[1], invMesh, invResult, cmap = 'jet_r')
#im.cmap.set_under(color = '0.75')
im.set_clim(1,1000)
grid[1].minorticks_on()
grid[1].set_title('Inversion Result')
grid[1].set_xlim(left=-10, right=500)
grid[1].set_ylim(top=25, bottom=-50)
grid[1].set_ylabel('Elevation (mASL)')
grid[1].set_xlabel('Distance (m)')


cbar = grid.cbar_axes[0].colorbar(im)
#labels = np.ndarray.tolist(np.around(10**cbar.ax.locator(),0))
#cbar.ax.set_xticklabels(labels)
cbar.set_label_text('Formation Resistivity ($\Omega$$\cdot$m)')
cbar.ax.locator()
tick_locator = ticker.LogLocator(subs = range(8))
#tick_locator = ticker.MaxNLocator(nbins = 10)
cbar.ax.xaxis.set_major_locator(tick_locator)
fig = plt.gcf()
fig.tight_layout()