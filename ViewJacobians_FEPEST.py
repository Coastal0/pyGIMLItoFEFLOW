# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:53:27 2018

@author: 264401k
"""
from matplotlib import pyplot as plt, tri as mtri, colors, ticker
import pandas as pd
import workbook as w
import numpy as np
import pygimli as pg
from mpl_toolkits.axes_grid1 import AxesGrid
from tkinter import filedialog, Tk

def formatActiveFig():
    fig = plt.gcf()
#    fig.set_size_inches(12, 3.5)
    plt.sca(fig.get_axes()[n])
    plt.plot(topoArray[:, 0], topoArray[:, 1], 'k')
    ax = plt.gca()
    ax.tick_params(which = 'both', direction = 'in')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim(left=-100, right=950)
    ax.set_ylim(top=60, bottom=-30)
    ax.set_aspect(aspect=1)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (mASL)')

def plotWells(n):
    plt.sca(plt.gcf().get_axes()[n])
    WellCoords = pd.read_table(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat",  delim_whitespace = True)

    def getElev(array, value):
      val = array[abs(array[:,0] - value).argmin()][1]
      return val
    plt.scatter(WellCoords.iloc[n].X, WellCoords.iloc[n].Y, s = 100, color = 'yellow')
    plt.annotate(s = WellCoords.iloc[n].LABEL, xy = [WellCoords.iloc[n].X, 6+getElev(topoArray,WellCoords.iloc[n].X)], ha = 'center', fontsize = 12)

fIn = r"K:\pilot_point_sensitivities.dat"
raw_data = pd.read_table(fIn)
coords = np.stack((raw_data.X.values,raw_data.Y.values), axis = 1)
fBounds = filedialog.askopenfilenames(title = "Select MATLAB boundary and nodes (as *.mat)",filetypes = (("matlab files","*.mat"),("all files","*.*")))
hull, bPoly, nodes = w.loadHull(fBounds[0])
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
bPolyMesh = w.fillPoly(bPoly, coords, hull, 0)  # Fill boundary mesh with nodes
dMesh = w.makeDataMesh(coords, 0)  # Make a mesh using datapoints

#%% Plot individual sensitivity
nData = raw_data.shape[1]-4
cMax = raw_data.iloc[:,4:-1].max().max()
cMin = raw_data.iloc[:,4:-1].min().min()
cMax = -cMin
norm = colors.SymLogNorm(linthresh=0.01, linscale=0.1, vmin=abs(cMin), vmax=abs(cMax))

fig = plt.figure(figsize=(17,10))
grid = AxesGrid(fig, 111,
                nrows_ncols = (nData,1),
                axes_pad = 0.2,
                share_all = True,
                cbar_mode = 'single',
                cbar_location = 'bottom',
                cbar_pad = 0.5,
                cbar_size="20%")
for n in range(nData):
    print(n+3)
    data = raw_data.iloc[:,n+4]
    dInterp = (pg.interpolate(dMesh, data, bPolyMesh.cellCenter()))
#    im = pg.mplviewer.drawModel(grid[n], bPolyMesh, dInterp, cMap = 'RdBu', cmin = cMin, cmax = cMax, logScale=True)
    im = pg.mplviewer.drawMPLTri(grid[n], bPolyMesh, data=dInterp, cmap='RdBu_r', logScale=True, norm=norm, shading = 'flat')
    im.set_clim(cMin,cMax)
    grid[n].scatter(raw_data['X'],raw_data['Y'], s = 1, color = 'k')
    plotWells(n)
    formatActiveFig()
    
cbar = grid.cbar_axes[0].colorbar(im)
cbar.set_label_text('Jacobian Sensitivity')
cbar.ax.xaxis.set_label_position('top')

fig.set_size_inches([17,10])
fig.tight_layout(pad = 1)

filename = 'K:\TempPlot{}.png'.format(n)
fig.savefig(fname = filename, format = 'png', dpi = 300)

#%% Global Sensitivity
def getElev(array, value):
  val = array[abs(array[:,0] - value).argmin()][1]
  return val
def plotWells(rLim = 400):
    # Plot Wells
    plt.sca(plt.gcf().get_axes()[0])
    WellCoords = pd.read_table(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat",  delim_whitespace = True)
    
    def getElev(array, value):
      val = array[abs(array[:,0] - value).argmin()][1]
      return val
    
    for well in WellCoords.iterrows():
    #    print(well)
        if well[1].X < rLim:
            plt.plot([well[1]['X'],well[1]['X']],
                     [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
            plt.annotate(s = well[1]['LABEL'], 
                         xy = [well[1]['X'],8+getElev(topoArray,well[1]['X'])],
                         ha = 'center', fontsize = 12)
        else:
            print('Well outside viewing area')
    plt.plot([min(topoArray[:,0]), max(topoArray[:,0])],[0,0], 'k--')
    
if 'sum' not in raw_data.columns:
    raw_data['sum'] = raw_data.iloc[:,4:].sum(axis = 1)
fig2, ax2 = plt.subplots(figsize = (12,3.5))
data = abs(raw_data.iloc[:,-1])
cMax = data.max()
cMin = data.min()
#norm = colors.Normalize(vmin = cMin, vmax = cMax)
norm = colors.LogNorm(vmin = cMin, vmax = cMax)
#norm = colors.SymLogNorm(linthresh = 0.0001, vmin = cMin, vmax = cMax)

d = pg.interpolate(dMesh, data, bPolyMesh.cellCenter())

temp = pg.meshtools.fillEmptyToCellArray(bPolyMesh, d)
dInterp = temp
#pg.show(data = temp, mesh = bPolyMesh)

im2 = pg.mplviewer.drawMPLTri(ax2, bPolyMesh, data=dInterp, cmap='gist_ncar', norm=norm, shading = 'flat')
cbar2 = plt.colorbar(im2, ax = ax2, orientation = 'horizontal', aspect = 30, shrink = 0.5, pad = 0.2, extend = 'neither')
cbar2.set_label('Sum Total Jacobian Sensitivity')
cbar2.ax.xaxis.set_label_position('top')
fig2.tight_layout(pad = 1)
plotWells(950)
ax2.scatter(raw_data['X'],raw_data['Y'], marker = '+', s = 1, color = 'k')
plt.plot(topoArray[:, 0], topoArray[:, 1], 'k')
ax = plt.gca()
ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlim(left=-100, right=950)
ax.set_ylim(top=60, bottom=-30)
ax.set_aspect(aspect=1)
ax.minorticks_on()
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (mASL)')
filename = 'K:\TempPlot_cumulativeSens.png'.format(n)
fig2.savefig(fname = filename, bbox_inches='tight', format = 'png', dpi = 600)
