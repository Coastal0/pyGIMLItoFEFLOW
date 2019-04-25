# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:16:17 2018

@author: 264401k
"""

import workbook as w
import matplotlib.pyplot as plt
import pandas as pd
import pygimli as pg
import numpy as np

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

#%% Load data and hull.
hull, bPoly = w.loadHull(r"K:/boundsXY.mat")

#fName, fName2 = [
#"file:///K:/QR_Quad_Fepest_NM3_output_mass_flowTest_MASSCONC_1p09MLpa.dat",
#"file:///K:/QR_Quad_Fepest_NM3_output_mass_flowTest_SAT_1p09MLpa.dat"
#]

fName, fName2 = [
"file:///K:/Porosity/QR_Quad_Richards_HeadBase_sat.dat",
"file:///K:/Porosity/QR_Quad_Richards_HeadBase_mass.dat"
]
data, coords = w.loadData(fName)
data2, coords2 = w.loadData(fName2)

if (coords2 == coords).all():
    #append dataset to dataframe
    data['SINIT'] = data2['SINIT'] # The header for data2 will depend on the datatype exported. adjust as needed
    #delete data2
    del data2
else:
    print('Error: Coordinates do not match')

bPolyMesh = w.fillPoly(bPoly, coords, hull) # Fill boundary mesh with nodes at coordinates
w.checkMesh(bPolyMesh) # Check each node has a value. If !0, ERT will not work.
topoArray = w.getTopo(hull) # Extract the topography from the concave hull
dMesh = w.makeDataMesh(coords, 0) # Make a mesh with the datapoints (interpolates across topography, hence need for above steps)

data['dataCol'] = data['MINIT'] * data['SINIT']
dInterp = w.makeInterpVector(data, dMesh, bPolyMesh) # Add data to the nodes

ax, cb = pg.show(bPolyMesh, dInterp, fillContour=True, colorBar = True, cMap=discrete_cmap(11, 'jet'), showMesh=0)


nData = pg.cellDataToPointData(bPolyMesh, dInterp)
#%%
mesh = bPolyMesh
data = dInterp
for c in mesh.cells():
    if c.center()[1] > 0:
        data[c.id()] = 0
cmap = plt.cm.get_cmap('binary')
cmap.set_under('lightgray',0.2)
# Salinity
#ax, cb = pg.show(mesh, data, label="Salinity (mg/l)", cMap=cmap, vmin = 360, cMin=360, cMax=36000, extend="both", logScale=True)
#pg.mplviewer.drawField(ax, mesh, nData, cMin=500, nLevs=5, cMax=35000, logScale=True, fillContour=False, color=['black'], linewidths=1, alpha=1)

# Hyd. Conductivity
ax, cb = pg.show(mesh, data, label="Hydrualic Conductivity (m/d)", cMap=cmap,  colorBar = False, vmin = 150, cMin=150, cMax=1500, extend="both", logScale=True)
pg.mplviewer.drawField(ax, mesh, nData, cMin=140, nLevs=5, cMax=700, logScale=True, fillContour=False, color=['black'])
plt.sca(plt.gcf().get_axes()[0])

plt.plot(topoArray[:,0], topoArray[:,1],'k')
ax.set_xlim(-20,450)
plt.gcf().set_size_inches(12,4)
plt.gcf().set_tight_layout('tight')
ax.set_title('Solute Distribution -- No High-K Structure',loc= 'left')
ax.set_ylabel('Elevation (mASL)')
ax.set_xlabel('Distance (m)')
ax.minorticks_on()

#%% Plot Wells
plt.sca(plt.gcf().get_axes()[0])
WellCoords = pd.read_table(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat",  delim_whitespace = True)

def getElev(array, value):
  val = array[abs(array[:,0] - value).argmin()][1]
  return val

for well in WellCoords.iterrows():
#    print(well)
    plt.plot([well[1]['X'],well[1]['X']], [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
    plt.annotate(s = well[1]['LABEL'], xy = [well[1]['X'],5+getElev(topoArray,well[1]['X'])])

