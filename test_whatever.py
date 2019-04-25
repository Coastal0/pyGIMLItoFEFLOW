# -*- coding: utf-8 -*-
"""

"""
import workbook as w
import os
import matplotlib.pyplot as plt

hull, bPoly = w.loadHull(r"G:\results\boundsXY.mat")

fName = r"F:\results\QR_stable_015mday.dat"
data, coords = w.loadData(fName)

bPolyMesh = w.fillPoly(bPoly, coords, hull)

w.checkMesh(bPolyMesh)

topo = w.getTopo(hull)

ertScheme, meshERT = w.createArray(0, 500, 5, 'gr', topo)

dMesh = w.makeDataMesh(coords, 0)

dInterp = w.makeInterpVector(data, dMesh, bPolyMesh)

resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)

simdata = w.simulate(meshERT, resBulk, ertScheme)

#%% Show models, etc.

import pygimli as pg
ax, cb = pg.show(meshERT, resBulk, colorBar=True, cmap='jet_r', showMesh=0, cMin = 3, cMax = 500)
ax.minorticks_on()
pg.mplviewer.drawSensors(ax, ertScheme.sensorPositions(),Facecolor = '0.75', edgeColor = 'k', diam = 1)
ax.set_xlim(left=-20, right=500)
ax.set_ylim(top=30, bottom=-40)
ax.set_title(os.path.split(fName)[-1], loc=  'left')
ax.set_ylabel('Elevation (mASL)')
ax.set_xlabel('Distance (m)')

cb.ax.minorticks_on()
cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)

cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
fig = plt.gcf()
fig.tight_layout()

vecName = r"G:\BERT\data\GR_5m_Cases\result180314_1748_025\model_13.vector"
eriModelVec = pg.load(vecName)
eriModelMesh = pg.load(r"G:\BERT\data\GR_5m_Cases\result180314_1748_025\mesh\meshParaDomain.bms")
print(eriModelMesh, eriModelVec)

ax2, cb2  = pg.show(eriModelMesh,eriModelVec,colorBar=True, cmap='tab20c', showMesh=1, cMin = 3, cMax = 500)
ax2.set_xlim(left=-20, right=200)
ax2.set_ylim(top=25, bottom=-40)
ax2.set_title(vecName, loc=  'left')
ax2.set_ylabel('Elevation (mASL)')
ax2.set_xlabel('Distance (m)')
cb2.ax.minorticks_on()
cb2.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)

fig = plt.gcf()
fig.tight_layout()
