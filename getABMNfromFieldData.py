# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:38:47 2018

@author: 264401k
"""

import pybert as pb
import pandas as pd
import pygimli as pg
import numpy as np
import workbook as w
import tkinter as tk
from tkinter import filedialog


print("pyGIMLI:", pg.__version__)
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)

tk.Tk().withdraw()
fNames = tk.filedialog.askopenfilenames()
print(fNames)

# Load hull and topography
dataDict, data, coords = w.loadData(fNames)
hull, bPoly = w.loadHull(r"K:\boundsXY_10kCells.mat")
nodes = w.loadNodes(r"K:\nodesXY.mat")
topoArray = w.getTopo(hull)
bPolyMesh = w.fillPoly(bPoly, coords, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
dMesh = w.makeDataMesh(coords, 0)  # Make a mesh using datapoints
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    data = dataDict[d][0]
    if 'MINIT' not in data:
        print('Mass not present... breaking.')
        break
    else:
        data['dataCol'] = data['MINIT']
    dInterpDict[d], times = w.makeInterpVector(data, dMesh, bPolyMesh)  # Add data to the nodes

def getElev(array, value):
  val = array[abs(array[:,0] - value).argmin()][1]
  return val

# Load ertSchemefile
raw_ertScheme = pb.load(r"G:\BERT\data\Field\Dipole\dp.bin")

# Initialize arrays
a = b = m = n = x = y = z = []

# Define ertScheme
a = raw_ertScheme("a").array()
b = raw_ertScheme("b").array()
m = raw_ertScheme("m").array()
n = raw_ertScheme("n").array()
abmn = np.stack((a,b,m,n)).T

# Define sensors
sensors = np.asarray(raw_ertScheme.sensors())
x = sensors[:,0]
y = sensors[:,1]
z = sensors[:,2]

# Get topography along profile
xx = x.reshape(-1,1)
yy = np.zeros((len(xx),1))
zz = np.zeros((len(xx),1))
for i, pos in enumerate(xx):
    zz[i] = getElev(topoArray,xx[i])
    
xx = xx.ravel()
yy = yy.ravel()
zz = zz.ravel()
new_sensors = np.stack((xx,yy,zz)).T

# Assign topography and ertScheme to new container
ertScheme = pg.DataContainerERT()
for pos in new_sensors:
    ertScheme.createSensor(pos)
ertScheme.resize(len(a))
ertScheme.set('a', a)
ertScheme.set('b', b)
ertScheme.set('m', m)
ertScheme.set('n', n)

ertScheme.save('ErtScheme_FieldData.ohm')

# Topography before (left-of) electrodes
topoPnts_x = np.arange(topoArray[0,0],xx[0],xx[1]-xx[0])
topoPnts_z = np.interp(topoPnts_x, topoArray[:, 0], topoArray[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(new_sensors[:,[0,2]],0,topoPnts_stack,0)

#botLeft = np.array([topoPnts[0][0],-100])
#botRight = np.array([topoPnts[-1][0],-100])
#
#topoPnts = np.insert(botLeft,0,topoPnts,0)


# Create ERT mesh (based on sensors)
print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=32,
                                      paraMaxCellSize=3, paraDepth=100,
                                      paraDX=0.01)
print(meshERT)
from pygimli.physics.petro import resistivityArchie as pgArch

print('Converting fluid cond to formation cond...')
k = 0.55  # Linear conversion factor from TDS to EC
sigmaFluid = dInterpDict[0] / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
rFluid = 1/sigmaFluid
print('Interpolating mesh values...')

resBulk = pgArch(rFluid, porosity=0.3, m=2, mesh=bPolyMesh, meshI=meshERT, fill=1)
print('Resistivity range: ', min(resBulk), max(resBulk), 'Ohm.m')

print('No.# Values in fluid data',resBulk.shape[0])
print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())
print("Applying background, substrate, vadose resistivity...")

for c in meshERT.cells():
    if c.center()[1] > 0:
        resBulk[c.id()] = 1000. # Resistivity of the vadose zone
    elif c.center()[1] < -30:
        resBulk[c.id()] = 20. # Resistivity of the substrate

for c in meshERT.cells():
    if c.marker() == 1 and c.center()[0] < 0 and c.center()[1] > -30:
        resBulk[c.id()] = 2 # Resistivity of the ocean-side forward modelling region.
print('Done.')
pg.show(meshERT,resBulk)
ert = pb.ERTManager(debug=True)
simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, verbose = True, noiseAbs=0.0,  noiseLevel = 0.01)
simdata.save('simdata.dat')
ertScheme.set('k', np.ones(ertScheme.size()))
simdata.set("r", simdata("rhoa"))

simdata.save('simdata_r.dat')
print('Done.')