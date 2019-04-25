# -*- coding: utf-8 -*-

# Data Management/General Requirements
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
from dateutil.relativedelta import relativedelta

import pygimli as pg

def findTopo(hull):
    topo = hull[(hull[:, 1] >= 0)]
    topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
    _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
    tu = topo[idx]
    for i in enumerate(idc):
            if i[1] > 1:
                tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
    topo = tu
    return topo

#%% Read concave hull solution
# Compute the concave hull (alpha-shape) of the pointcloud.
print('Computing hull...')
# Back to matlab.
with h5py.File('boundsXY.mat', 'r') as file:
    print(list(file.keys()))
    hImport = np.array(file['boundsXY'])

hull = hImport.T
print(hull.shape, 'coordinates found')
# Check for duplicate start/end nodes (pyGimli doesn't handle duplicates).
if (hull[0, :] == hull[-1::]).all():
    print('Duplicate coordinates found (start/end)')
    hull = hull[0:-1]
    print(hull.shape, 'coordinates remaining')

# Create the boundary of the FEFLOW mesh (i.e. pointcloud).
bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
print('Boundary Polygon (bPoly):', bPoly)

#%% Read data points
# Use Pandas to read data in as table.
dList  = ['TimeLapse_biannual.dat']

#%% Read concave hull solution from matlab
print('Reading hull...')
with h5py.File('boundsXY.mat', 'r') as file:
#        print(list(file.keys()))
    hImport = np.array(file['boundsXY'])
    hull = hImport.T
print(hull.shape, 'coordinates found')
# Check for duplicate start/end nodes (pyGimli doesn't handle duplicates).
if (hull[0, :] == hull[-1::]).all():
    print('Duplicate coordinates found (start/end)')
    hull = hull[0:-1]
    print(hull.shape, 'coordinates remaining')

# Round impossibly small numbers down to zero.    
for i in hull:
    if i[1] != 0 and np.around(i[1],2) == 0:
        i[1] = np.around(i[1],2)

# Create the boundary of the FEFLOW mesh (i.e. pointcloud).
bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
print('Boundary Polygon (bPoly):', bPoly)
    
#%% Load datafiles (output from FEFLOW)
# Use Pandas to read data in as table.
print('Reading data...')
data = pd.read_table(dList[0], delim_whitespace=True)
print(max(data.Node), 'nodes found.')
# Extract coordinates of mesh.
if 'Time' in data.columns:
    print(pd.unique(data.Time).size, 'time steps found:', pd.unique(data.Time))
    coords = np.stack((data.X[data.Time == data.Time[0]].values,
                       data.Y[data.Time == data.Time[0]].values), axis=1)
    print(len(coords), 'X-Y locations found for time', data.Time[0])
    maxNodes = max(data.Time == data.Time[0])
else:
    maxNodes = max(data.Node)
    coords = np.stack((data.X.values, data.Y.values), axis=1)
    if maxNodes != coords.shape[0]:
        print('Number of reported nodes =', maxNodes)
        print('Number of nodes found =', coords.shape[0])
        print('Number of nodes does not match. (Inactive elements in FEFLOW?)')
    else:
        print(len(coords), 'X-Y locations found.')
#%%
# Fill boundary with nodes.
print('Filling polygon with nodes...')
nnodes = 0 # Added node counter
mnodes = 0 # Boundary node counter
for i in enumerate(coords):
    if any(np.all(np.round(coords[i[0], :],3) == np.round(hull[:],3), axis=1)):
        mnodes += 1
    else:
        bPoly.createNode(i[1])
        nnodes += 1
print('Found', mnodes, 'boundary nodes.')
print('Added', nnodes, 'nodes.')
print('Total nodes = ', nnodes+hull.shape[0])
# Mesh the nodes.
print('Boundary Polygon (w/o Nodes):', bPoly)
print('Meshing...')
bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
print('Boundary Polygon (w/Nodes):', bPolyMesh)
# pg.show(bPolyMesh)
#%%
# Check that each node has an associated cell (i.e. check for corrupt mesh)            
for n in bPolyMesh.nodes():
    if  len(n.cellSet()) == 0:
        
# Mesh the data.
print('Meshing...') 
dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords))
dataStack = np.zeros((bPolyMesh.cellCount(),len(pd.unique(data.Time))))
times = pd.unique(data.Time)
print(dMesh)
for t in enumerate(pd.unique(data.Time)):
    print(t)
    dataVector = np.array(data.MINIT[t[1] == data.Time])
    dInterp = pg.interpolate(dMesh, dataVector, bPolyMesh.cellCenter())
    dataStack[:,t[0]] = dInterp

print(dInterp)
print(bPolyMesh)


# %% Show Water Conc.
#Individiual figures
#startdate = datetime.date(1998,1,1)
#
#for d in enumerate(dataStack.T):
#    print(d)
#    dataVec = d[1]
#    fig = plt.figure(figsize=(19,4))
#    ax = plt.gca()
#    ax, cb = pg.show(ax = ax,mesh = bPolyMesh,data = d[1], cmap = 'jet', colorBar=True)
#    ax.minorticks_on()
#    ax.set_ylabel('Elevation (mASL)')
#    ax.set_xlabel('Distance (m)')
#    if (round(times[d[0]],1)).is_integer():
#        timeIncY = int(round(times[d[0]]))
#        timeIncM = 0
#    else:
#        timeIncY = int(round(times[d[0]]))
#        timeIncM = 6
#    title = startdate + relativedelta(years = timeIncY, months = timeIncM)
#    ax.set_title(str(title),loc='left')
#    ax.set_xlim(left=-20, right=500)
#    ax.set_ylim(top=25, bottom=-30)
#
#    cb.set_label('Mass Concentration (mg/L)')
#    fig = plt.gcf()
#    fig.tight_layout()
#    cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
#    fig.savefig('Timelapse' + str(title)+'.png', format = 'png', dpi = 300)
#    plt.close()
    
#%% Make modelling mesh
# Get topography
topo = hull[(hull[:, 1] >= 0)]
topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
_, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
tu = topo[idx]
for i in enumerate(idc):
        if i[1] > 1:
            tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
_, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
topo = tu[idx]
            
#%% Setup array (surface)
# schemeName : str ['none']
# Name of the configuration. If you provide an unknown scheme name, all
# known schemes ['wa', 'wb', 'pp', 'pd', 'dd', 'slm', 'hw', 'gr'] are listed.
import pybert as pb

print('Creating array...')
sensor_firstX = 0
sensor_lastX = 500
sensor_dx = 5

sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
sensor_z = np.interp(sensor_x, topo[:, 0], topo[:, 1])
sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T

ertScheme = pb.createData(sensors, schemeName='dd')
for pos in ertScheme.sensorPositions():
    print(pos)

schemeName = 'ertScheme_'+str(sensor_dx)+'m'
ertScheme.save(schemeName)

# Topography before (left-of) electrodes
topoPnts_x = np.arange(topo[0,0],sensor_firstX,sensor_dx)
topoPnts_z = np.interp(topoPnts_x, topo[:, 0], topo[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)

print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33,
                                      paraMaxCellSize=3, paraDepth=100,
                                      paraDX=0.33)
print('ERT Mesh = ', meshERT)
for n in meshERT.nodes():
    if  len(n.cellSet()) == 0:
        print(n, n.pos(), " have no cells!")
        
#%% Convert to resistivity
import pygimli.physics.petro as petro
rBulk = np.zeros((meshERT.cellCount(),dataStack.shape[1]))
k=0.612
for d in enumerate(dataStack.T):
    print(d)
    sigmaFluid = d[1] / (k*10000)
    rFluid = 1/sigmaFluid
    resBulk = petro.resistivityArchie(rFluid, porosity=0.3, m=2, mesh=bPolyMesh, meshI=meshERT, fill=1)
    for c in meshERT.cells():
        if c.center()[1] > 0:
            resBulk[c.id()] = 1000. # Resistivity of the vadose zone
        elif c.center()[1] < -30:
            resBulk[c.id()] = 20. # Resistivity of the substrate
    rBulk[:,d[0]] = resBulk

#%% Forward Model
ert = pb.ERTManager(debug=True)
print('#############################')
print('Forward Modelling...')
#    ertScheme.set('k', pb.geometricFactor(ertScheme))
ertScheme.set('k', np.ones(ertScheme.size()))
for d in enumerate(rBulk.T):
    print(d)
    data = ert.simulate(mesh=meshERT, res=d[1], scheme=ertScheme, noiseAbs=0.0,  noiseLevel = 0.01)
    data.set("r", data("rhoa"))
    dataName = str(d[0])
    data.save(dataName, "a b m n r err k")
print('Done.')
print('#############################')

#%% View vectors

      