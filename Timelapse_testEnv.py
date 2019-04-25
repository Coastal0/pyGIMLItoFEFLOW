# -*- coding: utf-8 -*-

import pandas as pd
import pygimli as pg
from pygimli.physics.petro import resistivityArchie as pgArch
import pybert as pb
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Extract the concave hull from MATLAB output.
with h5py.File('boundsXY.mat', 'r') as file:
    hImport = np.array(file['boundsXY'])
    hull = hImport.T
print(hull.shape, 'coordinates found')

# Remove duplicate coordinates.
if (hull[0, :] == hull[-1::]).all():
    print('Duplicate coordinates found (start/end)')
    hull = hull[0:-1]
    print(hull.shape, 'coordinates remaining')
# Round to 5 decimal places (avoids floatingpoint issues later)
hull = np.round(hull, decimals = 5)
    
# Create the exterio boundary of the FEFLOW mesh from the outer bounds.
bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
print('Boundary Polygon (bPoly):', bPoly)

# Read in mass-concentration node data
#   data = pd.read_pickle('data_t0')
fName = 'QR_IWSS_1998+_30day.dat'
data = pd.read_table(fName, delim_whitespace=True)
print('Number of nodes found =', max(data.Node))

# Extraxt XY coordinate pairs
coords = np.round(np.stack((data.X[data.Time == data.Time[0]].values, data.Y[data.Time == data.Time[0]].values), axis=1), decimals = 5)

# Fill the empty polygon with nodes
print('Filling polygon with nodes...')
nnodes = 0 # i.e. nodes skipped (must match #hull nodes)
mnodes = 0 # i.e. node added to mesh (must = maxnodes - hullnodes)
for i in enumerate(coords):
    if any(np.all(coords[i[0], :] == hull[:], axis=1)):
        mnodes += 1
    else:
        bPoly.createNode(i[1])
        nnodes += 1
# Create and show the mesh (Warning; May crash)
bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
pg.show(bPolyMesh)
print(bPolyMesh)
print(len(data.MINIT[data.Time == min(data.Time)]))

# Check each cell/node has a value.
i=0
for n in bPolyMesh.nodes():
    if  len(n.cellSet()) == 0:
        #print(n, n.pos(), " have no cells!")
        i=i+1
print(str(i)+' nodes have no cells')

#%% Get Topography
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
print('Creating array...')
sensor_firstX = 0
sensor_lastX = 500
sensor_dx = 5

sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
sensor_z = np.interp(sensor_x, topo[:, 0], topo[:, 1])
sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
      
ertScheme = pb.createData(sensors, schemeName='gr')

# Topography before (left-of) electrodes
topoPnts_x = np.arange(topo[0,0],sensor_firstX,sensor_dx)
topoPnts_z = np.interp(topoPnts_x, topo[:, 0], topo[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)

    # Create ERT mesh (based on sensors)
print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33,
                                      paraMaxCellSize=0.5, paraDepth=100,
                                      paraDX=0.01)
print('ERT Mesh = ', meshERT)
meshERTName = 'meshERT_test'
meshERT.save(meshERTName)

#%% Data mesh
dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords), quality = 0)
#pg.show(dMesh)
print(dMesh)

#%% 
times = pd.unique(data.Time)
nameList = []
for t in enumerate(times):
    print(str(round(t[1],2)))
    dataVector = data.MINIT[data.Time == t[1]].values
    dInterp = pg.interpolate(dMesh, dataVector, bPolyMesh.cellCenter())
    dInterp.save('dInterp_'+str(round(t[0],1)))
    print(t,dInterp)
    # Convert mass concnetration to water conductivity
    print('Conveting from fluid to formation resistivity...')
    k = 0.612  # Linear conversion factor from TDS to EC
    sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
    print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
    rFluid = 1/sigmaFluid
    print('Interpolating mesh values...')
    
    resBulk = pgArch(rFluid, porosity=0.3, m=2, mesh=bPolyMesh, meshI=meshERT, fill=1)
    print('No.# Values in fluid data',resBulk.shape[0])
    print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
    print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())
          
    for c in meshERT.cells():
        if c.center()[1] > 0:
            resBulk[c.id()] = 1000. # Resistivity of the vadose zone
        elif c.center()[1] < -30:
            resBulk[c.id()] = 20. # Resistivity of the substrate      
    resBulkName = 'resBulk_month'+str(+int(round(t[1]*12,0)))+'_ID_'+str(t[0])+'.vector'
    np.savetxt(resBulkName,resBulk)
    
    printFlag = 1
    if printFlag == 1:
        ax = plt.gca()
        covVector = np.ones(len(resBulk))*0.5
        for r in enumerate(resBulk):
            if r[1] == min(resBulk):
                covVector[r[0]] = 1
                
        dataplot, _ = pg.show(ax=ax, mesh=meshERT, data=resBulk,
                               cmap='jet_r', showMesh=0, cMin=2, cMax=500,
                               colorBar=True,coverage = covVector)
        # Formatting
        ax.set_xlim(left=-10, right=500)
        ax.set_ylim(top=40, bottom=-50)
        ax.minorticks_on()
        ax.set_title(t, loc='left')
        ax.set_ylabel('Elevation (mASL)')
        ax.set_xlabel('Distance (m)')
        dataplot.plot(dataplot.get_xlim(), [-30, -30], color='black')
        dataplot.plot(dataplot.get_xlim(), [0, 0], color='black')
        fig = plt.gcf()
        fig.set_size_inches(19, 5)
        fig.tight_layout()

        plt.ion()
        plt.show()
        plt.pause(0.001)
        # Assign names
        filename = os.path.basename(os.path.normpath(os.getcwd())+'_'+str(t[0]))
        # Store names for GIF maker
        nameList.append(filename+'.png')
        plt.savefig('{}.png'.format(filename),dpi=150)
        # Clear figure (rather than recreate each time)
        plt.gcf().clear()
plt.close('all')
    # Make GIF
images = []
for i in nameList:
    images.append(imageio.imread(i))
fTimes=list(np.ones(len(images))*0.2)
fTimes[0] = 1
fTimes[-1] = 1
imageio.mimsave('./test.gif',images, duration=fTimes)

    
    simulateFlag = 0
    if simulateFlag == 1
        ert = pb.ERTManager(debug=True)
        print('#############################')
        print('Forward Modelling...')
    #    ertScheme.set('k', pb.geometricFactor(ertScheme))
        ertScheme.set('k', np.ones(ertScheme.size()))
    
        simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme,
                                      noiseAbs=0.0,  noiseLevel = 0.01)
        simdata.set("r", simdata("rhoa"))
        dataName = 'data_'+str(t[0])+'.ohm'
        simdata.save(dataName, "a b m n r err")
        print('Done.')
        print(str('#############################'))
#    pg.show(meshERT, resBulk)

