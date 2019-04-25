# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:30:07 2019
@author: 264401K
"""
import pygimli as pg
import numpy as np
import os
import matplotlib.pyplot as plt

def getTopo(hull):
    topo = hull[(hull[:, 1] >= -5)]
    topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
    _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
    tu = topo[idx]
    for i in enumerate(idc):
            if i[1] > 1:
                tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
    _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
    topo = tu[idx]
    return topo

def idx(arr, val):
    idx = np.where( np.abs(arr-val) == np.abs(arr-val).min())
    return idx[0][0]

print('Creating PLC...')
# Load topography/simulation domain
hullFID = r"G:\BERT\data\Constraints\Regions\hull.txt"

# Extract topography (i.e. everything above a certain y)
topo = getTopo(np.loadtxt(hullFID))

# Insert a node at 0,0 for water table constraint
topo = np.insert(topo, idx(topo,0), [0,0], 0)

# Interp Array
x = np.linspace(topo[:,0].min(),topo[:,0].max(),len(topo)*4)
xp = topo[:,0]
fp = topo[:,1]
f_ = np.interp(x,xp,fp)
topo = np.stack([x,f_], axis = 1)

# Get sensors and make sure a node exists
#data = pg.load(r"G:\BERT\data\Constraints\Regions\GR\")
data = pg.load(r"G:\BERT\data\Constraints\Data\raw\3MLpy\gr\5m\QR_Quad_200md_3MLpy_MASS_5m_data_noise5pc_gr.ohm")
sensors = np.asarray(data.sensors())[:,:2]
for s in sensors:
    print(s)
    topo[idx(topo, s)] = s

# Re-insert node at 0,0 for water table line
topo = np.insert(topo, idx(topo,0), [0,0], 0)
topo = topo[topo[:,0].argsort()]

#plt.plot(topo[:,0], topo[:,1])
#plt.scatter(sensors[:,0],sensors[:,1])
#    
# Extend parameter mesh
upperLeft = [min(topo[:,0])-50,min(topo[:,1])]
upperRight = [50+max(topo[:,0]), max(topo[:,1])]
lowerLeft = [min(topo[:,0])-50, min(topo[:,1])-80]
lowerRight = [50+max(topo[:,0]), min(topo[:,1])-80]

# Insert parameter mesh domain into primary mesh
bounds = np.insert(topo, 0, [lowerLeft,upperLeft], 0)
bounds_ = np.append(bounds, [upperRight,lowerRight], axis = 0)

# Create polygon for boundary
boundary = pg.meshtools.createPolygon(verts=bounds_, isClosed=1)

# Define primary mesh extents
boundaryverts = np.array([upperLeft, [upperLeft[0] - 1000, upperLeft[1]],
                [lowerLeft[0] - 1000, lowerLeft[1]-1000],
                [lowerRight[0] + 1000, lowerRight[1]- 1000],
                [upperRight[0]+ 1000, upperRight[1]], upperRight])
para = pg.meshtools.createPolygon(boundaryverts, isClosed=False)

# Add water table (wt) and substrate (ss) boundary
wt = pg.meshtools.createLine([0,0], [boundary.xmax(),0])
ss = pg.meshtools.createLine([boundary.xmin(),-30], [boundary.xmax(),-30])

# Merge polygons (e.g. parameter boundary, primary boundary, wt and ss) 
plc = pg.meshtools.mergePLC([boundary, para, wt, ss])

# Define regions
plc.addRegionMarker([boundary.xmin() - 1, boundary.ymin() -1], 1, area = 0) # background
plc.addRegionMarker([((boundary.xmax()+boundary.xmin())/2), -1], 2, area = 10) # aquifer
plc.addRegionMarker([((boundary.xmax()+boundary.xmin())/2), 5], 3, area = 10) # vadose
plc.addRegionMarker([((boundary.xmax()+boundary.xmin())/2), -31], 4, area = 20) # substrate

# Export file
pg.meshtools.exportPLC(plc, os.path.dirname(hullFID) + '\\plc_region.poly')
pg.show(plc, showNodes = True)
pg.meshtools.createMesh(plc, 32)
print('PLC Created: {}'.format(plc))
print('Done.')