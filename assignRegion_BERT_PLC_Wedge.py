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

# Get sensors and make sure a node exists
#data = pg.load(r"G:\BERT\data\Constraints\Regions\GR\")
data = pg.load(r"G:/BERT/data/Constraints/Data/raw/3MLpy/gr/2m/QR_Quad_200md_3MLpy_MASS_2m_data_noise5pc_gr.ohm ")
sensors = np.asarray(data.sensors())[:,:2]
for s in sensors:
#    print(s)
    topo[idx(topo, s)] = s
    
# Insert the node at 0,0 again
topo = np.insert(topo, idx(topo,0), [0,0], 0)
topo = topo[topo[:,0].argsort()]

# Interp Array
x = np.linspace(topo[:,0].min(),topo[:,0].max(),len(topo)*4)
xp = topo[:,0]
fp = topo[:,1]
f_ = np.interp(x,xp,fp)
#plt.scatter(x,f_)

topo = np.stack([x,f_], axis = 1)
# Insert the node at 0,0 a third time
topo = np.insert(topo, idx(topo,0), [0,0], 0)
topo = topo[topo[:,0].argsort()]
#plt.plot(topo[:,0], topo[:,1])
#plt.scatter(sensors[:,0],sensors[:,1])

# Extend parameter mesh
upperLeft = [min(topo[:,0])-50,topo[0,1]]
upperRight = [50+max(topo[:,0]), topo[-1,1]]
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

# Add wedge
wdg1 = pg.meshtools.createLine([0,0], [50,-30])

# Merge polygons (e.g. parameter boundary, primary boundary, wt and ss) 
plc = pg.meshtools.mergePLC([boundary, para, wt, ss, wdg1])

# Define regions
plc.addRegionMarker([boundary.xmin() - 1, boundary.ymin() -1], 1, area = 0) # background
plc.addRegionMarker([((boundary.xmax()+boundary.xmin())/2), -1], 3, area = 5) # fresh aquifer
plc.addRegionMarker([(boundary.xmin()+1), -29], 2, area = 5) # saline aquifer
plc.addRegionMarker([(boundary.xmax()-1), 5], 5, area = 0) # vadose
plc.addRegionMarker([((boundary.xmax()+boundary.xmin())/2), -31], 4, area = 50) # substrate

mesh = pg.meshtools.createMesh(plc, 32)
# Export file
pg.meshtools.exportPLC(plc, os.path.dirname(hullFID) + '\\plc_region_wedge.poly')

#ax, _ = pg.show(mesh, markers = True, showMesh = True)
#ax.set_xlim(-50,130)
#ax.set_ylim(-95,20)
#ax.set_xlabel('Distance (m)')
#ax.set_ylabel('Elevation (mASL)')
#plt.gcf().set_size_inches(15,5)
#plt.gcf().set_tight_layout('tight')
#plt.gcf().savefig(fname = (os.path.dirname(hullFID) + '\\plc_region.png'), bbox_inches='tight', format = 'png', dpi = 600)
print('PLC Created: {}'.format(plc))
print('Done.')