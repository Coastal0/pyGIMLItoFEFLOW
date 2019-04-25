# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:30:07 2019

@author: 264401K
"""
import pygimli as pg
import numpy as np

meshFID = r"G:\BERT\data\Constraints\Regions\GR\mesh\mesh.bms"
mesh = pg.load(meshFID)
for c in mesh.cells():
    # Assign to Vadose region
    if c.center()[1] > 0:
        if c.marker() == 2:
            c.setMarker(3)
    # Assign to substrate region
    elif c.center()[1] < -30:
        if c.marker() == 2:
            c.setMarker(4)
            
mesh.save(meshFID[:-4]+'_regions_noOcean.bms')
pg.meshtools.exportPLC(mesh, meshFID[:-4]+'_regions.poly')

pg.show(mesh, markers = True)

#%% Assign Ocean region
# Load seabed topography
#seafloorFID = r"G:\BERT\data\Constraints\Regions\seafloor.txt"
#seafloor = np.loadtxt(seafloorFID)
#
#def find_nearest_index(array, value):
#    array = np.asarray(array)
#    idx = (np.abs(array - value)).argmin()
#    return idx
#
#for c in mesh.cells():
#    # Caclulate only relevent cells
#    if c.center().x() < np.min(np.abs(seafloor[:,0])) and \
#        np.abs(c.center().y()) < np.max(np.abs(seafloor[:,1])):
#        # Extract x,y's
#        mshx = c.center().x()
#        mshy = c.center().y()
#        idx = find_nearest_index(seafloor[:,0],mshx)
#        sfx = seafloor[idx][0]
#        sfy = seafloor[idx][1]
#        # If mesh-y is above seafloor, assign it to a region
#        if mshy > sfy and mshy < 0 and c.marker() != 1:
##            print(c.marker())
#            c.setMarker(5)
#mesh.save('mesh/mesh_regions.bms')
#pg.meshtools.exportPLC(mesh, 'mesh/mesh.poly')
#

#%% Show and save mesh figure
#import matplotlib.pyplot as plt
#ax, cb = pg.show(mesh, markers = True, showMesh = False)
#ax.set_xlim(-50,100)
#ax.set_ylim(-40,20)
#ax.set_xlabel('Distance (m)')
#ax.set_ylabel('Elevation (mASL)')
#plt.gcf().savefig('Fig_Regions.png', bbox_inches='tight', 
#       format = 'png', dpi = 600)
#
