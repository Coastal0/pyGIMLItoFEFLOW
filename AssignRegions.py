#!/usr/bin/env F:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python

# This script reads a pyBERT mesh and assigns values to the regions based on some other information.
import pygimli as pg
import matplotlib.pyplot as plt
import numpy as np

#1) Load electrode positions (for topography/top of vadose zone region)
dataIn = pg.load('data_213.ohm')
sensors = np.asarray(dataIn.sensorPositions())
sensors = sensors[:,:2]

#2) Define Regions (e.g. vadose, aquifer (saline/fresh), substrate)
# Get sensor geometry and add approximate watertable.
addPoint = np.array([[sensors[-1][0], 0], [0,0]])
vadoseVerts = np.vstack((sensors,addPoint))
vadoseRegion = pg.meshtools.createPolygon(vadoseVerts, isClosed=1, marker=1)

# Takes the bottom of the vadose and adds a rectangle of some length/depth.
aquiferVerts = [[min(vadoseVerts[:,0]),min(vadoseVerts[:,1])],[max(vadoseVerts[:,0]),-30]]
aquiferRegion = pg.meshtools.createRectangle([0,0],[500,-30],marker=2)

# Substrate region is just a rectangle tacked onto the bottom of aquifer.
substrateRegion = pg.meshtools.createRectangle([0,-30],[500,-50], marker=3)

geom = pg.meshtools.mergePLC([vadoseRegion,aquiferRegion,substrateRegion])
pg.show(geom, boundaryMarker = 1)

mesh = pg.meshtools.createMesh(geom,quality = 33, area = 5, smooth=[1, 10])
pg.show(mesh, markers = True)

# Regions:
# Modelling Region == 1
# Inversion Region == 2
# Constraints == 3+

for boundary in mesh.boundaries():
    if boundary.center().x() == geom.xmin():
        boundary.setMarker(1)
    elif boundary.center().x() == geom.xmax():
        boundary.setMarker(2)
    else:
        boundary.setMarker(0)
        
fig, ax = plt.subplots()
pg.mplviewer.drawMeshBoundaries(ax, mesh, useColorMap=True)
    
pg.show(mesh, mesh.cellMarkers())

world = pg.meshtools.createParaMeshPLC()
#4) Create region file