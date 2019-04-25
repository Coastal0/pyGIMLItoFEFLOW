# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:30:07 2019
@author: 264401K
"""
import pygimli as pg

meshFID = "mesh\mesh.bms"
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
            
mesh.save(meshFID[:-4]+'_regions.bms')
pg.show(mesh, markers = True)
