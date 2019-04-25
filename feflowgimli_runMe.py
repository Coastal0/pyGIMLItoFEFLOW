# -*- coding: utf-8 -*-
import workbook as w
import os
import matplotlib.pyplot as plt

hull, bPoly = w.loadHull(r"G:\results\boundsXY.mat")
fName = 'F:/results/QR_stable_005mday.dat'

data, coords = w.loadData(fName)
bPolyMesh = w.fillPoly(bPoly, coords, hull)
w.checkMesh(bPolyMesh)
topoArray = w.getTopo(hull)
dMesh = w.makeDataMesh(coords, 0)
dInterp = w.makeInterpVector(data, dMesh, bPolyMesh)
ertScheme, meshERT = w.createArray(0, 600, 5, 'wa', topoArray)
resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)
simdata = w.simulate(meshERT, resBulk, ertScheme, fName)