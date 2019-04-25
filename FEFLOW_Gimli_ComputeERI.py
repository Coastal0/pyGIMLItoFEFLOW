# -*- coding: utf-8 -*-
import workbook as w
import matplotlib as mpl
import pandas as pd
import pygimli as pg
import pybert as pb
import numpy as np
from tkinter import filedialog

print("pyGIMLI:", pg.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", mpl.__version__)
print("Numpy:", np.__version__)

#root = Tk()
#root.wm_attributes('-topmost',1)
#root.withdraw()
fNames = filedialog.askopenfilenames(title = "Select FEFLOW Output (Mass/etc)",
                filetypes = (("DAT file","*.dat"),("all files","*.*")))
print(fNames)

dataDict, data, coords = w.loadData(fNames)

# %% Mesh and coordinate geometry
fBounds = filedialog.askopenfilenames(title = "Select MATLAB boundary and nodes (as *.mat)",
                 filetypes = (("matlab files","*.mat"),("all files","*.*")))
hull, bPoly, nodes = w.loadHull(fBounds[0])
bPolyMesh = w.fillPoly(bPoly, coords, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
dMesh = w.makeDataMesh(coords, 0)  # Make a mesh using datapoints

# %% Data geometry
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    data = dataDict[d][0]
    if 'SINIT' in data:
        print('Mass and Saturation present')
        data['dataCol'] = data['MINIT'] * data['SINIT']
    elif 'MINIT' in data and 'SINIT' not in data:
        print('Mass present')
        data['dataCol'] = data['MINIT']
    elif 'MINIT' or 'SINIT' not in data:
        print('Neither mass or solute present in data')
        data['dataCol'] = data.iloc[:,-1]
    
    dInterpDict[d], times = w.makeInterpVector(data, dMesh, bPolyMesh)  # Add data to the nodes
# %% ERT Simulations    
sensor_firstX = 0
sensor_lastX = 470
sensor_dx = 10

sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T

dd_e = pb.createData(sensors, schemeName='dd', enlarge = 1)
dd_ = pb.createData(sensors, schemeName='dd', enlarge = 0)
gr = pb.createData(sensors, schemeName='gr')

ertScheme = dd_e
[a,b,m,n] = [dd_[i] for i in ['a','b','m','n']]
abmn = np.vstack((a,b,m,n)).T
for i in abmn:
#    print(i)
    ertScheme.addFourPointData(*map(int, i))
    
[a,b,m,n] = [gr[i] for i in ['a','b','m','n']]
for i in abmn:
#    print(i)
    ertScheme.addFourPointData(*map(int, i))
        

gf = pb.geometricFactors(ertScheme)
ertScheme.set('k', gf)
ertScheme.markInvalid(abs(ertScheme('k')) > 10000)
ertScheme.removeInvalid()
ertScheme.save('test')

dInterpDict[0][dInterpDict[0] < 1] = 358

# Topography before (left-of) electrodes
topoPnts_x = np.arange(topoArray[0,0],sensor_firstX,sensor_dx)
topoPnts_z = np.interp(topoPnts_x, topoArray[:, 0], topoArray[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)

# Create ERT mesh (based on sensors)
print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=32,
                                      paraMaxCellSize=3, paraDepth=100,
                                      paraDX=0.01)
print('ERT Mesh = ', meshERT)
meshERTName = 'meshERT_'+str(sensor_dx)+'m'
meshERT.save(meshERTName)
resBulk = w.convertFluid(dInterpDict[0], bPolyMesh, meshERT, k=0.6, m = 1.8, phi = 0.3)

#ax, cb = pg.show(meshERT, dInterpDict[0], showMesh = True, colorBar = True, cMap = 'jet_r')
#ax, cb = pg.show(meshERT, resBulk, showMesh = True, colorBar = True, cMap = 'jet_r')
#ax.set_ylim(-40,50)
#ax.set_xlim(-20,400)
print('Forward Modelling...')
# ERI Stuff
ert = pb.ERTManager()
#invalids = 0
#for i,m in enumerate(ertScheme("m")):
#    if m < int(ertScheme("b")[i]):
#        invalids = invalids + 1
#        ertScheme.markInvalid(int(i))
#for i,n in enumerate(ertScheme("n")):
#    if n < int(ertScheme("m")[i]):
#        invalids = invalids + 1
#        ertScheme.markInvalid(int(i))
#print(invalids)
#ertScheme.save('testests', "a b m n valid")

# Set geometric factors to one, so that rhoa = r
#ertScheme.set('k', pb.geometricFactor(ertScheme))
#ertScheme.set('k', np.ones(ertScheme.size()))
#simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, returnFields = False,
#                       verbose = True, noiseAbs=0.0,  noiseLevel = 0.01)
ert = pb.Resistivity()
mesh = meshERT
rho = resBulk
scheme = ertScheme
potMat = ert.simulate(mesh, res=rho, scheme=scheme, returnFields=True)

# potMat is a potential matrix with a row per electrode and a column per mesh node
print(potMat.rows(), potMat.cols())
# You can turn it to numpy via pg.utils.gmat2numpy(potMat)
elec1 = 0 # choose two electrodes
elec2 = 1
pot = potMat.row(elec2) - potMat.row(elec1) # U_12 = U2 - U1
ax, cb = pg.show(meshERT, pot, colorBar = True)
ax.set_ylim(-40,50)
ax.set_xlim(-20,400)

simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, returnFields = True)
simdata.set("r", simdata("rhoa"))

# Calculate geometric factors for flat earth
#flat_earth_K = pb.geometricFactors(ertScheme)
#simdata.set("k", flat_earth_K)

# Set output name
dataName = fNames[0][:-4]+'_data.ohm'
simdata.save(dataName, "a b m n r err k")
print('Done.')
