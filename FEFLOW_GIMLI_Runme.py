# -*- coding: utf-8 -*-
import workbook as w
import os
import matplotlib.pyplot as plt
import pandas as pd
import pygimli as pg

# %% Load data and hull.
hull, bPoly = w.loadHull(r"G:/BERT/data/boundsXY.mat")

fNames = ["file:///K:/Porosity/QR_Quad_Richars_0100mday_poro_zone2_SAT.dat",
"file:///K:/Porosity/QR_Quad_Richars_0100mday_poro_zone2_START-MASS.dat"
]

# Assign fNames to a dictionary to store dataframe and coords.
dataDict = {}
for i, f in enumerate(fNames):
    print('Loading', f)
    dataDict[i] = w.loadData(f)

# Check coordinates match, and combine data columns.
if len(fNames) > 1:
    if (dataDict[0][1] == dataDict[1][1]).all():
        coords = dataDict[1][1]
        # Make single dataframe entity using keywords from column headers.
        for i in dataDict:
            if 'MINIT' in dataDict[i][0].columns:
                data = dataDict[i][0]
        for i in dataDict:
            if 'SINIT' in dataDict[i][0].columns:
                data['SINIT'] = dataDict[i][0]['SINIT']
    else:
        print('Error: Coordinates do not match')

# Convert time-column to date
if 'Time' in data:
    print('Time data found, assuming [days] increment')
    import datetime as dt
    startDate = dt.date(1990, 1, 1)
    dTime = []
    for d in data['Time']:
        dTime.append(startDate + dt.timedelta(days = 365 * d))
    data['DateTime'] = dTime

#%% This is the main body of the workbook
#%% Mesh and coordinate geometry
bPolyMesh = w.fillPoly(bPoly, coords, hull) # Fill boundary mesh with nodes at coordinates
w.checkMesh(bPolyMesh) # Check each node has a value. If !0, ERT will not work.
topoArray = w.getTopo(hull) # Extract the topography from the concave hull

dMesh = w.makeDataMesh(coords, 0) # Make a mesh using datapoints

#%% Data geometry
data['dataCol'] = data['MINIT'] * data['SINIT']
# Non-Timelapse
dInterp = w.makeInterpVector(data, dMesh, bPolyMesh) # Add data to the nodes

# %% Show Model
fig, ax, cb = pg.show(bPolyMesh, dInterp, colorBar=True, cmap='jet', showMesh=0, cMin = 50, cMax = 35000)
fig = plt.gcf()
cb = fig.get_axes()[1]
fig.set_size_inches(17, 4)
ax.minorticks_on()
ax.set_xlim(left=-20, right=500)
ax.set_ylim(top=35, bottom=-30)

ax.set_title(fName,loc= 'left')
ax.set_ylabel('Elevation (mASL)')
ax.set_xlabel('Distance (m)')

cb.minorticks_on()
cb.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
cb.set_label('Mass Concentration [mg/L]')
fig.set_tight_layout('tight')

# Timelapse
#import pybert as pb
#import numpy as np
#ert = pb.ERTManager(debug=True)
#
#for t in pd.unique(data['Time'])[1:]:
#    print(t)
#    dInterp = w.makeInterpVector(data, dMesh, bPolyMesh, t) # Add data to the nodes
#    resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)
#    print('Forward Modelling...')
#    ertScheme.set('k', np.ones(ertScheme.size()))
#    simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, noiseAbs=0.0,  noiseLevel = 0.05)
#    simdata.set("r", simdata("rhoa"))
#    dataName = 'MitigationPumping_time'+str(int(t))+'_data.ohm'
#    simdata.save(dataName, "a b m n r err k")
#    print('Done.')


#%% ERI Stuff
ertScheme, meshERT = w.createArray(0, 710, 10, 'dd', topoArray, enlarge = 1)
invalids = 0
for i,m in enumerate(ertScheme("m")):
    if m < int(ertScheme("b")[i]):
        invalids = invalids + 1
        ertScheme.markInvalid(int(i))

for i,n in enumerate(ertScheme("n")):
    if n < int(ertScheme("m")[i]):
        invalids = invalids + 1
        ertScheme.markInvalid(int(i))
print(invalids)
ertScheme.save('testests', "a b m n valid")

resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)
dataName = 'QR_Quad_030mday_dd_filters'
simdata = w.simulate(meshERT, resBulk, ertScheme, dataName)


#%% Loop

fList = ['F:/results/QR_stable_015mday.dat',
        'F:/results/QR_stable_020mday.dat',
        'F:/results/QR_stable_025mday.dat',
        'F:/results/QR_stable_030mday.dat']

for fName in fList:
    resetVars()
    hull, bPoly = w.loadHull(r"G:\results\boundsXY.mat")
    data, coords = w.loadData(fName)
    bPolyMesh = w.fillPoly(bPoly, coords, hull)
    w.checkMesh(bPolyMesh)
    topoArray = w.getTopo(hull)
    dMesh = w.makeDataMesh(coords, 0)
    dInterp = w.makeInterpVector(data, dMesh, bPolyMesh)
    ertScheme, meshERT = w.createArray(0, 250, 5, 'wa', topoArray)
    resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)
    simdata = w.simulate(meshERT, resBulk, ertScheme, fName)

def resetVars():
    hull = []
    bPoly = []
    data = []
    coords = []
    bPolyMesh = []
    topoArray = []
    dMesh = []
    dInterp = []
    ertScheme = []
    meshERT = []
    resBulk = []

#%% Show models, etc.
def showModel(bPolyMesh, dInterp,fName):
    import pygimli as pg
    import numpy as np
    ax, cb = pg.show(bPolyMesh, dInterp, colorBar=True, cmap='jet', showMesh=0, cMin = 50, cMax = 35000)
    fig = plt.gcf()
#    fig.set_size_inches(14,3)
    fig.set_size_inches(17, 4)
    #ax, cb = pg.show(meshERT, resBulk, colorBar=True, cmap='jet_r', showMesh=0, cMin = 1, cMax = 1000)
    ax.minorticks_on()
    ax.set_xlim(left=-20, right=600)
    ax.set_ylim(top=35, bottom=-30)

#    tList=['Through-flow = 0.548 ML/year (-0.05 m/day)',
#            'Through-flow = 1.096 ML/year (-0.10 m/day)',
#            'Through-flow = 1.643 ML/year (-0.15 m/day)',
#            'Through-flow = 2.191 ML/year (-0.20 m/day)',
#            'Through-flow = 2.739 ML/year (-0.25 m/day)',
#            'Through-flow = 3.273 ML/year (-0.30 m/day)']

#    ax.set_title(tList[count], loc= 'left')
#    ax.set_title(os.path.split(fList[0])[-1], loc=  'left')
    ax.set_title(fName,loc= 'left')
    ax.set_ylabel('Elevation (mASL)')
    ax.set_xlabel('Distance (m)')
#
    cb.ax.minorticks_on()
    cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
    cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
    cb.set_label('Mass Concentration [mg/L]')
    fig.set_tight_layout('tight')
    return ax, cb

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygimli as pg
import numpy as np

def checkVecMesh(eriModelVec,eriModelMesh):
    print('Checking mesh and vector cells...')
    if eriModelVec.size() == eriModelMesh.cellCount():
        print("#Cells: " + str(eriModelMesh.cellCount()) +" = #Data: " + str(eriModelVec.size()))
    else:
        print("ERROR, CELLS DO NOT MATCH #VECTORVALUES")

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

#%% Load either field or synth to view (comment one out if needed)
# Field Data
vecName = r"G:\BERT\data\DD_5m_cases\010mday\model_15.vector"
eriModelVec = pg.load(vecName)
meshDir = os.path.split(vecName)[0]+"\mesh\meshParaDomain.bms"
eriModelMesh = pg.load(meshDir)
checkVecMesh(eriModelVec,eriModelMesh)

# Synth Data Inv Model
vecName = r"G:\BERT\data\Wenner\010\model_12.vector"
eriModelVec = pg.load(vecName)
meshDir = os.path.split(vecName)[0]+"\mesh\meshParaDomain.bms"
eriModelMesh = pg.load(meshDir)
checkVecMesh(eriModelVec,eriModelMesh)

nCols = 10
cmap = plt.get_cmap('jet_r')
jet_disc = cmap_discretize('jet', nCols)
jet_r_disc = cmap_discretize('jet_r', nCols)

# Standard plotting (limited control)
## NOTE: PLOTS PERFECTLY WITH 8 COLORS ONLY!!!
#ax2, cb2  = pg.show(eriModelMesh,eriModelVec,colorBar=True, cmap=jet_r_disc, showMesh=1, cMin = 1, cMax = 1000)

# Better control over colorbars
vals = np.geomspace(1,1000,nCols)
norm = mpl.colors.BoundaryNorm(vals, cmap.N)
ax = plt.axes()
im = pg.mplviewer.drawModel(ax = ax, mesh = eriModelMesh, data= eriModelVec, cmap = jet_r_disc, showMesh = 1, norm = norm)
ax.set_xlim(left=0, right=365)
ax.set_ylim(top=20, bottom=-60)
#ax.set_title('Field Data - Quinns Rocks 2015', loc=  'left')
#ax.set_title(r'Formation Resistivity Inverse Model (Wenner- $\alpha$)', loc=  'left')
ax.set_title(r'Formation Resistivity Inverse Model (Dipole-dipole)', loc=  'left')
ax.set_ylabel('Elevation (mASL)')
ax.set_xlabel('Distance (m)')

cb = plt.colorbar(ax = ax, mappable = im, orientation = 'horizontal', aspect = 50, spacing = 'uniform', ticks = vals, norm = norm, boundaries = vals, format = '%i')
cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')

fig = plt.gcf()
fig.set_size_inches(10,4)
fig.tight_layout()
fig.savefig('PNGout.png',dpi=300)
fig.savefig('SVGout.svg')


# Synth Data Fwd Model
vals = np.geomspace(1,1000,nCols)
norm = mpl.colors.BoundaryNorm(vals, cmap.N)
ax = plt.axes()
im = pg.mplviewer.drawModel(ax = ax, mesh = meshERT, data= resBulk, cmap = jet_r_disc, showMesh = 0, norm = norm)
ax.set_xlim(left=-20, right=250)
ax.set_ylim(top=40, bottom=-50)
ax.set_title('Formation Resistivity Forward Model - 1.09 ML/year (0.10 m/d)', loc=  'left')
ax.set_ylabel('Elevation (mASL)')
ax.set_xlabel('Distance (m)')

cb = plt.colorbar(ax = ax, mappable = im, orientation = 'horizontal', aspect = 50, spacing = 'uniform', ticks = vals, norm = norm, boundaries = vals, format = '%i')
cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')

fig = plt.gcf()
fig.set_size_inches(10,4)
fig.tight_layout()
fig.savefig('PNGout.png',dpi=300)



#%% Testing to extract the quadripoles from the field data for inversion
import pygimli as pg

fData = pg.load("G:\BERT\data\Field\Dipole\dp.bin.data")
fData("a").array()
ertScheme("a").array()

testScheme = fData
testScheme.setSensorPositions(ertScheme.sensorPositions())
testScheme.set('k', np.ones(testScheme.size()))
testScheme.save('testScheme', "a b m n")
testScheme = pg.load('testScheme')

import pybert as pb
ert = pb.ERTManager(debug=True)
testScheme.set('k', np.ones(testScheme.size()))

simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=testScheme, noiseAbs=0.0,  noiseLevel = 0.01)
simdata.set("r", simdata("rhoa"))
dataName = 'data_.ohm'
simdata.save(dataName, "a b m n r err k")
#    pg.show(meshERT, resBulk)


#%% Subroutine to make all possible electrode combinations
# From Florian
import numpy as np
import pybert as pb
import itertools

def abmn(n):
    """
    Construct all possible four-point configurations for a given
    number of sensors after Noel and Xu (1991).
    """
    combs = np.array(list(itertools.combinations(list(range(1, n+1)), 4)))
    perms = np.empty((int(n*(n-3)*(n-2)*(n-1)/8), 4), 'int')
    print(("Comprehensive data set: %d configurations." % len(perms)))
    for i in range(np.size(combs, 0)):
        perms[0+i*3, :] = combs[i,:] # ABMN
        perms[1+i*3, :] = (combs[i, 0], combs[i, 2], combs[i, 3], combs[i, 1]) #AMNB
        perms[2+i*3, :] = (combs[i, 0], combs[i, 2], combs[i, 1], combs[i, 3]) #AMBN

    return perms - 1

# Create empty DataContainer
dataComp = pb.DataContainerERT()

# Add electrodes
start = 0
end = 100
spacing = 5

print('Creating array...')
sensor_firstX = start
sensor_lastX = end
sensor_dx = spacing

sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
topoPnts_x = np.arange(topoArray[0,0],sensor_firstX,sensor_dx)
topoPnts_z = np.interp(topoPnts_x, topoArray[:, 0], topoArray[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)
print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33,
                                      paraMaxCellSize=0.5, paraDepth=100,
                                      paraDX=0.01)
print('ERT Mesh = ', meshERT)

for s in sensors:
    print(s)
    dataComp.createSensor(s) # 2D, no topography

# Add configurations
cfgs = abmn(dataComp.sensorCount()) # create all possible 4P cgfs for n electrodes
for i, cfg in enumerate(cfgs):
    dataComp.createFourPointData(i, *map(int, cfg)) # (We have to look into this: Mapping of int necessary since he doesn't like np.int64?)

resBulk = w.convertFluid(dInterp, bPolyMesh, meshERT)
dataName = 'QR_Quad_030mday_comprehensive'
simdata = w.simulate(meshERT, resBulk, dataComp, dataName)



# Optional: Save in unified data format for use with command line apps
dataComp.save("dataComp.shm", "a b m n")
