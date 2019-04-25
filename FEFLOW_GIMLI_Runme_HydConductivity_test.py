"""
Export FEFLOW Hydraulic Conductivity as Centres.

Note that exporting from DAC will be fail due to poor FEFLOW logic (v7.1 2018), so export from FEM file.
Note also the 'centres' refers to the centre of each cell, so script requires FEM mesh details.

Required Files;
1. Concave hull boundars of FEM model [via MATLAB script]
2. Nodes of FEM model [via MATLAB script]
3. Hydrauic conductivity patch [FEFLOW export]


"""
import workbook as w
import matplotlib.pyplot as plt
import pandas as pd
import pygimli as pg
import numpy as np
from tqdm import tqdm
from matplotlib import ticker


#%% Load data and hull.
hull, bPoly = w.loadHull(r"K:\boundsXY.mat")
nodes = w.loadNodes(r"K:\nodesXY.mat")
topoArray = w.getTopo(hull) # Extract the topography from the concave hull

# FIn's
# "K:/FePest/QR_Quad_Fepest_NM3_output_mass_stable.fem_HydCondValues_asCentre.dat"


#fIn = r"K:/FePest/QR_Quad_Fepest_NM3_output_mass_stable.fem_HydCondValues_asCentre.dat"
fIn = "file:///K:/QR_Quad-Poro_Zone2_CondExport.dat"
fIn = r"K:\results\QR_Quad_FEPEST_ModifiedCond_asCentre.dat"
data = pd.read_table(fIn, delim_whitespace = True)

dataVals = data.iloc[:,-1].values
dataCoords = np.ndarray.tolist(data[['CENTER_X','CENTER_Y']].values)

roundedCoords = np.round(dataCoords,3)
roundedHull = np.round(hull,3)
roundedNodes = np.round(nodes,3)
#
#plt.scatter(roundedHull[:,0], roundedHull[:,1])
#plt.scatter(roundedCoords[:,0], roundedCoords[:,1])
#plt.scatter(nodes[:,0], nodes[:,1])

# Add nodes to empty polygon
counter= 0
for node in tqdm(roundedNodes):
    test = np.all(roundedHull[:] == node, axis = 1)
    if any(test):
        counter = counter + 1
    else:
        bPoly.createNode(node)
print(counter)
print(bPoly)
pg.show(bPoly)

# Mesh bPoly
bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
print(bPolyMesh)
pg.show(bPolyMesh)
w.checkMesh(bPolyMesh)

# Interp across meshes
dMesh = pg.meshtools.createMesh(dataCoords, quality = 0)
print(dMesh)
dInterp = pg.interpolate(srcMesh = dMesh, inVec = dataVals, destPos = bPolyMesh.cellCenter())
test = dInterp
test[test == 0] = 140

#%% Plot Figures
cmap = plt.cm.get_cmap('jet')
cmap.set_under("grey")
ax, cb = pg.show(bPolyMesh,dInterp, cmap = cmap, cMin = 400, cMax = 36000, colorBar = True, vmin = 360, logScale = True)
fig = plt.gcf()
fig.set_size_inches(17, 4)
fig.set_tight_layout('tight')

plt.sca(fig.get_axes()[0])
plt.plot(topoArray[:,0], topoArray[:,1], 'k')
ax = plt.gca()
ax.set_xlim(left=-20, right=500)
ax.set_ylim(top=45, bottom=-30)
ax.set_aspect(aspect=1)
ax.minorticks_on()
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (mASL)')
ax.set_title('Solute Distribution -- No High-K Structure',loc= 'left')
#ax.set_title('Hydraulic Conductivity inc. High-Permeability Structure', loc = 'left')
#ax.set_title('Hydraulic Conductivity -- FEPEST', loc = 'left')

#%% Plot Wells
plt.sca(fig.get_axes()[0])
WellCoords = pd.read_table(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat",  delim_whitespace = True)

def getElev(array, value):
  val = array[abs(array[:,0] - value).argmin()][1]
  return val

for well in WellCoords.iterrows():
#    print(well)
    plt.plot([well[1]['X'],well[1]['X']], [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
    plt.annotate(s = well[1]['LABEL'], xy = [well[1]['X'],5+getElev(topoArray,well[1]['X'])])
