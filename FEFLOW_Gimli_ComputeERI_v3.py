# -*- coding: utf-8 -*-
import workbook as w
import matplotlib.pyplot as plt
from matplotlib import colors,ticker
import pandas as pd
import pygimli as pg
import pybert as pb
import pygimli.mplviewer as pgmpl
import numpy as np
from tkinter import filedialog, Tk

root = Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
fNames = filedialog.askopenfilenames(title = "Select FEFLOW Output (Mass/etc)",
                filetypes = (("DAT file","*.dat"),("all files","*.*")))
print(fNames)

dataDict = w.loadData(fNames)
fBounds = filedialog.askopenfilenames(title = "Select MATLAB boundary and nodes (as *.mat)",
                 filetypes = (("matlab files","*.mat"),("all files","*.*")))
#%%
hull = w.loadHull(fBounds[0])
bPolyMesh = w.fillPoly(dataDict, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    dInterpDict[d] = w.makeInterpVector(dataDict[d][0], bPolyMesh)  # Add data to the nodes
 
# %% ERT Simulations    
# Use standard arrays
useCustomArray = 0
if useCustomArray:
    sensor_firstX = 0
    sensor_lastX = 500
    sensor_dx = 10
    
    sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
    sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
    sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
    sensors_ = sensors
    
#    wa = pb.createData(sensors, schemeName='wa')

    dd_e = pb.createData(sensors, schemeName='dd', enlarge = 1)
    dd_ = pb.createData(sensors, schemeName='dd', enlarge = 0)
    gr = pb.createData(sensors, schemeName='gr')
    
    ertScheme = gr
#    ertScheme.add(dd_)
#    ertScheme.add(gr)
#    
#    ertScheme = data
#    sum(abs(ertScheme['k']) < 200)
    gf = pb.geometricFactors(ertScheme)
    ertScheme.set('k', gf)
    ertScheme.markInvalid(abs(ertScheme('k')) > 10000)
    ertScheme.removeInvalid()
    len(np.asarray(ertScheme['valid']))
    ertScheme.save('test')
    
    dInterpDict[0][dInterpDict[0] < 1] = 358

# Use field data
useFieldData = 1
if useFieldData:
    fieldData = pb.importData(r"G:\BERT\data\QuinnsRocks_\Field\Dipole.Dat")
    ertScheme = fieldData
    sensors_ = np.asarray(fieldData.sensors())
    sensors_[:,1] = np.interp(sensors_[:,0], topoArray[:, 0], topoArray[:, 1])
    ertScheme.setSensorPositions(sensors_)
    sensor_firstX = sensors_[0][0]
    sensor_dx = sensors_[1][0] - sensors_[0][0]

# Topography before (left-of) electrodes
topoPnts_x = np.arange(topoArray[0,0],sensor_firstX,sensor_dx)
topoPnts_z = np.interp(topoPnts_x, topoArray[:, 0], topoArray[:, 1])
topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
topoPnts = np.insert(sensors_[:,[0,1]],0,topoPnts_stack,0)

# Create ERT mesh (based on sensors)
print('Creating modelling mesh...')
meshERT = pg.meshtools.createParaMesh(topoPnts, quality=30.0,
                                      paraMaxCellSize=3.0, paraDepth=100.0,
                                      paraDX=0.05, boundaryMaxCellSize = 10000.0)
print('ERT Mesh = ', meshERT)
meshERTName = 'meshERT_'+str(sensor_dx)+'m'
#%%  Show Forward model
def showModel_(meshERT, resBulk, drawContours = True):
    nData = pg.cellDataToPointData(meshERT, resBulk)
    ax, cb = pg.show(meshERT, resBulk, cMap = 'jet_r', colorBar = True, cMin = 1, cMax = 1000)
    if drawContours:
        pg.mplviewer.drawField(ax, meshERT, nData, levels=[2,10,50], 
                           fillContour=False, colors = 'k', 
                           linewidths=1, alpha=1, linestyles = '-')
    fig = plt.gcf()
    fig.set_size_inches(15, 4)
    
    ax.tick_params(which = 'both', direction = 'in')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim(left=0, right=430)
    ax.set_ylim(top=40, bottom=-40)
    ax.set_aspect(aspect=1)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (mASL)')
    fig.set_tight_layout('tight')
    return fig
    

#%%
showModel = 1
for i,d in dInterpDict.items():
    print(i)
    resBulk = w.convertFluid(d, bPolyMesh, meshERT, k=0.6, m = 1.6, phi = 0.3, subRes = 20, vadRes = 1000)

    if showModel == 1:
        fig = showModel_(meshERT, resBulk)
        filename = fNames[i][:-4]+'_fwdmodel.png'
        fig.savefig(fname = filename, bbox_inches='tight', format = 'png', dpi = 600)

    print('Forward Modelling...')
    ert = pb.ERTManager()
    ertScheme.set('k', np.ones(ertScheme.size()))
    simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme, returnFields = False,
                       verbose = True, noiseAbs=1E-6,  noiseLevel = 0.05)
    simdata.set("r", simdata("rhoa"))
#     Set output name
#    dataName = fNames[i][:-4]+'_data.ohm'
    dataName = fNames[i][:-4]+'_'+str(sensor_dx)+'m_data_noise5pc_gr.ohm'

    simdata.save(dataName, "a b m n r err k")
    print('Done.')

#%%
w.bert_to_res2d()
os.startfile(os.path.dirname(fNames[0]))