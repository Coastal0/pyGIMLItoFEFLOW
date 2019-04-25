# -*- coding: utf-8 -*-
import workbook as w
import matplotlib.pyplot as plt
import pygimli as pg
import pybert as pb
import numpy as np
from tkinter import filedialog, Tk
from matplotlib import ticker

def showModel_(meshERT, resBulk, drawContours = False):
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
    ax.set_xlim(left=-50, right=500)
    ax.set_ylim(top=40, bottom=-50)
    ax.set_aspect(aspect=1)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (mASL)')
    fig.set_tight_layout('tight')
    return fig

def idx(arr, val):
    idx = np.where( np.abs(arr-val) == np.abs(arr-val).min())
    return idx[0][0]
#%% Get data
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
schemes = ['wa','gr','dd']
spacing = [5]
for s in schemes:
    for sp in spacing:
        sensor_firstX = 0
        sensor_lastX = 500
        sensor_dx = sp
        sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
        sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
        sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T

        ertScheme = pb.createData(sensors, schemeName=s)
        if 'dd' in s:
            dd_e = pb.createData(sensors, schemeName='dd', enlarge = 1)
            ertScheme.add(dd_e)
        gf = pb.geometricFactors(ertScheme)
        ertScheme.set('k', gf)
        ertScheme.markInvalid(abs(ertScheme('k')) > 10000)
        ertScheme.removeInvalid()
        len(np.asarray(ertScheme['valid']))

        topo = topoArray
        for s_ in sensors:
        #    print(s)
            topo[idx(topo, s_)] = s_
        topoPnts = topo
#        plt.plot(topo[:,0], topo[:,1])
#        plt.scatter(sensors[:,0],sensors[:,1])
        # Create ERT mesh (based on sensors)
        print('Creating modelling mesh...')
        meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33.0,
                                              paraMaxCellSize=1.0, paraDepth=100.0,
                                              paraDX=0.01, boundaryMaxCellSize = 0)
        print('ERT Mesh = ', meshERT)
        meshERTName = 'meshERT_'+str(sensor_dx)+'m'
#        pg.show(meshERT, showNodes = True)
        #%%
        resLim = 1000
        massLim = 35.8
        for i,d in dInterpDict.items():
            print(fNames[i])
            d[d < massLim] = massLim
            resBulk = w.convertFluid(d, bPolyMesh, meshERT, k=0.6, m = 1.6, phi = 0.3, subRes = 20, vadRes = 1000)
#            resBulk[resBulk > resLim] = resLim
            print('Forward Modelling: {}, {}m'.format(s,sp))
#            showModel_(meshERT, resBulk, drawContours = True)
            
            #%% 
            ert = pb.ERTManager()
            ertScheme.set('k', np.ones(ertScheme.size()))
            simdata = ert.simulate(mesh=meshERT, res=resBulk, 
                                   scheme=ertScheme, verbose = True, 
                                   noiseAbs=1E-5,  noiseLevel = 0.05)
            simdata.set("r", simdata("rhoa"))
            
            dataName = fNames[i][:-4]+'_'+str(sp)+'m_data_noise5pc_'+s+'.ohm'
            simdata.save(dataName, "a b m n r err k")
            print('Done.')
