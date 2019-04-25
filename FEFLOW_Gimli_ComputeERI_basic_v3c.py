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
schemes = ['gr']
spacing = [5]
for s in schemes:
    for sp in spacing:
        # Create sensor array
        sensor_firstX = 0
        sensor_lastX = 500
        sensor_dx = sp
        sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
        sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
        sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
        
        # Create ERT scheme
        ertScheme = pb.createData(sensors, schemeName=s)
        if 'dd' in s:
            dd_e = pb.createData(sensors, schemeName='dd', enlarge = 1)
            ertScheme.add(dd_e)
            
        # Filter ERT scheme
        gf = pb.geometricFactors(ertScheme)
        ertScheme.set('k', gf)
        ertScheme.markInvalid(abs(ertScheme('k')) > 10000)
        ertScheme.removeInvalid()
        len(np.asarray(ertScheme['valid']))
        
        # Create topography/seafloor
        topo = topoArray[topoArray[:,0] > 0]
        topo = np.insert(topo,0,np.linspace([-100,0],[0,0],500), axis = 0)
        for s_ in sensors:
            print(s_)
            topo[idx(topo, s_)] = s_

        # Re-insert node at 0,0 for water table line
        topo = np.insert(topo, idx(topo[:,0],0), [0,0], 0)
#        topo = topo[topo[:,0].argsort()]
        
#        plt.plot(topo[:,0], topo[:,1])
#        plt.scatter(sensors[:,0],sensors[:,1])

#        plt.plot(topo[:,0], topo[:,1])
#        plt.scatter(sensors[:,0],sensors[:,1])
        # Create ERT mesh (based on sensors)
        print('Creating modelling mesh...')
#        fig, ax = plt.subplots(nrows = 3, ncols = 1, sharex = True, sharey = True)
#        areas = [1, 1/10, 1/100]
#        for n, p in enumerate(areas):
#            meshERT = pg.meshtools.createParaMesh(topo, quality=33, paraBoundary = 1.5,
#                                                  paraMaxCellSize= 1, paraDepth=100.0,
#                                                  paraDX=p, boundaryMaxCellSize = 0)
#            pg.show(meshERT, showNodes = True, ax = ax[n])
#        ax[0].set_xlim([-100,100])
#        ax[0].set_ylim([-50,30])
        meshERT = pg.meshtools.createParaMesh(topo, quality=33, paraBoundary = 1.5,
                                      paraMaxCellSize= 1, paraDepth=100.0,
                                      paraDX=1/50, boundaryMaxCellSize = 0)
        print('ERT Mesh = ', meshERT)
        meshERTName = 'meshERT_'+str(sensor_dx)+'m'
        #%%
        setOcean = 1
        resLim = 1000
        massLim = 35.8
        for i,d in dInterpDict.items():
            print(fNames[i])
            d[d < massLim] = massLim
            resBulk = w.convertFluid(d, bPolyMesh, meshERT, k=0.6, m = 1.6, phi = 0.3, subRes = 20, vadRes = 1000)
#            resBulk[resBulk > resLim] = resLim
            if setOcean:
                # Load seabed topography
#                seafloorFID = r"G:\BERT\data\Constraints\Regions\seafloor.txt"
#                seafloor = np.loadtxt(seafloorFID)
                seafloor = topoArray[topoArray[:,0] <= 0]
                if all(seafloor[-1] != [0,0]):
                    seafloor = np.vstack((seafloor,[0,0])) 
                newSeafloor_x = np.linspace(seafloor[0,0], seafloor[-1,0], len(seafloor)*10)
                newSeafloor_v = np.interp(newSeafloor_x, seafloor[:,0], seafloor[:,1])
                seafloor = np.stack((newSeafloor_x, newSeafloor_v), axis = 1)
                for c in meshERT.cells():
                    # Caclulate only relevent cells
                    if c.center().x() <= 0.0 and \
                        c.center().y() >= np.min(seafloor[:,1]):
                        # Extract x,y's
                        mshx = c.center().x()
                        mshy = c.center().y()
                        idx_ = idx(seafloor[:,0],mshx)
                        sfx = seafloor[idx_][0]
                        sfy = seafloor[idx_][1]
                        # If mesh-y is above seafloor, assign a value
                        if mshy >= sfy and mshy <= 0:
#                            print(c.marker())
                            c.setMarker(5)
                            resBulk[c.id()] = 0.2
#            showModel_(meshERT, resBulk, drawContours = True)
#        pg.show(meshERT,resBulk, colorBar = True, cmap = 'jet_r')
#        pg.show(meshERT, markers = True, showMesh = True)
        #%% 
        print('Forward Modelling: {}, {}m'.format(s,sp))
        ert = pb.ERTManager()
        ertScheme.set('k', np.ones(ertScheme.size()))
        simdata = ert.simulate(mesh=meshERT, res=resBulk, returnFields = True,
                               scheme=ertScheme, verbose = True, 
                               noiseAbs=1E-5,  noiseLevel = 0.05)
        simdata.set("r", simdata("rhoa"))
        dataName = fNames[i][:-4]+'_'+str(sp)+'m_data_noise5pc_noOcean_'+s+'.ohm'
#        simdata.save(dataName, "a b m n r err k")
        print('Done.')
        
#%% Show fields
potMat = simdata
import matplotlib
import matplotlib.colors as colors

print(potMat.rows(), potMat.cols())
# You can turn it to numpy via pg.utils.gmat2numpy(potMat)
elec1 = 1 # choose two electrodes
elec2 = 10
pot = (potMat.row(elec2) - potMat.row(elec1)) # U_12 = U2 - U1
norm=colors.SymLogNorm(linthresh=1, linscale=0.1, vmin=-50.0, vmax=200)

ax, cb = pg.show(meshERT, data=np.log10(abs(pot)), colorBar=True, cMap="RdBu_r",
          orientation='horizontal', label='log10(Vp)', vmin = -5, vmax = 3,
          nLevs = 21, showMesh=False)
cmap = plt.cm.get_cmap('RdBu_r')
cmap.set_over('k')
#ax, cb = pg.show(mesh = meshERT, data = pot, cMap=cmap, nLevs = 21,
#                 colorBar=True, orientation='horizontal', label='Vp', extend = 'both',
#                 fillContour = True, logScale = False, showMesh = False)

nData = pg.cellDataToPointData(meshERT, resBulk)
pg.mplviewer.drawField(ax, meshERT, nData, levels=[0.5,1.0,10,100.0,1000.0], 
                       fillContour=False, colors = 'grey', 
                       linewidths=1, alpha=1, linestyles = 'solid')
        
#ax_, _ = pg.show(meshERT, resBulk, fillContour = True, cMap="RdBu_r",
#                 nLevs = 21, colorBar = False, ax = ax)
#patches = [child for child in ax_.get_children() if isinstance(child, matplotlib.collections.PolyCollection)]
#patches[0].set_alpha(0.2)

# Plot lines (boundaries)
ax.plot(topo[:,0],topo[:,1], '-k') # Topography
if setOcean:
    ax.plot(seafloor[:,0],seafloor[:,1], '-k') # Seafloor
ax.plot([0,ax.get_xlim()[1]],[0,0], '-k') # Water table
ax.plot([ax.get_xlim()[0],ax.get_xlim()[1]],[-30,-30], '-k') # Substrate
ax.scatter(sensors[:,0],sensors[:,1], s=10,c='k') # Sensors

# Create mesh for streamlines
tmpLst = np.vstack((sensors,[sensors[:,0].max(),-50], [-100,-50], [-100, 0]))
tmpPly = pg.meshtools.createPolygon(tmpLst, isClosed = 1)
gridCoarse = pg.meshtools.createMesh(tmpPly, quality = 33, area = 80)
pg.mplviewer.drawStreams(ax, meshERT, pot, coarseMesh=gridCoarse, startStream = 1, color='Black')

# Figure formatting
fig = plt.gcf()
fig.set_size_inches(8, 5)
ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlim(left=-50, right=100)
ax.set_ylim(top=20, bottom=-50)
ax.set_aspect(aspect=1)
ax.minorticks_on()
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (mASL)')
fig.set_tight_layout('tight')
if setOcean:
    plt.gcf().savefig('fields_{}_{}_ocean.png'.format(elec1,elec2), bbox_inches='tight', format = 'png', dpi = 300)
else:
    plt.gcf().savefig('fields_{}_{}.png'.format(elec1,elec2), bbox_inches='tight', format = 'png', dpi = 300)

#%% Compare fields
#pot_ocean = pot
#pot_noOcean = pot
diff_pot = np.asarray((pot_ocean - pot_noOcean))
ax, cb = pg.show(meshERT, data=diff_pot, colorBar=False, cMap="PRGn",
          orientation='horizontal', label='Absolute Difference (Vp)', vmin = 0,
          nLevs = 21, showMesh=False)
lines = [child for child in ax.get_children() if isinstance(child, matplotlib.collections.LineCollection)]
[l.set_alpha(0.2) for l in lines]
fig = plt.gcf()
fig.set_size_inches(15, 10)
ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlim(left=-50, right=100)
ax.set_ylim(top=20, bottom=-50)
ax.set_aspect(aspect=1)
ax.minorticks_on()
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (mASL)')
fig.set_tight_layout('tight')
plt.gcf().savefig('diff_fields_zoom.png'.format(elec1,elec2), bbox_inches='tight', format = 'png', dpi = 300)
