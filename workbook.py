# -*- coding: utf-8 -*-
"""
This contains most of the subroutines used for FEFLOW conversion.
These are designed to be called from a master script.
"""

def loadHull(hullID):
    """
    hullID is a directory pointing to the concave hull of the pointcloud
    e.g. r"G:\directory\boundsXY.mat"
    """
    import h5py
    import numpy as np

    # Extract the concave hull from MATLAB output.
    with h5py.File(hullID, 'r') as file:
        bounds = np.array(file['boundsXY'])
        nodes = np.array(file['nodesXY'])
        hull = bounds.T
        nodes = nodes.T
    print(hull.shape, 'coordinates found')

    # Remove duplicate coordinates.
    if (hull[0, :] == hull[-1::]).all():
        print('Duplicate coordinates found (start/end)')
        hull = hull[0:-1]
        print(hull.shape, 'coordinates remaining')

    # Round to 5 decimal places (avoids floatingpoint issues later)
#    hull = np.round(hull, decimals = 5)
    return hull


def loadNodes(nodesID):
    """
    nodesID is a directory pointing to the concave hull of the pointcloud
    e.g. r"G:\directory\boundsXY.mat"
    """
    import h5py
    import numpy as np
    import pygimli as pg
    with h5py.File(nodesID, 'r') as file:
        print(file.keys())
        hImport = np.array(file['nodesXY'])
        nodes = hImport.T
    print(nodes.shape, 'coordinates found')
    return nodes

def loadDataToDict(fName):
    import pandas as pd
    import numpy as np
    # Read in mass-concentration node data
    data = pd.read_table(fName, delim_whitespace=True)
    if 'node' in data.columns:
        maxNodes = max(data.Node)
    elif 'element' in data.columns:
        maxNodes = max(data.Element)
    else:
        maxNodes = data.iloc[:,3].max()
    print('Number of nodes found =', maxNodes)
    # Extract coordinates of mesh.
    if 'CENTER_X' in data.columns:
        data['X'] = data.CENTER_X
        data['Y'] = data.CENTER_Y
    if 'Time' in data.columns:
        print(pd.unique(data.Time).size, 'time steps found:', pd.unique(data.Time))
        coords = np.round(np.stack((data.X[data.Time == data.Time[0]].values, data.Y[data.Time == data.Time[0]].values), axis=1), decimals = 5)
        print(len(coords), 'X-Y locations found for time', data.Time[0])
#        maxNodes = max(data.Time == data.Time[0])
    else:
#        maxNodes = max(data.Node)
        coords = np.round(np.stack((data.X.values, data.Y.values), axis=1), decimals = 5)
        if maxNodes != coords.shape[0]:
            print('Number of reported nodes =', maxNodes)
            print('Number of nodes found =', coords.shape[0])
            print('Number of nodes does not match. (Inactive elements in FEFLOW?)')
        else:
            print(len(coords), 'X-Y locations found.')
    return data, coords

def loadData(fNames, convertTime = 0):
    dataDict = {}
    for i, f in enumerate(fNames):
        print('Loading', f)
        dataDict[i] = loadDataToDict(f)
#
#    # Check coordinates match, and combine data columns.
#    if len(fNames) > 1:
#        print('Multiple input files entered...')
#        if (dataDict[0][1] == dataDict[1][1]):
#            coords = dataDict[1][1]
#            # Make single dataframe entity using keywords from column headers.
#            for i in dataDict:
#                if 'MINIT' in dataDict[i][0].columns:
#                    data = dataDict[i][0]
#            for i in dataDict:
#                if 'SINIT' in dataDict[i][0].columns:
#                    data['SINIT'] = dataDict[i][0]['SINIT']
#        else:
#            print('Error! Coordinates in files do not match.')
#    else:
#        print('Single datafile supplied...')
#        data = dataDict[0][0]
#        coords = dataDict[0][1]

    # Convert time-column to date
#    if 'Time' in data.columns and convertTime == 0:
#        if len(pd.unique(data['Time'])) > 1:
#            print('Time data found, assuming [days] increment')
#            import datetime as dt
#            startDate = dt.date(1990, 1, 1)
#            dTime = []
#            for d in data['Time']:
#                dTime.append(startDate + dt.timedelta(days = 365 * d))
#            data['DateTime'] = dTime

    print('Loading finished.')
    return dataDict

#def fillPoly_old:
    """
    Takes a boundary polygon and fills with nodes, excluding hullnodes.
    """
    # Fill the empty polygon with nodes
    print('Filling polygon with nodes...')
#    roundedCoords = np.round(coords, 3)
#    roundedHull = np.round(hull, 3)
#    counter = 0
#
#    for node in tqdm(roundedCoords):
#        if any(np.all(roundedHull[:] == node, axis = 1)):
#            counter = counter + 1
#        else:
#            bPoly.createNode(node)
    
def fillPoly(dataDict, hull, quality = 0, showMesh = False):
    import pygimli as pg
    from matplotlib import path
    
    coords = dataDict[0][0][['X','Y']].values
    hull_ = path.Path(hull, closed = True)

    if any(hull_.contains_points(coords) == False):
        print('Some coordinates are outside of supplied hull...! \n',
              'Please check mesh for desired output.')
    bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
#    print('Creating nodes...')
#    for node in coords:
#        bPoly.createNode(node)
    print(bPoly)
    bPolyMesh = pg.meshtools.createMesh(bPoly, quality = 32, area = 0.5)
    if showMesh:
        pg.show(bPolyMesh)
    return bPolyMesh


def checkMesh(mesh):
    """
    Check if each cell of a mesh has a node.
    """
    # Check each cell/node has a value.
    i = 0
    for n in mesh.nodes():
        if len(n.cellSet()) == 0:
            print(n, n.pos(), " have no cells!")
            i = i+1
    print(str(i)+' nodes have no cells')


def getTopo(hull):
    """
    Takes the hull and finds top boundary values.
    """
    import numpy as np

    topo = hull[(hull[:, 1] >= -5)]
    topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
    _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
    tu = topo[idx]
    for i in enumerate(idc):
            if i[1] > 1:
                tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
    _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
    topo = tu[idx]
    return topo


def createArray(start, end, spacing, schemeName, topoArray, enlarge = 1):
    """
    Creates an ERT array and makes ertMesh based on array.
    """
    import numpy as np
    import pygimli as pg
    import pybert as pb

    print('Creating array...')
    sensor_firstX = start
    sensor_lastX = end
    sensor_dx = spacing

    sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
    sensor_z = np.interp(sensor_x, topoArray[:, 0], topoArray[:, 1])
    sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T

    if schemeName == 'dd' and enlarge == 1:
        print('Expanding array...')
        ertScheme = pb.createData(sensors, schemeName=schemeName, enlarge = 1)
    else:
        ertScheme = pb.createData(sensors, schemeName=schemeName, enlarge = 0)
        print('Not exapnding array...')
    ertScheme.save('ertScheme')
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
    meshERTName = 'meshERT_'+schemeName+'_'+str(spacing)+'m'
    meshERT.save(meshERTName)
    return ertScheme, meshERT



#%%
def makeInterpVector(data, bPolyMesh, t = None):
    import pygimli as pg
    import numpy as np
    import pygimli.meshtools as mt

    print('Interpolation data to mesh...')
    if "Time" in data and len(np.unique(data.Time)) > 1:
        if t is not None:
            if t in np.unique(data.Time):
                print('Using specified time value: ', t)
                dataCol = data['MINIT'][data['Time'] == t].values
                dMesh = pg.meshtools.createMesh(data[['X','Y']].values, quality = 0)
                dInterp = pg.interpolate(dMesh, dataCol, bPolyMesh.cellCenter())
                print('Done')
                return dInterp, None
            else:
                print('Value not found. No interpolation done...')
                return
        else:
            print('Multiple times found')
            times = np.zeros([int(len(data)/len(np.unique(data.Time))),len(np.unique(data.Time))])
            dMesh = pg.meshtools.createMesh(data[['X','Y']].values, quality = 0)
            dInterp = np.zeros([bPolyMesh.cellCount(),len(np.unique(data.Time))])
            for i, t in enumerate(np.unique(data.Time)):
                print("Converting time to data vector:", t)
                times[:,i] = data['MINIT'][data['Time'] == t].as_matrix()
                dInterp[:,i] = pg.interpolate(dMesh, times[:,i], bPolyMesh.cellCenter())
            print('Done')
            return dInterp, times
    else:
        dMesh = pg.meshtools.createMesh(data[['X','Y']].values, quality = 0)
        dInterp = pg.interpolate(dMesh, data.iloc[:,-1], bPolyMesh.cellCenter())
        dInterp = mt.fillEmptyToCellArray(bPolyMesh, dInterp)
        print('Done')
        return dInterp

def convertFluid(dInterp, bPolyMesh, meshERT, saveVec=0, k=0.7, m = 2, phi = 0.3, vadRes = 1000, subRes = 20):
    import numpy as np
    from pygimli.physics.petro import resistivityArchie as pgArch

    print('Converting fluid cond to formation cond...')
#    k = 0.7  # Linear conversion factor from TDS to EC
    sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
#    print('Fluid conductivity range: {0} to {1} mS/m'.format(min(1000*sigmaFluid), max(1000*sigmaFluid)))
    rFluid = 1/sigmaFluid
#    print('Fluid resistivity range: {min} to {max} Ohm.m', min(rFluid), max(rFluid), 'mS/m')

#    print(rFluid)
    print('Interpolating mesh values...')
    resBulk = pgArch(rFluid, porosity=phi, m=m, mesh=bPolyMesh, meshI=meshERT, fill=1)
    print('Formation Resistivity Range (Ohm.m): %s to ' % min(resBulk), max(resBulk))
#    print('No.# Values in fluid data',resBulk.shape[0])
#    print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
#    print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())

    # Apply background resistivity model
    for c in meshERT.cells():
#        if c.center()[1] > 0:
#            resBulk[c.id()] = vadRes # Resistivity of the vadose zone
#            if c.marker() == 2:
#                c.setMarker(3)
        if c.center()[1] < -30:
            resBulk[c.id()] = subRes # Resistivity of the substrate
            if c.marker() == 2:
                c.setMarker(4)
            
    for c in meshERT.cells():
        if c.marker() == 1 and c.center()[0] < 0 and c.center()[1] > -30:
            resBulk[c.id()] = 2 # Resistivity of the ocean-side forward modelling region.
#    print('Done.')

    if saveVec == 1:
        print('Saving...')
        resBulkName = 'resBulk_.vector'
        np.savetxt(resBulkName,resBulk)
    return(resBulk)


def simulate(meshERT, resBulk, ertScheme, fName):
    import pybert as pb
    import numpy as np

    ert = pb.ERTManager(debug=True)
    print('#############################')
    print('Forward Modelling...')

    # Set geometric factors to one, so that rhoa = r
#    ertScheme.set('k', pb.geometricFactor(ertScheme))
    ertScheme.set('k', np.ones(ertScheme.size()))

    simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme,
                                  noiseAbs=0.0,  noiseLevel = 0.01)
    simdata.set("r", simdata("rhoa"))

    # Calculate geometric factors for flat earth
    flat_earth_K = pb.geometricFactors(ertScheme)
    simdata.set("k", flat_earth_K)

    # Set output name
    dataName = fName[:-4]+'_data.ohm'
    simdata.save(dataName, "a b m n r err k")
    print('Done.')
    print(str('#############################'))
    #    pg.show(meshERT, resBulk)
    return simdata

def showDifference(dInterpDict, bPolyMesh):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pygimli as pg
    print('Plotting differences between two datasets.')
    if len(dInterpDict) == 2:
        diffVector = (dInterpDict[0] - dInterpDict[1])/dInterpDict[1] 
        norm = mpl.colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-max(abs(diffVector)), vmax=max(abs(diffVector)))
        fig, ax = plt.subplots()
        im = pg.mplviewer.drawMPLTri(ax, bPolyMesh, data=diffVector, cMin=None, cMax=None, cmap='RdBu_r', logScale=True, norm=norm)
        cbar = fig.colorbar(im, ax=ax, extend='neither', orientation = 'horizontal', aspect = 100, format = mpl.ticker.LogFormatter(linthresh = norm.linthresh))
        
        def logformat(cbar):
            new_labels = []
            for label in cbar.ax.xaxis.get_ticklabels():
                if len(label.get_text())>0:
                    print(label.get_text())
                    new_labels.append("%g" % float(label.get_text()))
                else:
                    new_labels.append("")
        
            cbar.ax.xaxis.set_ticklabels(new_labels)
        logformat(cbar)
    else:
        print('Incorrect vectors found to compare. Please select only two.')

def showModel(bPolyMesh, dInterp,fName):
    import pygimli as pg
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax, cb = pg.show(bPolyMesh, dInterp, colorBar=True, cmap='jet', showMesh=0, cMin = 50, cMax = 35000)
    fig = plt.gcf()
    fig.set_size_inches(17, 4)
    ax.minorticks_on()
#    ax.set_xlim(left=-20, right=600)
#    ax.set_ylim(top=35, bottom=-30)

    ax.set_title(fName,loc= 'left')
    ax.set_ylabel('Elevation (mASL)')
    ax.set_xlabel('Distance (m)')

    cb.ax.minorticks_on()
    cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
    cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
    cb.set_label('Mass Concentration [mg/L]')
    fig.set_tight_layout('tight')
    return fig, ax, cb

#%% Convert BERT to res2dinv General
def bert_to_res2d(dataIn = None):
    import numpy as np
    from tkinter import filedialog, Tk
    if dataIn is None:
        # Load Data
        Tk().withdraw()
        dataIN = filedialog.askopenfilename(filetypes = (("ohm file","*.ohm"),("all files","*.*")))
    print('Loading '+ dataIN)
    with open(dataIN, 'r+', newline = '\r\n' ) as f:
        datLine = f.read()
    datLine = datLine.splitlines()
    
    # Get Sensors
    x = []
    y = []
    z = []
    
    nSensors = int(datLine[0])
    print(str(nSensors) +' sensors found')
    for lines in np.arange(nSensors):
        xi = float(datLine[2+lines].split()[0])
        yi = float(datLine[2+lines].split()[1])
        zi = float(datLine[2+lines].split()[2])
        
        x = np.append(x,xi)
        y = np.append(y,yi)
        z = np.append(z,zi)
    
    # Get Data
    nData = int(datLine[nSensors+2])
    print(str(nData) +' data found')
    headers = datLine[nSensors+3].split()
    datMat = np.zeros((nData,len(headers)-1))
    
    a = []
    b = []
    m = []
    n = []
    r = []
    rhoa = []
    
    for line in datLine[(nSensors+4):]:
    #    print(line)
        datMatI = line.split()
        if len(datMatI) == len(headers)-1:
            ai = int(datMatI[0])
            a = np.append(a,ai)
            bi = int(datMatI[1])
            b = np.append(b,bi)
            mi = int(datMatI[2])
            m = np.append(m,mi)
            ni = int(datMatI[3])
            n = np.append(n,ni)
            if 'rhoa' in headers:
                rhoai = float(datMatI[headers.index('rhoa')-1])
                rhoa = np.append(rhoa,rhoai)
            if 'r' in headers:
                ri = float(datMatI[headers.index('r')-1])
                r = np.append(r,ri)
    sensDx = x[1]-x[0]
    
    # Assemble RES2DINV Structures
    with open(dataIN[:-4] + "_res2dinv.dat", "w") as dFile:
        dFile.write(dataIN + '\n')
        dFile.write(str(sensDx)+' \n')
        dFile.write('11 \n')
        dFile.write('0 \n')
        dFile.write('Type of measurement \n')
        dFile.write('1 \n')
        dFile.write((str(nData)+' \n'))
        dFile.write('2 \n')
        dFile.write('0 \n')
        for i in np.arange(nData):
            ax = x[int(a[i])-1]
            ay = y[int(a[i])-1]
            bx = x[int(b[i])-1]
            by = y[int(b[i])-1]
            mx = x[int(m[i])-1]
            my = y[int(m[i])-1]
            nx = x[int(n[i])-1]
            ny = y[int(n[i])-1]
            r_ = r[i]
            
            fmt = '{:d}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4E}\n'
            outStringFmt = fmt.format(4, ax, ay, bx, by, mx, my, nx, ny, r_)
            dFile.write(outStringFmt)
    print('Done')
    return None

# Formatting figures
def plotWells(topoArray, rLim = 400):
    import matplotlib.pyplot as plt
    from pandas import read_csv
    # Plot Wells
    plt.sca(plt.gcf().get_axes()[0])
    wellFin = r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat"
    WellCoords = read_csv(wellFin, sep = '\t')
    
    def getElev(array, value):
      val = array[abs(array[:,0] - value).argmin()][1]
      return val
    
    for well in WellCoords.iterrows():
        if well[1].X < rLim:
            plt.plot([well[1]['X'],well[1]['X']],
                     [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
            plt.annotate(s = well[1]['LABEL'], 
                         xy = [well[1]['X'],8+getElev(topoArray,well[1]['X'])],
                         ha = 'center', fontsize = 12)
        else:
            print('Well outside viewing area')
    plt.plot([min(topoArray[:,0]), max(topoArray[:,0])],[0,0], 'k--')
    return None

def formatActiveFig(topoArray, lLim = -50, rLim = 400, uLim = 50, bLim = -30, title = None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig = plt.gcf()
    fig.set_size_inches(12, 3.5)
    
    plt.sca(fig.get_axes()[0])
    plt.plot(topoArray[:, 0], topoArray[:, 1], 'k')
    ax = plt.gca()
    ax.tick_params(which = 'both', direction = 'in')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim(left=lLim, right=rLim)
    ax.set_ylim(top=uLim, bottom=bLim)
    ax.set_aspect(aspect=1)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (mASL)')
    if title is not None:
        ax.set_title((title), loc = 'left', fontsize = 12)
    fig.set_tight_layout('tight')
    return None
    
def plotPESTNodes():
    nodeFile = "K:\FePest_Nodes_Output#3.dat"
    PESTNodes = pd.read_csv(nodeFile, sep = '\t')
    plt.scatter(PESTNodes.X, PESTNodes.Y, marker = '+', c = 'white', s = 1)

def plotPressure():
    pressureFile = r"G:\PROJECTS\EXP_ABST -- Random Seawater fields\test.lin"
    pFile = np.loadtxt(pressureFile, skiprows = 1, comments="END")    
    plt.plot(pFile[:,0],pFile[:,1], 'k--')

def setAboveWT(mesh, dataVec, val):
    print('Nulling values above water table...')
    for c in mesh.cells():
        if c.center()[1] > 0:
            dataVec[c.id()] = val
    return dataVec

def fixZeroes(data, dataVec,val):
    print('Fixing any zeros with value: {}'.format(val))
    dataVec[dataVec == 0] = val
    return dataVec
