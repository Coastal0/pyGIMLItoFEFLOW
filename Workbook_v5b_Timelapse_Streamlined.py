#%% Setup functions and Import packages
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
import pygimli as pg
import pygimli.physics.petro as petro
import pybert as pb
import os
import time
import shutil

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

#%%
os.chdir(r'G:\BERT\data\New folder\Timelapse3_Monthly')
#dList = ['Stable_025mday.dat',
#         'Stable_020mday.dat',
#         'Stable_015mday.dat',
#         'Stable_010mday.dat',
#         'Stable_005mday.dat']
dList = ['QR_IWSS_1998+_30day.dat'] # This might be called something else e.g. 'TimeLapse_biannual.dat'
spaList = [5]
arrList = ['gr']
for dFile in enumerate(dList):
    for s in spaList:
        for a in arrList:
            print(dFile)
            print(s)   
            print(a)
        #%% Read concave hull solution from matlab
            print('Reading hull...')
            with h5py.File('boundsXY.mat', 'r') as file:
            #        print(list(file.keys()))
                hImport = np.array(file['boundsXY'])
                hull = hImport.T
            print(hull.shape, 'coordinates found')
            # Check for duplicate start/end nodes (pyGimli doesn't handle duplicates).
            if (hull[0, :] == hull[-1::]).all():
                print('Duplicate coordinates found (start/end)')
                hull = hull[0:-1]
                print(hull.shape, 'coordinates remaining')
            
            # Round impossibly small numbers down to zero.    
            for i in hull:
                if i[1] != 0 and np.around(i[1],2) == 0:
                    i[1] = np.around(i[1],2)
            
            # Create the boundary of the FEFLOW mesh (i.e. pointcloud).
            bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
            print('Boundary Polygon (bPoly):', bPoly)
                
        #%% Load datafiles (output from FEFLOW)
            # Use Pandas to read data in as table.
            print('Reading data...')
            tic()
            data = pd.read_table(dFile[1], delim_whitespace=True)
            print(max(data.Node), 'nodes found.')
            toc()
            # Extract coordinates of mesh.
            if 'Time' in data.columns:
                print(pd.unique(data.Time).size, 'time steps found:', pd.unique(data.Time))
                coords = np.stack((data.X[data.Time == data.Time[0]].values,
                                   data.Y[data.Time == data.Time[0]].values), axis=1)
                print(len(coords), 'X-Y locations found for time', data.Time[0])
                maxNodes = max(data.Node[data.Time == data.Time[0]])
            else:
                maxNodes = max(data.Node)
                coords = np.stack((data.X.values, data.Y.values), axis=1)
                if maxNodes != coords.shape[0]:
                    print('Number of reported nodes =', maxNodes)
                    print('Number of nodes found =', coords.shape[0])
                    print('Number of nodes does not match. (Inactive elements in FEFLOW?)')
                else:
                    print(len(coords), 'X-Y locations found.')
            
            #%% Fill boundary with nodes.
            print('Filling polygon with nodes...')
            tic()
            # New option (snap boundary to nearest node then fill)
            nHull = []
            dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords), quality=0)
            pg.show(dMesh)
            for n in hull:
                j = dMesh.findNearestNode(n)
                node = np.asarray(dMesh.node(j).pos())
                nHull.append(node)
            nHull = np.stack(nHull)
            nHull = nHull[:,(0,1)]
            bPoly = pg.meshtools.createPolygon(verts=nHull, isClosed=1)
            pg.show(bPoly)
            plt.scatter(nHull[:,0],nHull[:,1])
            plt.scatter(coords[:,0],coords[:,1])
            
            nnodes = 0
            mnodes = 0
            for c in enumerate(coords):
                if any(np.all(coords[c[0], :] == nHull[:], axis=1)):
                    mnodes += 1
                else:
                    bPoly.createNode(c[1])
                    nnodes += 1
            print('Found', mnodes, 'boundary nodes.')
            print('Added', nnodes, 'nodes.')
            bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
            print(bPolyMesh)
            pg.show(bPolyMesh)
            
            # Original option (Add mesh after boundary - not flexible)
#            nnodes = 0
#            mnodes = 0
#            for i in enumerate(coords):
#                if any(np.all(coords[i[0], :] == hull[:], axis=1)):
#                    mnodes += 1
#                else:
#                    bPoly.createNode(i[1])
#                    nnodes += 1
#            print('Found', mnodes, 'boundary nodes.')
#            print('Added', nnodes, 'nodes.')
#            print('Total nodes = ', nnodes+hull.shape[0])
#            toc()
#            # Mesh the nodes.
#            print('Boundary Polygon (w/o Nodes):', bPoly)
#            print('Meshing...')
#            tic()
#            bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
#            toc()
#            print('Boundary Polygon (w/Nodes):', bPolyMesh)
#            pg.show(bPolyMesh)
            
            # Check that each node has an associated cell (i.e. check for corrupt mesh)   
            i=0
            for n in bPolyMesh.nodes():
                if  len(n.cellSet()) == 0:
                    #print(n, n.pos(), " have no cells!")
                    i=i+1
            print(str(i)+' nodes have no cells')
                    
            #%% Mesh the data.
            print('Meshing...') 
            
            times = pd.unique(data.Time)
            t = pd.unique(data.Time)[0]

            dInterpMatrix = []
            for t in times:
                print(t)
                dtVec = (data.MINIT[round(data.Time, 5) == round(t,5)]).values
                dInterp = pg.interpolate(dMesh, dtVec, bPolyMesh.cellCenter())
                print(dInterp)
                dInterpMatrix.append(np.asarray(dInterp))
            pg.show(bPolyMesh, dInterp)
            #%% Get topography
            topo = nHull[(nHull[:, 1] >= 0)]
            topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
            _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
            tu = topo[idx]
            for i in enumerate(idc):
                    if i[1] > 1:
                        tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
            _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
            topo = tu[idx]
            
            bad_nodes = 0
            good_nodes = 0
            for t in topo:
                for n in nHull:
                    if all(t == n):
                        print(t)
                        good_nodes = good_nodes+1
                    else:
                        bad_nodes = bad_nodes+1
            
            #%% Setup array (surface)
            print('Creating array...')
            sensor_firstX = 0
            sensor_lastX = 500
            sensor_dx = s
            
            sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
            sensor_z = np.interp(sensor_x, topo[:, 0], topo[:, 1])
            sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
   
            ertScheme = pb.createData(sensors, schemeName=a)
            for pos in ertScheme.sensorPositions():
                print(pos)
    
            schemeName = 'ertScheme_'+str(sensor_dx)+'m'
            ertScheme.save(schemeName)
        
            # Topography before (left-of) electrodes
            topoPnts_x = np.arange(topo[0,0],sensor_firstX,sensor_dx)
            topoPnts_z = np.interp(topoPnts_x, topo[:, 0], topo[:, 1])
            topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
            topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)
            #%%
            print('Creating modelling mesh...')
            tic()
            meshERT = pg.meshtools.createParaMesh(topoPnts, quality=32,
                                                  paraMaxCellSize=5, paraDepth=100,
                                                  paraDX=0.01)
            toc()
            print('ERT Mesh = ', meshERT)
            pg.show(meshERT)
            meshERTName = 'meshERT'+dFile[1][6:-4]
            meshERT.save(meshERTName)
        
            #%% Setup Modelling Mesh
            print('Conveting from fluid to formation resistivity...')
            # Check for negative numbers, set to arbitrary value.
            # (This will generally only happen for a bad feflow mesh.)
            dInterp = np.asarray(dInterp)
            if any(dInterp <= 1):
                dInterp[dInterp <= 1] = 100
            
            # Convert mass concnetration to water conductivity
        #    print(dInterp)
            k = 0.612  # Linear conversion factor from TDS to EC
            sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
            print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
            rFluid = 1/sigmaFluid
        #    print(rFluid)
            print('Interpolating mesh values...')
            print(bPolyMesh)
            print(len(rFluid))
            print(meshERT)
            tic()
            test = pg.interpolate(bPolyMesh, rFluid, meshERT.cellCenter())
            resBulk = petro.resistivityArchie(rFluid, porosity=0.3, m=2, mesh=bPolyMesh, meshI=meshERT, fill=1)
            toc()
            print('No.# Values in fluid data',resBulk.shape[0])
            print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
            print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())
            # apply background resistivity model
            rho0 = np.zeros(meshERT.cellCount()) + 100.  # Set background resistivity
            for c in meshERT.cells(): # I don't think this actually does anything
                if c.center()[1] < 0:
                    rho0[c.id()] = 100.
                elif c.center()[1] > 0:
                    rho0[c.id()] = 2000.
            # Apply non-aquifer resistivity
            for c in meshERT.cells():
                if c.center()[1] > 0:
                    resBulk[c.id()] = 1000. # Resistivity of the vadose zone
    #            elif c.center()[1] < -0 and c.center()[1] > -30:
    #                resBulk[c.id()] = 200
                elif c.center()[1] < -30:
                    resBulk[c.id()] = 20. # Resistivity of the substrate
            pg.show(meshERT,resBulk)
            pg.show(bPolyMesh,resBulk)
            resBulkName = 'resbulk'+dFile[1][6:-4]+'_'+a+'.vector'
            np.savetxt(resBulkName,resBulk)
                
            #%% Simulate data
            simFlag = 1
            if simFlag == 1:
                ert = pb.ERTManager(debug=True)
                print('#############################')
                print('Forward Modelling...')
                tic()
            #    ertScheme.set('k', pb.geometricFactor(ertScheme))
                ertScheme.set('k', np.ones(ertScheme.size()))
            
                data = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme,
                                              noiseAbs=0.0,  noiseLevel = 0.01)
                toc()
                data.set("r", data("rhoa"))
                dataName = 'data_'+a+'_'+str(sensor_dx)+'m_'+dFile[1][7:-8]+'.ohm'
                data.save(dataName, "a b m n r err k")
                print('Done.')
                print(str('#############################'))
            #%%  Filter simulated data (can do it here, or in BERT)
            filterFlag = 0
            if filterFlag == 1 and simFlag == 1:
                print('Filtering.')
                print('Simulated rhoa:', min(data('rhoa')), max(data('rhoa')))
                print(sum(data('rhoa') < 0), 'invalid points (-ve RhoA)')
                data.markInvalid(data('rhoa') < 0)
                data.removeInvalid()
                print('Filtered rhoa:', min(data('rhoa')), max(data('rhoa')))
                
                kFilter = 100000.0
                print('Simulated k:', min(data('k')), max(data('k')))
                print(sum(data('k') > kFilter) + sum(data('k') < -kFilter), 'invalid points (k > +/- ', kFilter,')')
                data.markInvalid(data('k') > 100000)
                data.markInvalid(data('k') < -100000)
                data.removeInvalid()
                print('Filtered k:', min(data('k')), max(data('k')))
                
                errFilter = 1e1
                print('Simulated err:', min(data('err')), max(data('err')))
                print(sum(data('err') > errFilter),'/',data('err').size(), 'invalid points (err > ', errFilter,')')
                data.markInvalid(data('err') > errFilter)
                data.removeInvalid()
                print('Filtered err:', min(data('err')), max(data('err')))
            
                print('Done.')
                dataName_f = dataName[:-4]+'_f'+dataName[-4:]
                data.save(dataName_f, "a b m n r err")
        
        #    datatitle = dFile[1][:-4]+'_'+str(spaList)+'m_fwdModel_dp_5pcNoise.dat'
        #    pb.data.showData(data)
        #    data.save(datatitle)
