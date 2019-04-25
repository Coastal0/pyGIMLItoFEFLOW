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
os.chdir(r'F:\results')
dList = ['QR_stable_025mday.dat',
         'QR_stable_020mday.dat',
         'QR_stable_015mday.dat',
         'QR_stable_010mday.dat',
         'QR_stable_005mday.dat']
#dList = ['MassConc_Tides_HighTide-910d.dat']
#dList = ['MassConc_Tides_LowTide-1100d.dat']

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
            # Round to 5 dp (avoid floating point issues)
            hull = np.round(hull, decimals = 5)
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
                maxNodes = max(data.Time == data.Time[0])
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
#            pg.show(dMesh)
            for n in hull:
                j = dMesh.findNearestNode(n)
                node = np.asarray(dMesh.node(j).pos())
                nHull.append(node)
            nHull = np.stack(nHull)
            nHull = nHull[:,(0,1)]
            bPoly = pg.meshtools.createPolygon(verts=nHull, isClosed=1)
#            pg.show(bPoly)
#            plt.scatter(nHull[:,0],nHull[:,1])
#            plt.scatter(coords[:,0],coords[:,1])
            
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
            
#            print('Filling polygon with nodes...')
#            tic()
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
            # pg.show(bPolyMesh)
            
            # Check that each node has an associated cell (i.e. check for corrupt mesh)            
            i=0
            for n in bPolyMesh.nodes():
                if  len(n.cellSet()) == 0:
                    #print(n, n.pos(), " have no cells!")
                    i=i+1
            print(str(i)+' nodes have no cells')
                    
            #%% Mesh the data.
            print('Meshing...') 
            tic()
            dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords))
            toc()
            print(dMesh)
            dInterp = pg.interpolate(dMesh, data.MINIT, bPolyMesh.cellCenter())
        
            # Get topography
            topo = hull[(hull[:, 1] >= 0)]
            topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
            _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
            tu = topo[idx]
            for i in enumerate(idc):
                    if i[1] > 1:
                        tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
            _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
            topo = tu[idx]
        ##############################################################################
            showMassModelFlag = 1
            if showMassModelFlag == 1:
#                ax, cb = pg.show(bPolyMesh, np.log10(dInterp), colorBar=True, cmap='jet', showMesh=0, cMin = np.log10(400), cMax = np.log10(35000))
                ax, cb = pg.show(bPolyMesh, dInterp, colorBar=True, cmap='jet', showMesh=0, cMin = 350, cMax = 35000)
                ax.minorticks_on()
                ax.set_xlim(left=200, right=260)
                ax.set_ylim(top=-15, bottom=-30)
                ax.set_title("Highest Annual Tide [Zoom]",loc='left')
                
#                majLocator = ticker.LogLocator()
#                cb.locator = majLocator
                cb.update_ticks()
                cb.ax.minorticks_on()
                cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
            
                cb.set_label('Mass Concentration (mg/L)')
                fig = plt.gcf()
                fig.tight_layout()
#                fig.savefig((dFile[1][0:-4]+' [Zoom]'),dpi=300)
            
            # Create ERT mesh (Based on topography)
        #    meshERT = pg.meshtools.createParaMesh(topo, quality=34,
        #                                          paraMaxCellSize=1, paraDepth=100,
        #                                          paraDX=0.33)
        #    
        #    print('ERT Mesh = ', meshERT)
        #    for n in meshERT.nodes():
        #        if  len(n.cellSet()) == 0:
        #            print(n, n.pos(), " have no cells!")
        #    meshERT.save('meshERT')
                
        #    for spaList in spaList:
        #    print(spaList)
        
            #%% Setup array (surface)
        # schemeName : str ['none']
        # Name of the configuration. If you provide an unknown scheme name, all
        # known schemes ['wa', 'wb', 'pp', 'pd', 'dd', 'slm', 'hw', 'gr'] are listed.
            print('Creating array...')
            sensor_firstX = 0
            sensor_lastX = 500
            sensor_dx = s
            
            sensor_x = np.arange(sensor_firstX, sensor_lastX+sensor_dx, sensor_dx)
    #        sensor_y = np.zeros(len(sensor_x))
            sensor_z = np.interp(sensor_x, topo[:, 0], topo[:, 1])
            sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
                  
            # Setup array (borehole)
            bhx = [] # Leave empty for none
            bhSensors = []
            bhSensors_z = []
            bhSensors_x = []
            nBoreholes = len(bhx)
            bz_EndDepth = -25
            bz_dz = -5
            
            for i in bhx:
                bz_startDepth = np.interp(i, topo[:, 0], topo[:, 1])
                bhSensors_z = np.append(bhSensors_z,np.arange(bz_startDepth+bz_dz, bz_EndDepth, bz_dz))
                bhSensors_x = np.append(bhSensors_x,i * np.ones((np.arange(bz_startDepth+bz_dz, bz_EndDepth, bz_dz).shape)))
                bhSensors = np.stack([bhSensors_x, bhSensors_z],1)
                
            if len(bhSensors) == 0:
                sensors = sensors
                print('No borehole sensors found.')
            else:
                print('Appending borehole sensors..')
                sensors = np.append(sensors,bhSensors,0)
        #   #Round sensor positions to nearest point on mesh (needs fine mesh)
        #    newSensors = np.zeros(sensors.shape)
        #    for pos in enumerate(sensors):
        #        j = meshERT.findNearestNode(pos[1])
        #        n = meshERT.node(j)
        #        mx = n.x()
        #        my = n.y()
        ##        print('Sensor:', pos[1],'Nearest Node: ',[mx,my])
        #        newSensors[pos[0]] = [mx,my]     
            # Create ertScheme    
        #    ertScheme = pb.createData(newSensors, schemeName='dd')
            ertScheme = pb.createData(sensors, schemeName=a)
            for pos in ertScheme.sensorPositions():
                print(pos)
    
        #    schemeName = 'ertScheme_'+str(spaList)+'m'
            schemeName = 'ertScheme_'+str(sensor_dx)+'m'
            ertScheme.save(schemeName)
        
            # Topography before (left-of) electrodes
            topoPnts_x = np.arange(topo[0,0],sensor_firstX,sensor_dx)
            topoPnts_z = np.interp(topoPnts_x, topo[:, 0], topo[:, 1])
            topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
            topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)
    #        yVals = np.zeros(len(sensors)+len(topoPnts_stack))
    #        topoPnts = np.insert(arr = topoPnts, obj = 1, values = yVals,axis =1)
        #    # Topography after (right-of) electrodes
        #    topoPnts_x = np.arange(sensor_lastX,np.around(topo[-1,0],-1),2*sensor_dx)
        #    topoPnts_z = np.interp(sens_points_x, topo[:, 0], topo[:, 1])
        #    topoPnts_stack = np.stack([sens_points_x,np.around(sens_points_z,2)]).T
        #    sensors = np.append(sensors,sens_points_stack,axis = 0)
            
            # Points at the water-table and substrate (Does not work currently)
        #    waterTable_x = np.arange(min(sensors[:,0]), max(sensors[:,0]))
        #    waterTable_z = np.zeros(waterTable_x.shape)
        #    waterTablePnts = np.stack([waterTable_x,waterTable_z]).T
        #    
        #    subStrate_x = np.arange(min(topoPnts[:,0]), max(topoPnts[:,0]))
        #    subStrate_z = -25*np.ones(subStrate_x.shape)
        #    subStratePnts = np.stack([subStrate_x,subStrate_z]).T
        
        #    meshPoints = np.vstack([topoPnts,waterTablePnts,subStratePnts])
                # Create ERT mesh (based on sensors)
            print('Creating modelling mesh...')
            tic()
            meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33,
                                                  paraMaxCellSize=3, paraDepth=100,
                                                  paraDX=0.33)
            toc()
            print('ERT Mesh = ', meshERT)
            for n in meshERT.nodes():
                if  len(n.cellSet()) == 0:
                    print(n, n.pos(), " have no cells!")
            meshERTName = 'meshERT'+dFile[1][6:-4]
            meshERT.save(meshERTName)
        
            #%% Setup Modelling Mesh
            print('Conveting from fluid to formation resistivity...')
            # Check for negative numbers, set to arbitrary value.
            # (This will generally only happen for a bad feflow mesh.)
            if any(dInterp <= 1):
                dInterp[dInterp <= 1] = 100
            
            # Downsample forwardmodels (e.g. for seed models)
##############################################################################
            downSample = 0
            if downSample == 1:
                print('Creating down-sampled (simple) seed model')
                dSampVec = np.asarray(dInterp)
                ranges = [10000,35000] # Starting water concentrations
                digIx = np.digitize(dInterp,ranges, right = False)
                ranges[0] = 500
                for dx in range(len(ranges)):
                    print(dx)
                    dSampVec[digIx == dx] = ranges[dx]
                dInterp = dSampVec
                
##############################################################################
                showdSampVecFlag = 0
                if showdSampVecFlag == 1:
                    ax, cb = pg.show(bPolyMesh, dSampVec, colorBar=True, cmap='jet', showMesh=0, cMin = 100, cMax = 35000)
                    ax.minorticks_on()
                    pg.mplviewer.drawSensors(ax, ertScheme.sensorPositions(),Facecolor = '0.75', edgeColor = 'k', diam = 1)
                    ax.set_xlim(left=-20, right=800)
                    ax.set_ylim(top=25, bottom=-40)
                    
                    majLocator = ticker.LogLocator()
                    cb.locator = majLocator
                    cb.update_ticks()
                    cb.ax.minorticks_on()
                    cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
                
                    cb.set_label('Mass Concentration (mg/L)')
                    fig = plt.gcf()
                    fig.tight_layout()
    
            #%% Convert mass concnetration to water conductivity
        #    print(dInterp)
            k = 0.612  # Linear conversion factor from TDS to EC
            sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
            print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
            rFluid = 1/sigmaFluid
        #    print(rFluid)
            print('Interpolating mesh values...')
            tic()
            resBulk = petro.resistivityArchie(rFluid, porosity=0.3, m=2, mesh=bPolyMesh,
                                              meshI=meshERT, fill=1)
            toc()
            print('No.# Values in fluid data',resBulk.shape[0])
            print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
            print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())
            # apply background resistivity model
            rho0 = np.zeros(meshERT.cellCount()) + 100.  # Set background resistivity
#            for c in meshERT.cells(): # I don't think this actually does anything
#                if c.center()[1] < 0:
#                    rho0[c.id()] = 100.
#                elif c.center()[1] > 0:
#                    rho0[c.id()] = 2000.
            # Apply non-aquifer resistivity
##############################################################################
            simpleLayers = 0
            if simpleLayers == 1:
                print('Creating simple layer cake starting model...')
                layerRes = [1,2,5,10,25,50,100,500]
                for res in layerRes:
                    print(res)
                    for c in meshERT.cells():
                        if c.center()[1] > 0:
                            resBulk[c.id()] = 1000. # Resistivity of the vadose zone
                        elif c.center()[1] < -0 and c.center()[1] > -30 and simpleLayers == 1:
                            resBulk[c.id()] = res
                        elif c.center()[1] < -30:
                            resBulk[c.id()] = 20. # Resistivity of the substrate
                    np.savetxt('resbulk'+dFile[1][6:-4]+'_'+a+'_simpleLayer_'+str(res)+'ohmm.vector',resBulk)

            if downSample == 1:
                np.savetxt('resbulk'+dFile[1][6:-4]+'_'+a+'_downsample_.vector',resBulk)
            else:
                resBulkName = 'resbulk'+dFile[1][6:-4]+'_'+a+'_'+str(s)+'.vector'
                np.savetxt(resBulkName,resBulk)
            # %% Show Model
##############################################################################
            showModelFlag = 0
            if showModelFlag == 1:
                ax, cb = pg.show(meshERT, resBulk, colorBar=True, cmap='jet_r', showMesh=0, cMin = 1, cMax = 1000)
                ax.minorticks_on()
                pg.mplviewer.drawSensors(ax, ertScheme.sensorPositions(),Facecolor = '0.75', edgeColor = 'k', diam = 1)
                ax.set_xlim(left=-20, right=500)
                ax.set_ylim(top=25, bottom=-40)
                ax.set_title(str(dList), loc=  'left')
                ax.set_ylabel('Elevation (mASL)')
                ax.set_xlabel('Distance (m)')
                
                majLocator = ticker.LogLocator()
                cb.locator = majLocator
                cb.update_ticks()
                cb.ax.minorticks_on()
                cb.ax.xaxis.set_ticklabels(cb.get_ticks().astype(int), minor = False)
            
                cb.set_label('Formation Resistivity ($\Omega$$\cdot$m)')
                fig = plt.gcf()
                fig.tight_layout()
                
            #%% Simulate data
##############################################################################
            simFlag = 0
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
                if downSample == 1:
                    dataName = 'data_'+a+'_'+str(sensor_dx)+'m_'+dFile[1][7:-8]+'downsample.ohm'
                else:
                    dataName = 'data_'+a+'_'+str(sensor_dx)+'m_'+dFile[1][7:-8]+'_'+str(s)+'_'+a+'.ohm'
                data.save(dataName, "a b m n r err")
                print('Done.')
                print(str('#############################'))
            #%%  Filter simulated data (can do it here, or in BERT)
##############################################################################
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
           
            #%% Move Files/Folders
##############################################################################
            movDirFlag = 0
            if movDirFlag == 1:
                currDir = os.getcwd()
    #            BERTdir = 'G:\\BERT\\data'
                dstDir = os.path.join(currDir,dFile[1][:-4],a,str(sensor_dx)+'m')
                if os.path.isdir(dstDir) == False:
                   os.makedirs(dstDir)
                
                with open('config_auto.cfg','w') as cfgOut:
                    cfgOut.write('DATAFILE='+dataName)
                shutil.copy('config_auto.cfg',dstDir)
                shutil.copy(dataName,dstDir)
                shutil.copy(schemeName,dstDir)
                shutil.copy(resBulkName,dstDir)
                shutil.copy(meshERTName+'.bms',dstDir)

        
        #    os.chdir(BERTdir)
    print('Done')
    #%% Inversion
    #data = ert.loadData('data_f.ohm')
    #ert.setVerbose(True)
    #pb.data.showData(data)
    #pb.show(data)
    #model = ert.invert(data = data, mesh=meshERT, lam=30)
    #model.save('model')
    #ert.showResultAndFit()
    
    #
    #model = ERT.invert(data = data_array, mesh = meshERT, verbose = 1)
    #ert.showResultAndFit()
    









