# -*- coding: utf-8 -*-
class workbook:
    #%%
    def loadHull(hullID):
        import h5py
        import numpy as np
        import pygimli as pg

        # Extract the concave hull from MATLAB output.
        with h5py.File(hullID, 'r') as file:
            hImport = np.array(file['boundsXY'])
            hull = hImport.T
        print(hull.shape, 'coordinates found')
        
        # Remove duplicate coordinates.
        if (hull[0, :] == hull[-1::]).all():
            print('Duplicate coordinates found (start/end)')
            hull = hull[0:-1]
            print(hull.shape, 'coordinates remaining')
            
        # Round to 5 decimal places (avoids floatingpoint issues later)
        hull = np.round(hull, decimals = 5)
        
        # Create the exterior boundary of the FEFLOW mesh from the outer bounds.
        bPoly = pg.meshtools.createPolygon(verts=hull, isClosed=1)
        print('Boundary Polygon (bPoly):', bPoly)
        return hull, bPoly
    #%%
    def loadData(fName):
        import pandas as pd
        import numpy as np
    
        # Read in mass-concentration node data
        fName = fName
        data = pd.read_table(fName, delim_whitespace=True)
        print('Number of nodes found =', max(data.Node))
        
        # Extract coordinates of mesh.
        if 'Time' in data.columns:
            print(pd.unique(data.Time).size, 'time steps found:', pd.unique(data.Time))
            coords = np.round(np.stack((data.X[data.Time == data.Time[0]].values, data.Y[data.Time == data.Time[0]].values), axis=1), decimals = 5)
            print(len(coords), 'X-Y locations found for time', data.Time[0])
            maxNodes = max(data.Time == data.Time[0])
        else:
            maxNodes = max(data.Node)
            coords = np.round(np.stack((data.X.values, data.Y.values), axis=1), decimals = 5)
            if maxNodes != coords.shape[0]:
                print('Number of reported nodes =', maxNodes)
                print('Number of nodes found =', coords.shape[0])
                print('Number of nodes does not match. (Inactive elements in FEFLOW?)')
            else:
                print(len(coords), 'X-Y locations found.')
            return data, coords
    
    #%%
    def fillPoly(bPoly, coords):
        import numpy as np
        import pygimli as pg
    
        """
        Takes a boundary polygon and fills with nodes, excluding hullnodes.
        """
        # Fill the empty polygon with nodes
        print('Filling polygon with nodes...')
        nnodes = 0 # i.e. nodes skipped (must match #hull nodes)
        mnodes = 0 # i.e. node added to mesh (must = maxnodes - hullnodes)
        for i in enumerate(coords):
            if any(np.all(coords[i[0], :] == hull[:], axis=1)):
                mnodes += 1
            else:
                bPoly.createNode(i[1])
                nnodes += 1
        # Create and show the mesh (Warning; May crash)
        bPolyMesh = pg.meshtools.createMesh(poly=bPoly, quality=0)
        pg.show(bPolyMesh)
        return bPolyMesh
    
    #%%
    def checkMesh(mesh):
        """
        Check if each cell of a mesh has a node.
        """
        # Check each cell/node has a value.
        i=0
        for n in mesh.nodes():
            if  len(n.cellSet()) == 0:
                #print(n, n.pos(), " have no cells!")
                i=i+1
        print(str(i)+' nodes have no cells')
    
    #%%
    def getTopo(hull):
        """
        Takes the hull and finds top boundary values.
        """
        import numpy as np
    
        topo = hull[(hull[:, 1] >= 0)]
        topo = topo[np.lexsort((topo[:, 1], topo[:, 0]))]
        _, idx, idc = np.unique(topo[:, 0], return_index=1, return_counts=1)
        tu = topo[idx]
        for i in enumerate(idc):
                if i[1] > 1:
                    tu[i[0]] = np.max(topo[topo[i[0], 0] == topo[:, 0]], 0)
        _, idx, idc = np.unique(tu[:, 0], return_index=1, return_counts=1)
        topo = tu[idx]
        return topo
    #%% Setup array (surface)
    def createArray(start, end, spacing, schemeName, topo):
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
        sensor_z = np.interp(sensor_x, topo[:, 0], topo[:, 1])
        sensors = np.stack([sensor_x, np.around(sensor_z, 2)]).T
              
        ertScheme = pb.createData(sensors, schemeName=schemeName)
    
        # Topography before (left-of) electrodes
        topoPnts_x = np.arange(topo[0,0],sensor_firstX,sensor_dx)
        topoPnts_z = np.interp(topoPnts_x, topo[:, 0], topo[:, 1])
        topoPnts_stack = np.stack([topoPnts_x,np.around(topoPnts_z,2)]).T
        topoPnts = np.insert(sensors[:,[0,1]],0,topoPnts_stack,0)
        
            # Create ERT mesh (based on sensors)
        print('Creating modelling mesh...')
        meshERT = pg.meshtools.createParaMesh(topoPnts, quality=33,
                                              paraMaxCellSize=0.5, paraDepth=100,
                                              paraDX=0.01)
        print('ERT Mesh = ', meshERT)
        meshERTName = 'meshERT_test'
        meshERT.save(meshERTName)
        return ertScheme, meshERT
    
    #%% Data mesh
    def makeDataMesh(coords, show):
        import pygimli as pg
        import numpy as np
    
        dMesh = pg.meshtools.createMesh(np.ndarray.tolist(coords), quality = 0)
        if show == 1:
            pg.show(dMesh)
        print(dMesh)
        return (dMesh)
    #%% 
    def convertFluid(dInterp, bPolyMesh, meshERT):
        import numpy as np
        from pygimli.physics.petro import resistivityArchie as pgArch
        
        k = 0.612  # Linear conversion factor from TDS to EC
        sigmaFluid = dInterp / (k*10000)  # dInterp (mg/L) to fluid conductivity (S/m)
        print('Fluid conductivity range: ', min(1000*sigmaFluid), max(1000*sigmaFluid), 'mS/m')
        rFluid = 1/sigmaFluid
    #    print(rFluid)
        print('Interpolating mesh values...')
        resBulk = pgArch(rFluid, porosity=0.3, m=2, mesh=bPolyMesh,
                                          meshI=meshERT, fill=1)
        print('No.# Values in fluid data',resBulk.shape[0])
        print('No.#Cells in ERT Mesh: ',meshERT.cellCount())
        print('No.# Data == No.# Cells?', resBulk.shape[0] == meshERT.cellCount())
        # apply background resistivity model
        rho0 = np.zeros(meshERT.cellCount()) + 100.  # Set background resistivity
        
        for c in meshERT.cells():
            if c.center()[1] > 0:
                resBulk[c.id()] = 1000. # Resistivity of the vadose zone
            elif c.center()[1] < -30:
                resBulk[c.id()] = 20. # Resistivity of the substrate      
        resBulkName = 'resBulk_.vector'
        np.savetxt(resBulkName,resBulk)
        return(resBulk)
    
    #%%
    def simulate(meshERT, resBulk, ertScheme):
        import pybert as pb
        import numpy as np
        
        ert = pb.ERTManager(debug=True)
        print('#############################')
        print('Forward Modelling...')
        #    ertScheme.set('k', pb.geometricFactor(ertScheme))
        ertScheme.set('k', np.ones(ertScheme.size()))
        
        simdata = ert.simulate(mesh=meshERT, res=resBulk, scheme=ertScheme,
                                      noiseAbs=0.0,  noiseLevel = 0.01)
        simdata.set("r", simdata("rhoa"))
        dataName = 'data_.ohm'
        simdata.save(dataName, "a b m n r err")
        print('Done.')
        print(str('#############################'))
        #    pg.show(meshERT, resBulk)
        return simdata