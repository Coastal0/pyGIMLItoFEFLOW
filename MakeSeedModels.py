#!/usr/bin/env F:\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64\python

import pygimli as pg
import numpy as np
import glob
import os.path

#% 1) Load BERT-generated mesh
print('Loading mesh...')
invMesh = pg.load('mesh\meshParaDomain.bms')
print('meshParaDomain.bms has '+str(invMesh.cellCount())+' cells')

#% 2) Load pyGimli Forward Mesh+Values
#dPath = r'G:\BERT\data\New\DD\seed' # contains resBulk vector and fwd mesh
dPath=os.getcwd()
#rates = ['025', '020','015','010','005','CON','RES','SIM']
vectors = glob.glob("resBulk*.vector") 
print('Data Path = '+dPath)
#iPath = r'G:\BERT\data\Seed Model\Stable_005mday\5m\SeedConductive'
#print('Inversion Mesh Path = '+iPath)

for i in vectors:
    print('Loading '+i)

    fwdMeshPath = glob.glob("meshERT*.bms")
    fwdMesh = pg.load(fwdMeshPath[0])
    print('Forward model mesh cells:',fwdMesh.cellCount())
    fwdData = pg.load(i)
    print('Forward model vector size ',fwdData.size())
    
    #% 3) Interpolate #2 onto #1
    dSeed = pg.interpolate(fwdMesh, fwdData, invMesh.cellCenter(),verbose = True, fillValue = 1000)
    print('Interpolated mesh nodes:',dSeed.size())
#    pg.show(invMesh, dSeed)
    #% 4) Export values for BERT mesh as vector
    i.partition('_')[-1]
    np.savetxt(os.path.join('dSeed_'+i.partition('_')[-1]),dSeed)
    print('Saved as '+'dSeed_'+vector.partition('_')[-1])
    print('################################')
