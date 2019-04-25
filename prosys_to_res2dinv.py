# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:03:12 2019

@author: 264401K

Convert from PROSYS2 Exported spreadsheet into RES2DINV formats
"""
import pandas as pd
import numpy as np
from tkinter import filedialog, Tk

# Select file
root = Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
fName = filedialog.askopenfilename(title = "Select file",
                filetypes = (("data file","*.dat"),("all files","*.*")))

print(fName)

# Load coordinate list (e.g. mapping of electrode number to subsurface coordinate)
unitSpacing = 3
keys = np.arange(1,33,1)
array = np.arange(370.24,277.24-unitSpacing,-unitSpacing).T
coordList = dict(zip(keys,array))

if 'csv' in fName:
    # Load spreadheet (exported from prosys)
    fID = fName
    rawData = pd.read_csv(fID)
    rawData = rawData.dropna(axis = 1)
    rawData.columns = rawData.columns.str.strip()
    rawData['r'] = rawData['Vp']/rawData['In']
    # TO-DO: Add check for which spacing has been used (i.e. empty Spa.*)
    rawData = rawData[['Date','Spa.1','Spa.2','Spa.3','Spa.4','Vp','In','r','Name']]
    rawData.columns = ['Date', 'Ca', 'Cb', 'Pa','Pb','Vp','In','r','Name']
    # Reduce coordinates to electrode numbers (e.g. 0-10-30-40 to 1-2-4-5)
    unitSpacing = 3
    rawData['Ca'] = rawData['Ca']/unitSpacing + 1
    rawData['Cb'] = rawData['Cb']/unitSpacing + 1
    rawData['Pa'] = rawData['Pa']/unitSpacing + 1
    rawData['Pb'] = rawData['Pb']/unitSpacing + 1
    
    # Load coordinate list (e.g. mapping of electrode number to subsurface coordinate)
    keys = np.arange(1,33,1)
    array = np.arange(370.24,277.24-unitSpacing,-unitSpacing).T
    coordList = dict(zip(keys,array))
    rawData['Ca'] = rawData['Ca'].map(coordList)
    rawData['Cb'] = rawData['Cb'].map(coordList)
    rawData['Pa'] = rawData['Pa'].map(coordList)
    rawData['Pb'] = rawData['Pb'].map(coordList)
    
    # Filter datasets (e.g. by time, name, quadrupoles)
    #import datetime
    #rawData['Date'] = pd.to_datetime(rawData['Date'], dayfirst = True)
    #rawData['tDelta'] =  pd.to_timedelta(rawData.Date)
    #rawData['tDelta'] = rawData['tDelta'] - rawData['tDelta'][0]
    #
    #rawData['tDelta2'] = np.NaN
    #for i in range(1,len(rawData)):
    #    if rawData['tDelta'][i] > (rawData['tDelta'][i-2]-rawData['tDelta'][i-1]):
    #        rawData['tDelta2'][i] = 1
    #    else:
    #        rawData['tDelta2'][i] = 0
elif 'dat' in fName:
    with open(fName) as file:
        dataIn = file.readlines()
    fTitle = dataIn[0]
    unitSpacing = int(float(dataIn[1].strip()))
    dType = int(dataIn[5])
    nData = int(dataIn[6])
    data = pd.DataFrame(dataIn[9:])
    data = data[0].str.split(expand = True)
    data.columns = ['N', 'Ca-x', 'Ca-z', 'Cb-x', 'Cb-z', 'Pa-x', 'Pa-z','Pb-x', 'Pb-z','r']
    columnsTitles = ['N', 'Ca-z', 'Ca-x', 'Cb-z', 'Cb-x', 'Pa-z', 'Pa-x','Pb-z', 'Pb-x','r']
    data=data.reindex(columns=columnsTitles)
    data.columns = ['N', 'Ca-x', 'Ca-z', 'Cb-x', 'Cb-z', 'Pa-x', 'Pa-z','Pb-x', 'Pb-z','r']
    data = data.apply(pd.to_numeric, errors='coerce')
    # Reduce coordinates to electrode numbers (e.g. 0-10-30-40 to 1-2-4-5)
    for i in range(2,10,2):
        data.iloc[:,i] = (data.iloc[:,i]/unitSpacing + 1).map(coordList)
    data = data.dropna()

dx = 20
x_s = (np.geomspace(1.,11.,dx)-1).reshape(dx,1)
z_s = np.zeros((len(x_s),1))
dummySurface = np.hstack((x_s,z_s))
x_ = np.ones((len(array),1))*int(max(dummySurface[:,0]))
z_ = np.sort(array).reshape(len(array),1)
dummyBorehole = np.hstack((x_,z_))
borehole = np.hstack((np.zeros((len(array),1)),z_))
# Export file in RES2DInv format (e.g...)
"""
titlename.bin	
3 # unit electrode spacing
13 # 13 for resistances, 12 for app rho (x-hole).
770 # number of points
2 #
0	
Surface Electrodes	# e.g. x-grid spacing
35
0.00	0.00
0.10	0.00
...
9.50	0.00
10.00	0.00
Number of boreholes	# Must be at least two
2	
Borehole 1 electrodes	
32	# nElectrodes
0	277.24  # x,z locations
0	280.24
0	283.24
...
0	364.24
0	367.24
0	370.24
Borehole 2 electrodes	
32	# nElectrodes
10	277.24 # x,z locations
10	280.24
10	283.24
...
10	364.24
10	367.24
10	370.24
Measured Data # (e.g. nPoints, Ca-x, Ca-z, Cb-x, Cb-z, Pa-x, Pa-z, Pb-x, Pb-z, measuredValue)
4	0.00	277.24	0.00	280.24	0.00	283.24	0.00	286.24	-0.02388651
4	0.00	277.24	0.00	280.24	0.00	286.24	0.00	289.24	-0.004904266
4	0.00	277.24	0.00	280.24	0.00	289.24	0.00	292.24	-0.002277527
"""
with open(fName[:-4] + "_res2dinv_borehole.dat", "w") as dFile:
    dFile.write(fName + '\n')
    dFile.write(str(unitSpacing)+' \n')
    dFile.write('13 \n')
    dFile.write(str(len(data))+' \n')
    dFile.write('2 \n')
    dFile.write('0 \n')
    dFile.write('Surface Electrodes \n')
    dFile.write(str(len(dummySurface))+' \n')
    np.savetxt(dFile, dummySurface, delimiter = " ",fmt='%.3f')
    dFile.write('Numbr of Boreholes \n')
    dFile.write(str(2) + '\n')
    dFile.write('Borehole 1 electrodes \n')
    dFile.write(str(len(array)) + '\n')
    np.savetxt(dFile, borehole, delimiter = " ",fmt='%.3f')
    dFile.write('Borehole 2 electrodes \n')
    dFile.write(str(len(array)) + '\n')
    np.savetxt(dFile, dummyBorehole, delimiter = " ",fmt='%.3f')   
    dFile.write('Measured Data \n')
#    fmt = ("%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f","%.6f")

    np.savetxt(dFile, data.values, delimiter = " ",fmt='%.6f')
