# -*- coding: utf-8 -*-

from typing import NamedTuple

class ObsNode():
    C: float
    H: float

ID = []
C = []
H = []
P = []
S = []
MC = []
Vx = []
Vy = []


# Create structs for each obs node.
timeData = {};




#N1 = ObsNode(1,2,3,4,5)

# Read data line-by-line
with open('QR_Quad_Richards_IWSS_1998+.dac.dar', 'r+', newline = '\r\n' ) as f:
    darIN = f.read()
darIN = darIN.splitlines()

# Split rows by /n and assign to structs if numeric.
currentTime = -1;
for line in darIN:
    print(line)
    tLine = line.split()
    
    if any("TIME" in s for s in tLine):
        itx = tLine.index("TIME")
        currentTime = tLine[itx+2];
    if any([s for s in tLine if s.isdigit()]):
        try:
            
            test = [float(tLine) for tLine in tLine]
            ID=test[0]
            # Call a switch-type statement here to assign the parameters to an ID
            
            #find data associate with time
            if not currentTime in timeData.keys() :
                timeData[currentTime] = {}

            obsNodes = timeData[currentTime];
            
            #if the ID has not been found, create a null object
            if not ID in obsNodes.keys() :
                n = ObsNode();
                obsNodes[ID] = n;
            #retreive the node with that id    
            n = obsNodes[ID];
            n.C = test[1];
            n.H = test[2];

        except ValueError:
            continue
        
#%% Plot
