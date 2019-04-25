# -*- coding: utf-8 -*-
import re


ArrList = ['dd','gr','wa']
ModelList = ['005','010','015','020','025']
logIN = 'RunAll.log'
with open(logIN, 'r+', newline = '\r\n' ) as f:
    datLine = f.read()
    
log = datLine.splitlines()

# Split rows by /n and assign to structs if numeric.
for line in log[:3]:
    print(line)
    if '#' in line:
        print(line)
        tLine = line.split('/')
        
        # Get base FEFLOW 
        indx = [i for i, s in enumerate(tLine) if 'Stable' in s]
        Model = re.findall('\d+', tLine[indx[0]])[0]
            
        # Get Array 
        Arr = [x for x in ArrList if x in tLine]
        
        # Get spacing
        Spacing = tLine[[i for i, s in enumerate(tLine) if 'm' in s][-1]]
        Spacing = int(re.findall('\d+', Spacing)[0])
    
    