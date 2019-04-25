# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:52:27 2019

@author: 264401k
"""

import pandas as pd
import numpy as np
from tkinter import filedialog, Tk

nElectrodes = 32
electrodeSpacingX = 0
electrodeSpacingY = 0
electrodeSpacingZ = -3
electrodeStartX = 0
electrodeStartY = 0
electrodeStartZ = -170

coords = np.zeros([nElectrodes,4]).astype(int)
coords[:,0] = np.arange(1,nElectrodes+1)
if electrodeSpacingX != 0:
    coords[:,1] = np.arange(electrodeStartX, 
          (electrodeStartX+electrodeSpacingX*nElectrodes), 
          electrodeSpacingX)
if electrodeSpacingY != 0:
    coords[:,2] = np.arange(electrodeStartY, 
          (electrodeStartY+electrodeSpacingY*nElectrodes), 
          electrodeSpacingY)
if electrodeSpacingZ != 0:
    coords[:,3] = np.arange(electrodeStartZ, 
          (electrodeStartZ+electrodeSpacingZ*nElectrodes), 
          electrodeSpacingZ)

root = Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
fName = filedialog.askopenfilename(title = "Select sequence CSV",
                filetypes = (("DAT file","*.csv"),("all files","*.*")))
sequence = pd.read_csv(fName)

fOut = (fName)[:-4]+".txt"
with open(fOut, "w") as output:
    output.write("# \t X \t Y \t Z \n")
    for lines in coords:
        [output.write(str(s) + "\t") for s in lines]
        output.write('\n')
    output.write("# \t A \t B \t M \t N \n")
    output.write(sequence.to_string(columns = sequence.columns[0:5], index = 0, header = 0, justify = 'left'))
