# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:36:10 2018

@author: 264401k

Suppy a directory and script will output figures, collabortaed data, etc.

"""

import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

tk.Tk().withdraw()
fNames = tk.filedialog.askopenfilenames()
print(fNames)

print('Reading data  (*.xlsx)...')
data = pd.read_excel(fNames[0], header = 0, index_col = None, verbose = True)

wells = {
        61611702: {'name':'SIM2', 'dist':30, 'elev':6.13}, 
        61611701: {'name':'SIM1', 'dist':105, 'elev':10.93}, 
        61611703: {'name':'SIM3', 'dist':190, 'elev':15.05}, 
        61611706: {'name':'SIM6', 'dist':360, 'elev':23.974}, 
        61611704: {'name':'SIM4', 'dist':550, 'elev':31.25}, 
#        61611707: {'name':'SIM9', 'dist': 110, 'elev':999}
        }

data = data[data['Site Ref'].isin([*wells])] # Filter by well ID

cond_cols = [col for col in list(data.columns) if 'uS/cm' in col] # Get columns with conductivity values
data['Conductivity_Combined'] = data.loc[:,cond_cols].sum(axis = 1, skipna = True, min_count = 1).dropna(how = 'all', axis = 0)

depth_cols = [col for col in list(data.columns) if 'depth' in col or 'Depths' in col] # Get columns with conductivity values

test = data.loc[:,depth_cols].sum(axis = 1, skipna = True, min_count = 1).dropna(how = 'all', axis = 0)
wellElevs = pd.Series([wells[el]["elev"] for el in data['Site Ref']])

data['SampleDepth_mAhD'] = wellElevs-test

dataFilt = data.iloc[:,[0,1,-1,-2]].dropna(how = 'any', axis = 0)

#writer = pd.ExcelWriter('EC_Depth_SIMwells.xlsx')

fig = plt.subplots(5,1)

for name, group in dataFilt.groupby('Site Ref'):
    print(name)
    x = group.iloc[:,1].values
    y = group.iloc[:,2].values
    z = group.iloc[:,3].values
    plt.scatter(x,y, c= z)
#    print(group)
#    group.to_excel(writer,str(name), index = False)
writer.save()    