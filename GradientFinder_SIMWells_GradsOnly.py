# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:59:22 2018

@author: 264401k

Try to find some relationship between tides and SIM well measurements.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from datetime import datetime as dt
import matplotlib.pylab as pl
import matplotlib as mpl
import itertools as it

def linest(x, y):
    y = y.values.reshape([y.size, 1])
    x = x.values.reshape([x.size, 1])

    regr = linear_model.LinearRegression(fit_intercept = True)
    regr.fit(x, y)
    regr.coef_
    return regr.coef_[0][0], regr

wells = {
        61611702: {'name':'SIM2', 'dist':30, 'elevTOC':6.651, 'elevMP': 6.706}, 
        61611701: {'name':'SIM1', 'dist':105, 'elevTOC':11.572, 'elevMP': 11.73}, 
        61611703: {'name':'SIM3', 'dist':190, 'elevTOC':15.601, 'elevMP': 15.767}, 
        61611706: {'name':'SIM6', 'dist':360, 'elevTOC':24.648, 'elevMP': 24.874}, 
        61611704: {'name':'SIM4', 'dist':550, 'elevTOC':31.86, 'elevMP': 32.254}, 
#        61611707: {'name':'SIM9', 'dist': 110, 'elev':999}
        }
#%% Import observed head values
ObservedHeads_FID = "G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\QuinnsRocks_WirBores\WaterLevelsDiscreteForSiteCrossTab.xlsx"
ObservedHeads = pd.read_excel(ObservedHeads_FID, header = [0,1], parse_dates=True, infer_datetime_format=True, dayfirst=True)
ObservedHeads.columns = ObservedHeads.columns.map(''.join)
ObservedHeads = ObservedHeads.reset_index()

# Isolate and assign useful columns
ObservedHeads = ObservedHeads.iloc[:,[0,1,-2]]
ObservedHeads.columns = ['WellID','DateTime', 'WaterLevel']
ObservedHeads = ObservedHeads[ObservedHeads.WellID.isin([*wells])] # Filter by well ID
ObservedHeads = ObservedHeads.dropna(how = 'any', axis = 0) # Remove NaNs

# Correct SIM 2 measurements
sim2mask = (ObservedHeads['WellID'] == 61611702) & (ObservedHeads['DateTime'] > '02/04/2005') & (ObservedHeads['DateTime'] < '17/11/2016')
ObservedHeads['WaterLevel'][sim2mask] += 0.230

sim1mask = (ObservedHeads['WellID'] == 61611701) & (ObservedHeads['DateTime'] > '22/08/2016')
ObservedHeads['WaterLevel'][sim1mask] += 0.230

sim3mask = (ObservedHeads['WellID'] == 61611703) & (ObservedHeads['DateTime'] > '22/08/2016')
ObservedHeads['WaterLevel'][sim3mask] += 0.230

# Filter outliers
for name, group in ObservedHeads.groupby(by = 'WellID'):
    print(name)
    outliers = group[np.abs(group.WaterLevel-group.WaterLevel.mean()) <= (3*group.WaterLevel.std())]

# Setup distance
wellsDist = {k:v['dist'] for k,v in wells.items()}
ObservedHeads['Distance'] = ObservedHeads['WellID'].map(wellsDist)

# Round DateTime
ObservedHeads['DT_Days'] = ObservedHeads['DateTime'].dt.floor(freq = 'D')
    
# Plot Water levels against time
fig, ax = plt.subplots()
for name, group in ObservedHeads.groupby(['WellID']):
    ax.plot(group['DateTime'], group['WaterLevel'], label = str(name))
ax.legend()
plt.ylabel("Water Level (m)")
plt.xlabel("Date")

# Plot water level vs distance
ObservedHeads['year'] = ObservedHeads['DT_Days'].astype('datetime64[Y]')
for i, year in ObservedHeads.groupby('year'):
    fig, ax = plt.subplots()
    fig.set_size_inches(12,4)
    plt.title(i.strftime('%Y'))
    plt.ylabel("Water Level (mAHd)")
    plt.xlabel("Distance from shoreline (m)")
    plt.ylim(0,1)
    plt.xlim(0,600)
    for name, group in year.groupby(by = 'DT_Days'):
        group = group.sort_values(by = 'Distance')
        x = group['Distance']
        y = group['WaterLevel']
        dname = name.strftime('%Y-%m-%d')
        ax.plot(x,y, '-o', label = dname)
    plt.legend()
    fig.set_tight_layout('tight')
    fig.savefig(i.strftime('%Y'))
    plt.close()
# Set up empty vectors
days = np.unique(ObservedHeads['DT_Days'].values.astype('datetime64[D]'))
a = np.zeros([len(days), 1], dtype = 'datetime64[D]')
b = np.zeros([len(days), 1])
values = []

# Group by unique datetimes and get linfit
pickWells = sorted(list(wells.keys()))[:]
combs = list(it.combinations(pickWells,5))
showGraph = 1
printToFile = 1
fig = plt.subplots()
for comb in combs:
    print(comb)
    coefList = []
    dates = []
    coefs = []
    flows = []
    for name, group in ObservedHeads.groupby(['DT_Days']):
        x = group[group['WellID'].isin(comb)]['Distance']
        y = group[group['WellID'].isin(comb)]['WaterLevel']
        
        if any([len(x),len(y)]):
            coef, regr = linest(x = x, y = y)
#            print('Coefficient for {0} is: {1}'.format(name.date(), coef))
            if coef != 0:
                dates.append(name.date())
                coefs.append(coef)
                flow = coef*6000*(365/1000)
                flows.append(flow)
                coefList.append([str(name.date()), str(coef), str(flow)])
                
    if showGraph == 1:
        fig[1].plot(dates, flows, label = str(comb))
    if printToFile == 1:
        fName = "{0}_{1}.txt".format(comb[0], comb[-1])
        with open(fName, "w") as output:
            output.write("Gradient between: {0} \n".format(pickWells))
            for lines in coefList:
                s = ", ".join(lines)
                output.write(s)
                output.write('\n')        
fig[1].legend()
plt.ylabel("groundwater Throughflow (ML/y)")
plt.xlabel("Date")

#    if len(group) > 1:
##        print(group)
#        coef, regr = linest(x = group['Distance'], y = group['WaterLevel_AHD'])
#        if coef != 0:
#        #    plt.plot(group['Distance'],group['WaterLevel_AHD'], marker = 'o', alpha = 0.1)
#        #    plt.plot(group['Distance'], coef*group['Distance']+regr.intercept_)
#            a[i] = name
#            b[i] = coef
#            i = i + 1
        

