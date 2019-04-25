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
import matplotlib

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def linest(x, y):
    y = y.values.reshape([y.size, 1])
    x = x.values.reshape([x.size, 1])

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    regr.coef_
    return regr.coef_[0][0], regr

# %% Import Data
# Import tidal data
#Tides = pd.read_csv(r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\Hillarys_Tides&Sim\Tide Data (Hillarys)\COMBINED.csv")
#Tides.iloc[:, 0] = pd.to_datetime(Tides.iloc[:, 0])
#Tides = Tides.set_index(Tides.iloc[:, 0])
#Tides = Tides.iloc[:, 1]
#Tides.isnull().sum().sum()
#Tides.dropna(inplace = True)
#Tides.isnull().sum().sum()

# Import observed head values
ObservedHeads_FID = "G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WellID_Date_1990-2017_HydHead.dat"
ObservedHeads = pd.read_csv(ObservedHeads_FID,sep='\t', index_col = None)

# Convert datetime and update index
ObservedHeads['DateTime'] = pd.to_datetime(ObservedHeads['DateTime'])
ObservedHeads['DateTime_1hrRnd'] = pd.to_datetime(ObservedHeads['DateTime']).dt.ceil('H')
#ObservedHeads = ObservedHeads.set_index(['DateTime'])
ObservedHeads_DateSort = ObservedHeads.sort_values('DateTime_1hrRnd')

ObservedHeads_WellGrp = ObservedHeads_DateSort.groupby('WellID')

for name, group in ObservedHeads_WellGrp:
    print(name)
    plt.plot(group['DateTime'].values, group['WaterLevel_AHD'].values)


values = []
for name, group in ObservedHeads_WellGrp:
    print(name)
    new_df = group.duplicated('DateTime_1hrRnd')
    values.extend(~new_df)
ObservedHeads = ObservedHeads[values].reset_index().drop(columns = ['index'])

# Set up index
wells = {'SIM2': 30, 'SIM1': 105, 'SIM3':190, 'SIM6':360,'SIM4':550}
ObservedHeads['Distance'] = ObservedHeads['WellID'].map(wells)
ObservedHeads = ObservedHeads.sort_values(by = ['Distance','DateTime'])

# %% Set up loop

#years = [1993]
years = np.unique(ObservedHeads['DateTime'].values.astype('datetime64[Y]'))
#fig, ax = plt.subplots(len(years),1, sharex = 'all', sharey = 'all', squeeze=0)
fig, ax = plt.subplots(squeeze = 0)

nVals = len(np.unique(ObservedHeads['DateTime'].values.astype('datetime64[D]')))
testArray1 = np.zeros([nVals, 1], dtype = 'datetime64[D]')
testArray2 = np.zeros([nVals, 1])
timeRange = []
counter = 0
for y in years:
#    fig, ax = plt.subplots(num = n, squeeze=0)
#    plt.ylim(-0.2, 1)
    n=0
#    print(y)
    # % Set up date mask'
    minDate = '01/01/'+str(y)
    maxDate = '31/12/'+str(y)
    dateMask = (ObservedHeads['DateTime'] > minDate) & (ObservedHeads['DateTime'] < maxDate)

    # Groupings
    ObsHeadsGroup = ObservedHeads[dateMask].reset_index().groupby(pd.Grouper(key = 'DateTime', freq = 'D'), as_index = False)
    group_list = [(index, group) for index, group in ObsHeadsGroup if len(group) == 5]


    # Plotting
    ax[n][0].set_xlabel('Distance (m)')
    ax[n][0].set_ylabel('Hydraulic Head (mAHD)')
    ax[n][0].tick_params(direction = 'in')

    # Custom color darkening/lightening
    hsv_dark = cmap_map(lambda x: x*0.80, matplotlib.cm.hsv)
    # Divide colormap into parts (e.g. summer = hot)
    colors = hsv_dark(np.linspace(0,1,12))

    i=0
    gradientList_dates = np.zeros([len(group_list), 1], dtype = 'datetime64[D]')
    gradientList_grads = np.zeros([len(group_list), 1], dtype = 'float')
    for name, group in group_list:
        if len(group) == 5:
#            print(name)
#            print(len(group))
            group.sort_values('Distance',inplace = True)
            tDelta = max(group['DateTime'])-min( group['DateTime'])
            timeRange.append(tDelta.total_seconds()/(60*60))
            # Get gradients
            coef, regr = linest(x = group['Distance'], y = group['WaterLevel_AHD'])
            gradientList_dates[i][0] = name
            gradientList_grads[i][0] = coef

            i = i+1

    for g, _ in enumerate(gradientList_dates):
#        print(g)
        testArray1[counter] = gradientList_dates[g]
        testArray2[counter] = gradientList_grads[g]
#        print(counter)
        counter = counter + 1

    ax[0][0].plot(gradientList_dates, gradientList_grads, marker = 'o')

        # Plot data
#        if any('SIM2' == group['WellID']):
##            group['WaterLevel_AHD'] = group['WaterLevel_AHD'] - group['WaterLevel_AHD'][min(group['WaterLevel_AHD'].index)]
##
##            if len(group['WaterLevel_AHD'].index) != 1:
##                group['WaterLevel_AHD'] = group['WaterLevel_AHD'] - group['WaterLevel_AHD'][group['WaterLevel_AHD'].index[1]]
##            else:
##                print(str(name) +' has only 1 measurement.')
##                group['WaterLevel_AHD'] = group['WaterLevel_AHD'] - group['WaterLevel_AHD'][min(group['WaterLevel_AHD'].index)]
#            ax[n][0].plot(group['Distance'], group['WaterLevel_AHD'],
#              label=name.date(), linestyle = '-', marker = '.',
#              alpha = 0.5, color=colors[name.month-1])
#        elif len(group['WellID']) < 5:
#            ax[n][0].plot(group['Distance'], group['WaterLevel_AHD'],
#              label=name.date(), linestyle = '--', marker = '.',
#              alpha = 0.5, color=colors[name.month-1])
#        else:
#            ax[n][0].plot(group['Distance'], group['WaterLevel_AHD'],
#              label=name.date(), linestyle = '--', marker = '.',
#              alpha = 0.5, color=colors[name.month-1])

#        ax[n][0].legend(frameon=False, loc = 2)
    fig.set_tight_layout('tight')
    fig.set_size_inches(14,4)
#    fig.savefig(str(name.date())+'_Not-corrected')
#    plt.close()