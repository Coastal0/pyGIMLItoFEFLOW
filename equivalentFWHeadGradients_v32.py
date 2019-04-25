# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:16:54 2019

Correct measured water levels for variable density heads

@author: 264401k
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

# Hampel-style filter
def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    return(vals)
    
wells = {
        61611702: {'name':'SIM2', 'dist':30, 'elevTOC':6.651, 'elevMP': 6.706}, 
        61611701: {'name':'SIM1', 'dist':105, 'elevTOC':11.572, 'elevMP': 11.73}, 
        61611703: {'name':'SIM3', 'dist':190, 'elevTOC':15.601, 'elevMP': 15.767}, 
        61611706: {'name':'SIM6', 'dist':360, 'elevTOC':24.648, 'elevMP': 24.874}, 
        61611704: {'name':'SIM4', 'dist':550, 'elevTOC':31.86, 'elevMP': 32.254}, 
#        61611707: {'name':'SIM9', 'dist': 110, 'elev':999}
        }
#%% Import observed head values
dataFID = r"G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WIR_SIMWells_03-APR-2019/SIM_Compiled.xlsx"
xls = pd.ExcelFile(dataFID)
obsH = pd.read_excel(xls,'WaterLevel', parse_dates=True, infer_datetime_format=True, dayfirst=True)

# Isolate and assign useful columns
obsH_ = obsH.iloc[:,[0,1,-1]]
obsH_.columns = ['WellID','DateTime', 'WaterLevel']
obsH_ = obsH_[obsH_.WellID.isin([*wells])] # Filter by well ID
obsH_ = obsH_.dropna(how = 'any', axis = 0) # Remove NaNs

pd.options.mode.chained_assignment = None 
# Correct SIM 2
correction = 0.23
correctionDate = dt(2005,2,4)
correctionDate2 = dt(2015,6,5)
correctionDate3 = dt(2016,5,17)

mask = (obsH_['WellID'] == 61611702) & \
(obsH_['DateTime'] > correctionDate) & \
(obsH_['DateTime'] < correctionDate2)

vals = obsH_.loc[mask]['WaterLevel']

obsH_.loc[mask,'WaterLevel'] = (vals + correction)

# Plot Water Levels
fig, ax = plt.subplots()
for name, group in obsH_.groupby(['WellID']):
    ax.plot(group['DateTime'], group['WaterLevel'], label = str(name))
ax.legend()
plt.ylabel("Water Level (m)")
plt.xlabel("Date")
#%%
# Setup distance
wellsDist = {k:v['dist'] for k,v in wells.items()}
obsH_['Distance'] = obsH_['WellID'].map(wellsDist)

# Round DateTime
obsH_['DT_Days'] = obsH_['DateTime'].dt.floor(freq = 'D')

# Get values for each unique day
headFrame = pd.DataFrame()
uniqueDates = obsH_['DT_Days'].unique()
for well in obsH_['WellID'].unique():
    tempFrame = pd.DataFrame(obsH_['DT_Days'].unique())
    tempFrame['WellID'] = well
    averageHeads = np.zeros([len(uniqueDates),1])
    for i,days in enumerate(uniqueDates):
        averageHeads[i] = obsH_.loc[(obsH_["WellID"] == well) & (obsH_["DT_Days"] == days)]['WaterLevel'].mean()
    tempFrame['WaterLevel'] = averageHeads
    headFrame = headFrame.append(tempFrame)
headFrame.columns = ['DateDay','WellID','WaterLevel']

#%% Import Mass
obsQ = pd.read_excel(xls,'WaterQuality', header = [0], parse_dates=True, infer_datetime_format=True, dayfirst=True)
#obsQ.columns = obsQ.columns.map('_'.join)
#obsQ.rename_axis('Date').reset_index()
# Filter by well ID
obsQ = obsQ[obsQ.iloc[:,0].isin([*wells])] 
obsQ = obsQ.dropna(how = 'all', axis = 1) # Remove NaNs

# Get columns with conductivity values
cond_cols = [col for col in list(obsQ.columns) if 'uS/cm' in col] 
temperature_cols = [col for col in list(obsQ.columns) if 'deg C' in col] 
compensated_cond = set(cond_cols).intersection(temperature_cols)
# TO-DO: Correct uncorrected conductivity to standard temperature!!!
obsQ['Conductivity_Combined'] = obsQ.loc[:,cond_cols].sum(axis = 1, skipna = True, min_count = 1).dropna(how = 'all', axis = 0)

# Get columns with depth values
depth_cols = [col for col in list(obsQ.columns) if 'Sample Depths M' in col] 
depths = obsQ.loc[:,depth_cols].sum(axis = 1, skipna = True, min_count = 1).dropna(how = 'all', axis = 0)

# Get well head elevations and correct screens to sealevel.
wellElevs = pd.Series([wells[el]["elevTOC"] for el in obsQ.iloc[:,0]])
obsQ['SampleDepth_mAhD'] = wellElevs-depths

# Isolate useful columns
obsQ_ = obsQ.iloc[:,[0,1,-1,-2]]
obsQ_.columns = ['WellID','DateTime', 'SampleDepth','Conductivity']
obsQ_ = obsQ_.dropna(how = 'any', axis = 0) # Remove NaNs

# Set day-limited datetime
obsQ_['DT_Days'] = obsQ_['DateTime'].dt.floor(freq = 'D')
uniqueDates = obsQ_['DT_Days'].unique()

# Exclude depth outliers (must be per-well basis)
EC_filtframe = pd.DataFrame()
for well in obsQ_['WellID'].unique():
    wellDF = obsQ_[obsQ_['WellID'] == well]
    wellDF_filt = wellDF[np.abs(wellDF['SampleDepth']-wellDF['SampleDepth'].mean()) <= (2*wellDF['SampleDepth'].std())]
    EC_filtframe = EC_filtframe.append(wellDF_filt)
EC_filtframe = EC_filtframe.reset_index(drop = True) 

# Filter conductivity outliers
EC_filtframe = EC_filtframe[(np.abs(EC_filtframe['Conductivity']-EC_filtframe['Conductivity'].mean()) <= (4*EC_filtframe['Conductivity'].std()))].reset_index(drop=True)

# Calculate average EC value for each well on measured days
EcFrame = pd.DataFrame()
for well in EC_filtframe['WellID'].unique():
    print(well)
    wellBool = EC_filtframe['WellID'] == well
    unqDatesWell = EC_filtframe['DT_Days'][wellBool].unique()
    tempFrame = pd.DataFrame(unqDatesWell)
    tempFrame['WellID'] = well
    averageEC = np.zeros([len(unqDatesWell),1])
    avgDepths = np.zeros([len(unqDatesWell),1])
    for i,days in enumerate(unqDatesWell):
        averageEC[i] = EC_filtframe[wellBool][EC_filtframe['DT_Days'] == days]['Conductivity'].mean()
        avgDepths[i] = EC_filtframe[wellBool][EC_filtframe['DT_Days'] == days]['SampleDepth'].mean()
    tempFrame['EC'] = averageEC
    tempFrame['Depth'] = avgDepths
    EcFrame = EcFrame.append(tempFrame)
EcFrame.columns = ['DateDay','WellID','EC', 'Depth']
EcFrame = EcFrame.reset_index(drop = True) 
#EcFrame = EcFrame.set_index('DateDay', drop = True)

mergeFrame = pd.merge(EcFrame,headFrame, how ='left', on=['DateDay','WellID']) 
test = mergeFrame.drop_duplicates()

#%% Setup distance
wellsDist = {k:v['dist'] for k,v in wells.items()}
mergeFrame['Distance'] = mergeFrame['WellID'].map(wellsDist)

# Conversion factors from EC to mass
k = 0.545 # conversion factor
mergeFrame['Mass'] = mergeFrame['EC']*k

mergeFrame['Mass_Filt'] = pd.Series()
for well in mergeFrame.WellID.unique():
    filtMass = hampel(mergeFrame[mergeFrame['WellID'] == well]['Mass'])
    mergeFrame['Mass_Filt'].update(filtMass)
mergeFrame = mergeFrame.dropna()

#%% Calculate equivalent freshwater head
import seawater
"""
Example from:
"Electrical Conductivity as a Proxy for Groundwater Density in Coastal Aquifers"
by V. E. A. Post
https://www.readcube.com/articles/supplement?doi=10.1111%2Fj.1745-6584.2011.00903.x&index=0
K_15 = 42914 # Conductivity of reference KCl solution (uS/cm)
t_ref = 20 # Temperature for which densities are calculated
EC_20 = 47913 # Standard seawater at 20 degrees C (uS/cm)
S = seawater.salt(EC_20/K_15, t_ref, 0) # Calculate salinity
print('density = %4.1f' % seawater.dens0(S, t_ref)) # Calculate and print density
print("Density = {:n}".format(seawater.dens0(S, t_ref))) # # Calculate and print density (new python print method)
"""
K_15 = 42914
t_ref = 22
#EC_20 = 660
mergeFrame['Salinity'] = [(seawater.salt(s/K_15, t_ref, 0)) for s in mergeFrame["EC"]]
mergeFrame['GWDensity'] = [seawater.dens0(seawater.salt(s/K_15, t_ref, 0), t_ref) for s in mergeFrame["EC"]]

rho_fw = min(mergeFrame["GWDensity"])
mergeFrame['FWHead'] = (mergeFrame['GWDensity']/rho_fw)*mergeFrame["WaterLevel"] - mergeFrame["Depth"]*((mergeFrame['GWDensity']-rho_fw)/rho_fw)

#%% Plot data against time
# Plot salinity
fig, ax = plt.subplots()
for name, group in mergeFrame.groupby(['WellID']):
    print(name)
    ax.plot(group['DateDay'], group['Salinity'], label = str(name))
ax.legend()

# Plot Water Levels
fig, ax = plt.subplots()
for name, group in mergeFrame.groupby(['WellID']):
    ax.plot(group['DateDay'], group['WaterLevel'], label = str(name))
ax.legend()
plt.ylabel("Water Level (m)")
plt.xlabel("Date")

# Plot EQ FW Heads
fig, ax = plt.subplots()
for name, group in mergeFrame.groupby(['WellID']):
    ax.plot(group['DateDay'], group['FWHead'], label = str(name))
ax.legend()
plt.ylabel("EFH (m)")
plt.xlabel("Date")


#%% Plot Equivalent FW level vs distance
mergeFrame['year'] = mergeFrame['DateDay'].astype('datetime64[Y]')
for i, year in mergeFrame.groupby('year'):
    fig, ax = plt.subplots()
    fig.set_size_inches(12,4)
    plt.title(i.strftime('%Y'))
    plt.ylabel("FWHead Level (mAHd)")
    plt.xlabel("Distance from shoreline (m)")
    plt.ylim(0,1)
    plt.xlim(0,600)
    for name, group in year.groupby(by = 'DateDay'):
        group = group.sort_values(by = 'Distance')
        x = group['Distance']
        y = group['FWHead']
        dname = name.strftime('%Y-%m-%d')
        ax.plot(x,y, '-o', label = dname)
    plt.legend()
    fig.set_tight_layout('tight')
    fig.savefig(i.strftime('%Y'))
    plt.close()

#%% Get yearly averages and ranges
years = mergeFrame.DateDay.dt.year.unique()
cols = [[int(s) for s in mergeFrame['WellID'].unique()],['Mean','Max', 'Min']]
Statsframe_salinity = pd.DataFrame(data = None, index = years, columns = pd.MultiIndex.from_product(cols))
Statsframe_FWheads = pd.DataFrame(data = None, index = years, columns = pd.MultiIndex.from_product(cols))
Statsframe_heads = pd.DataFrame(data = None, index = years, columns = pd.MultiIndex.from_product(cols))

WellGroups = mergeFrame.groupby('WellID')
for name, group in WellGroups:
    print(name)
    for y in years:
        print(y)
        Statsframe_salinity[name]['Mean'][y] = group[(group.DateDay.dt.year == y)]['Salinity'].describe()['mean']
        Statsframe_salinity[name]['Max'][y] = group[(group.DateDay.dt.year == y)]['Salinity'].describe()['max']
        Statsframe_salinity[name]['Min'][y] = group[(group.DateDay.dt.year == y)]['Salinity'].describe()['min']
        Statsframe_FWheads[name]['Mean'][y] = group[(group.DateDay.dt.year == y)]['FWHead'].describe()['mean']
        Statsframe_FWheads[name]['Max'][y] = group[(group.DateDay.dt.year == y)]['FWHead'].describe()['max']
        Statsframe_FWheads[name]['Min'][y] = group[(group.DateDay.dt.year == y)]['FWHead'].describe()['min']
        Statsframe_heads[name]['Mean'][y] = group[(group.DateDay.dt.year == y)]['WaterLevel'].describe()['mean']
        Statsframe_heads[name]['Max'][y] = group[(group.DateDay.dt.year == y)]['WaterLevel'].describe()['max']
        Statsframe_heads[name]['Min'][y] = group[(group.DateDay.dt.year == y)]['WaterLevel'].describe()['min']
    
#%% Group by unique datetimes and get linfit
pickWells = sorted(list(wells.keys()))
pickWells = np.delete(pickWells, 1)
hydCond = 200
aquiferThickness = 30
T = hydCond*aquiferThickness
combs = list(it.combinations(pickWells,4))
showGraph = 1
printToFile = 1
fig = plt.subplots()
for comb in combs:
    print(comb)
    coefList = []
    dates = []
    coefs = []
    flows = []
    for name, group in mergeFrame.groupby(['DateDay']):
        x = group[group['WellID'].isin(comb)]['Distance']
        y = group[group['WellID'].isin(comb)]['FWHead']
#        y = group[group['WellID'].isin(comb)]['WaterLevel']

        
        if any([len(x),len(y)]):
            coef, regr = linest(x = x, y = y)
#            print('Coefficient for {0} is: {1}'.format(name.date(), coef))
            if coef != 0:
                dates.append(name.date())
                coefs.append(coef)
                flow = coef*T*(365/1000)
                flows.append(flow)
                coefList.append([str(name.date()), str(coef), str(flow)])
                
    if showGraph == 1:
        fig[1].plot(dates, flows, label = str(comb))
    if printToFile == 1:
        fName = "HydraulicGradients_T-{0}_.txt".format(T)
        with open(fName, "w") as output:
            output.write("Hydraulic Gradient between: {0} for {1} m/d \n".format(pickWells, T))
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

#%% Plotting Date vs Salinity

#fig = plt.subplots()
for w in mergeFrame.WellID.unique():
#    w = 61611703
    x = mergeFrame[mergeFrame['WellID'] == w]['DateDay']
    y = mergeFrame[mergeFrame['WellID'] == w]['EC']
    #plt.plot(x, y, color = 'k')
    #plt.fill_between(x.dt.to_pydatetime(),y.min(), y, color = 'grey')
    x.astype('datetime64[D]')
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    xx = [mpl.dates.date2num(x_.to_pydatetime()) for x_ in x]
    xx = np.asarray(xx)
    xx = np.append(xx,xx.max())
    xx = np.insert(xx,0,xx.min())
    yy = np.asarray(y)
    yy = np.append(yy,0.0)
    yy = np.insert(yy,0,0.0)
    
    yy = yy/1000
    x_, y_ = np.meshgrid(xx,yy)
    z_ = y_
    
    fig, ax = plt.subplots(figsize = (3,3))
#    ax.set_title(w, loc = 'left', fontsize = 12)
    plt.tight_layout()
    fill = plt.contourf(x_,y_,z_, 100, cmap='Spectral_r', vmin = 0, vmax = 50)
    plt.fill_between(xx,yy, yy.max(), color = 'white')
    
    #path = Path(np.array([xx,yy]).transpose())
    #patch = PathPatch(path, facecolor='none')
    #plt.gca().add_patch(patch)
    
    #im = plt.imshow((yy.reshape(xx.size,1).T),   
    #                cmap='Spectral_r',
    #                clim = (100,50000),
    #                norm = SqueezedNorm(vmin = 0, vmax = 50000, mid = 200, s1 =1, s2 = 1.2),
    #                interpolation="bicubic",
    #                origin='lower',
    #                extent=[xx.min(),xx.max(),yy.min(),yy.max()],
    #                aspect="auto",
    #                clip_path=patch, 
    #                clip_on=True)
    
    plt.plot(xx, yy, color = 'k')
    plt.yscale("linear")
    plt.ylim(0,50)
    plt.show()
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=1)
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel('Date')
    plt.ylabel(r'EC (mS/cm)')
    ax.tick_params(which = 'both', direction = 'in')
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    plt.tight_layout()
    fig.savefig(str(w)+'.png',dpi = 500,transparent=True)
#%% Export to excel
writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')  
 
# Write each dataframe to a different worksheet.
mergeFrame.to_excel(writer, sheet_name='Data')
Statsframe_salinity.to_excel(writer, sheet_name='Stats_Salinity')
Statsframe_heads.to_excel(writer, sheet_name='Stats_Heads')
Statsframe_FWheads.to_excel(writer, sheet_name='Stats_FWHeads')

# Close the Pandas Excel writer and output the Excel file.
writer.save()