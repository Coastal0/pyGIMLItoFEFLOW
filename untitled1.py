# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:16:24 2018

@author: 264401k
"""

import pandas as pd

fName = r'G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WaterLevelsForSite_filt.xlsx'
raw_data = []
raw_data = pd.read_excel(fName)
colNames = raw_data.columns.tolist()
raw_data.iloc[:,0] = raw_data.iloc[:, 0].astype('category')

import matplotlib.pyplot as plt

for label, grp in raw_data.groupby(raw_data.iloc[:,0]):
    d = grp[['Date', 'WaterLevel_AHD']]
    d.set_index('Date', inplace = True)
    d.index = pd.DatetimeIndex(d.index).round('H')

    d = d[~d.index.duplicated()]
    dr = d.resample('H').asfreq()
    drInterp = dr.interpolate(method = 'pchip')
    drInterp['ma'] = drInterp['WaterLevel_AHD'].rolling(window = 24*365, min_periods = 1, center = True, closed = 'both').mean()
#    plt.plot(drInterp['ma'], label = grp['Site'].unique().categories.values.tolist())
    fName = r'G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/'+str(grp['Site'].unique().categories.values.tolist()[0])+'.xlsx'
    drInterp.to_excel(fName)
plt.legend()
ax = plt.gca()
ax.grid(alpha = 0.3)
ax.set_axisbelow(True)
ax.minorticks_on()
