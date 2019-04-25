# -*- coding: utf-8 -*-
"""
Data from: http://www.bom.gov.au/oceanography/projects/abslmp/data/index.shtml

@author: 264401k
"""

#%% Load CSV's
import pandas as pd
import os
import re
import glob
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

data = []
workDir = r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\Hillarys_Tides&Sim\Tide Data (Hillarys)" #Set data directory.
os.chdir(workDir) #Change to data directory.
yearsList = glob.glob("*.csv") #Make a list of the csv's in the directory.
years = [re.split(r'[_|.]',y)[1] for y in yearsList] #Split the year from the filename.

data = pd.read_csv(yearsList[0]) #Read the first years data (probably be the lowest number if sorted properly).
for y in yearsList[1:]:
    print(y) #Check it's loading as you think it should be.
    df2 = pd.read_csv(y) #Load it.
    data = data.append(df2, ignore_index=True) #Append it to the original, ignoring the index.



data = data.replace(-9999, np.NaN) #Replace the null values (-9999) with nans to stop them plotting.
print('Converting to datetime64... (May cause hangup)')
data.iloc[:,0] = pd.to_datetime(data.iloc[:,0])
print('Done.')
data = data.dropna(axis = 0, thresh = 3)
#data_fn = data.fillna(method = 'ffill')
#data = data.reset_index()
#del yearsList, y, workDir, df2, os, re, glob

#%% Plot CSV's
import matplotlib as mpl
import matplotlib.pyplot as plt

ma1yr = data["Sea Level"].rolling(window = (365*24), center = True, win_type = 'hann', min_periods = 1).mean()
ma10yr = data["Sea Level"].rolling(window = (10*365*24), center = True, win_type = 'hann', min_periods = 1).mean()
maxBiannual = data["Sea Level"].rolling(window = 180*24, min_periods = 1, center = True).max()
minBiannual  = data["Sea Level"].rolling(window = 180*24, min_periods = 1, center = True).min()

dPlot = data.plot(figsize = (12,9), x = [" Date & UTC Time"],y = ["Sea Level"], linewidth = 0.1, alpha = 1, color = 'black')
dPlot.minorticks_on()
dPlot.set_ylabel('Sea Level (mAHD)')
ax = plt.gca()
ax.set_title("Hourly Tidal Measurements - Hillary's, Perth W.A.", fontweight = 'bold')
ax.grid('on', alpha = 0.5)
ax.axes.set_xlim(left = min(data[" Date & UTC Time"]), right = max(data[" Date & UTC Time"]))
ax.set_ylim(-1,2)
ax.xaxis.set_major_locator(mpl.dates.YearLocator(2))
ax.xaxis.set_minor_locator(mpl.dates.YearLocator(1))
fig = plt.gcf()
fig.set_tight_layout('tight')

d = data[" Date & UTC Time"].values # bug with dataframes not parsing datetime64 to plt. Save as np array explicitly
#m = data["Sea Level"]
#plt.plot(d,m, alpha = 0.2)

ma1yr = data["Sea Level"].rolling(window = (365*24), center = True, win_type = 'hann', min_periods = 1).mean()
ma10yr = data["Sea Level"].rolling(window = (10*365*24), center = True, win_type = 'hann', min_periods = 1).mean()
maxBiannual = data["Sea Level"].rolling(window = 180*24, min_periods = 1, center = True).max()
minBiannual  = data["Sea Level"].rolling(window = 180*24, min_periods = 1, center = True).min()

plt.plot(data[" Date & UTC Time"],ma1yr, color='red', alpha=1, linewidth = 2, zorder=10, label = 'Sea Level (mAHD) - 1 yr Moving Average')
plt.plot(data[" Date & UTC Time"],ma10yr, color='red', alpha=1, linestyle='--', zorder=9, label = 'Sea Level (mAHD) - 10 yr Moving Average')
plt.plot(data[" Date & UTC Time"],maxBiannual, color='blue', alpha=0.3, label = 'Sea Level (mAHD) - 6 month maximum')
plt.plot(data[" Date & UTC Time"],minBiannual, color='red', alpha=0.3, label = 'Sea Level (mAHD) - 6 month minimum')
plt.fill_between(d,maxBiannual,minBiannual,alpha=0.1, color='lightseagreen',zorder=1)
plt.legend(loc = 4)
#Inset#1
a = plt.axes([0.12, 0.15, 0.4, 0.2])
ma = data["Sea Level"].rolling(window = 24, center = True, win_type = 'hann').mean()
plt.plot(d,ma1yr, color = 'red', alpha = 1, linewidth = 2)

plt.plot(data[" Date & UTC Time"],data["Sea Level"], color='black', alpha=0.8)
max24 = data["Sea Level"].rolling(window = 24, min_periods = 1, center = True).max()
min24 = data["Sea Level"].rolling(window = 24, min_periods = 1, center = True).min()
plt.fill_between(d,max24,min24,alpha=0.2, color='lightseagreen')
plt.plot(d,max24, color='blue', alpha=0.3)
plt.plot(d,min24, color='red', alpha=0.3)
a.set_xlim(left = 736010, right = 736015)
a.set_ylim(0.3,1.1)
a.set_title('February, 2016', fontweight = 'bold')
xformatter = mpl.dates.DateFormatter('%H:00\n%d/%m')
a.xaxis.set_major_locator(mpl.dates.HourLocator(interval = 24))
a.xaxis.set_minor_locator(mpl.dates.HourLocator(interval = 12))
a.xaxis.set_major_formatter(xformatter)
a.grid('off')

fig.savefig('Tidal.png', dpi = 600)

#%% Write POW
l = list(range(227927))
l = np.divide(l,24)
np.savetxt('HourDAC.pow',l)
