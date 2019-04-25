# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:40:37 2018

@author: 264401k
"""

## Process and present SIM1-15 hourly timelapse hydraulic head data.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

data = pd.read_excel("G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\SIM 1-15_AHD_Levels.xlsx")

data = data.assign(RollingMean_day = data['Head (mAHD)'].rolling(window=24, center=True).mean())
data = data.assign(RollingMean_1month = data['Head (mAHD)'].rolling(window=24*7, center=True).mean())
data = data.assign(RollingMean_6month = data['Head (mAHD)'].rolling(window=24*180, center=True).mean())
ax = data.plot(x='Date', y='Head (mAHD)')
data.plot(x='Date', y='RollingMean_day', ax = ax)
data.plot(x='Date', y='RollingMean_1month', ax = ax)
data.plot(x='Date', y='RollingMean_6month', ax = ax)

## Fancier plotting (with matplotlib).
d = data['Date'].values # bug with dataframes not parsing datetime64 to plt. Save as np array explicitly
ma = data['Head (mAHD)'].rolling(window = 24, center = True, win_type = 'hann').mean()
max24 = data['Head (mAHD)'].rolling(24).max()
min24 = data['Head (mAHD)'].rolling(24).min()
plt.plot(d,ma, color='black', alpha=0.8)
plt.plot(d,max24, color='blue', alpha=0.3)
plt.plot(d,min24, color='red', alpha=0.3)
plt.fill_between(d,max24,min24,alpha=0.2, color='grey')

max12 = data['Head (mAHD)'].rolling(window = 12).max()
min12 = data['Head (mAHD)'].rolling(12).min()

maxWeek = data['Head (mAHD)'].rolling(24*7, center = True).max()
minWeek = data['Head (mAHD)'].rolling(24*7, center = True).min()
maxWeekI = maxWeek.interpolate('akima')
plt.plot(d,ma, color='black', alpha=0.8)
plt.plot(d,maxWeek, color='blue', alpha=0.3)
plt.plot(d,minWeek, color='red', alpha=0.3)
plt.fill_between(d,maxWeek,minWeek,alpha=0.2, color='blue')

maxMonth = data['Head (mAHD)'].rolling(24*30, center = True).max()
minMonth = data['Head (mAHD)'].rolling(24*30, center = True).min()
plt.plot(d,ma, color='black', alpha=0.8)
plt.plot(d,maxMonth, color='blue', alpha=0.3)
plt.plot(d,minMonth, color='red', alpha=0.3)
plt.fill_between(d,minMonth,maxMonth,alpha=0.2, color='blue')

max6Month = data['Head (mAHD)'].rolling(24*180, center = True).max()
min6Month = data['Head (mAHD)'].rolling(24*180, center = True).min()

plt.plot(d,data['Head (mAHD)'],'grey', linewidth=0.5)
plt.plot(d,ma,'b')
plt.plot(d,ma, color='black', alpha=0.8)
plt.plot(d,max6Month, color='blue', alpha=0.3)
plt.plot(d,min6Month, color='red', alpha=0.3)
plt.fill_between(d,max6Month,min6Month,alpha=0.2, color='blue')

## Run from here
ma = data['Head (mAHD)'].rolling(window = 24*30, center = True, win_type = 'hann').mean()

plt.plot(d,data['Head (mAHD)'], color = 'black', linewidth = 0.5)
plt.plot(d,ma, color = 'red', alpha = 1, linewidth = 2)
plt.fill_between(d,minWeek,maxWeek,alpha=0.2, color='blue')

fig = plt.gcf()
fig.set_size_inches([10,7])
ax = plt.gca()
ax.set_ylabel('Hydraulic Head (mAHD)')
ax.set_xlabel('Date')
ax.set_title('Hourly Hydraulic Head Measurements at the Coast', fontweight = 'bold')
fig.set_tight_layout('tight')
ax.minorticks_on()
ax.grid('on', 'major', linewidth = 0.5)
ax.grid('on', 'minor', linestyle ='-', linewidth = 0.2)

# Inset #1 (Daily)
a = plt.axes([0.135, 0.7, 0.2, 0.2])
ma = data['Head (mAHD)'].rolling(window = 24*7, center = True, win_type = 'hann').mean()
plt.plot(d,ma, color = 'red', alpha = 1, linewidth = 2)
max24 = data['Head (mAHD)'].rolling(24).max()
min24 = data['Head (mAHD)'].rolling(24).min()
plt.plot(d,data['Head (mAHD)'], color='black', alpha=0.8)
plt.plot(d,max24, color='blue', alpha=0.3)
plt.plot(d,min24, color='red', alpha=0.3)
plt.fill_between(d,max24,min24,alpha=0.2, color='grey')
a.set_xlim(left = 736010, right = 736012)
a.set_ylim(-0.3,0.15)
a.set_title('16/17th Feb, 2016', fontweight = 'bold')
xformatter = md.DateFormatter('%H:00')
xlocator = md.HourLocator(interval = 12)
a.xaxis.set_major_locator(xlocator)
a.xaxis.set_major_formatter(xformatter)
a.grid('off')
a.minorticks_on()

# Inset #2 (Weekly)
b = plt.axes([0.75, 0.7, 0.2, 0.2])
ma = data['Head (mAHD)'].rolling(window = 24*7, center = True, win_type = 'hann').mean()
plt.plot(d,ma, color = 'red', alpha = 1, linewidth = 2)
maxWeek = data['Head (mAHD)'].rolling(24*7, center = True).max()
minWeek = data['Head (mAHD)'].rolling(24*7, center = True).min()
plt.plot(d,data['Head (mAHD)'], color='black', alpha=0.8)
plt.plot(d,max24, color='blue', alpha=0.3)
plt.plot(d,min24, color='red', alpha=0.3)
plt.fill_between(d,maxWeek,minWeek,alpha=0.2, color='grey')
b.set_xlim(left = 736180, right = 736205)
b.set_ylim(-0.2,0.4)
b.set_title('August, 2016', fontweight = 'bold')
xformatter = md.DateFormatter('%d')
xlocator = md.DayLocator(interval = 4)
b.xaxis.set_major_locator(xlocator)
b.xaxis.set_major_formatter(xformatter)
b.grid('off')
b.minorticks_on()


mstd = data['Head (mAHD)'].rolling(window = 24*7, center = True).std()
#plt.fill_between(d,ma - 2*mstd,ma + 2*mstd,alpha=0.2, color='blue')
