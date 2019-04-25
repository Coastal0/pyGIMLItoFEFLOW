# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:10:28 2018

@author: 264401k
"""

import matplotlib.pyplot as plt
import pandas as pd

# Setup index
wells = ['SIM2', 'SIM1', 'SIM3', 'SIM6', 'SIM4']

#%% OBSERVED MEASUREMENTs
# EC
Obs_EC_FID = "G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WellID_Date_1990-2017_Depth_EC.dat"
Obs_EC = pd.read_csv(Obs_EC_FID,delim_whitespace=True, index_col = None)
Obs_EC['Date'] = pd.to_datetime(Obs_EC['Date'])
Obs_EC['Mass'] = Obs_EC['Cond']*0.5
dateMask = Obs_EC['Date'] < '01/01/1998'
Obs_EC_Stats = Obs_EC[dateMask].groupby('WellID')['Mass'].describe()
Obs_EC_median = Obs_EC[dateMask].groupby('WellID')['Mass'].median()
Obs_EC_max = Obs_EC[dateMask].groupby('WellID')['Mass'].max()
Obs_EC_min = Obs_EC[dateMask].groupby('WellID')['Mass'].min()
Obs_EC_range = Obs_EC_max-Obs_EC_min

# HEAD
Obs_HEA_FID = "G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WellID_Date_1990-2017_HydHead.dat"
Obs_HEA = pd.read_csv(Obs_HEA_FID,delim_whitespace=True, index_col = None)
Obs_HEA['Date'] = pd.to_datetime(Obs_HEA['Date'])
dateMask = dateMask
Obs_HEA_Stats = Obs_HEA[dateMask].groupby('WellID')['WaterLevel_AHD'].describe()
Obs_HEA_median = Obs_HEA[dateMask].groupby('WellID')['WaterLevel_AHD'].median()
Obs_HEA_max = Obs_HEA[dateMask].groupby('WellID')['WaterLevel_AHD'].max()
Obs_HEA_min = Obs_HEA[dateMask].groupby('WellID')['WaterLevel_AHD'].min()
Obs_HEA_range = Obs_HEA_max-Obs_HEA_min


#%% SIMULATED MEASUREMENTS
# Mass
FID = r"K:/Porosity/QR_Quad_Richars_0100mday_poro_zone2_MASS-CONC-Graph.dat"
mass = pd.read_csv(FID, delim_whitespace=True)
massmin = mass.loc[mass.groupby('Curve')["X"].idxmin()][['Curve','Y']].values
massmax = mass.loc[mass.groupby('Curve')["X"].idxmax()][['Curve','Y']].values
index = massmin[:,0]
SimMassStats = pd.DataFrame({'No Zone': massmin[:,1], 'High-K Zone': massmax[:,1], 'Obs. (1998+)': Obs_EC_median}, index = index)
SimMassStats = SimMassStats.reindex(index = wells)
SimMassStats = SimMassStats/1000

# HEAD
FID = r"K:/Porosity/QR_Quad_Richars_0100mday_poro_zone2_HYD-HEAD-Graph.dat"
head = pd.read_csv(FID, delim_whitespace=True)
headmin = head.loc[head.groupby('Curve')["X"].idxmin()][['Curve','Y']].values
headmax = head.loc[head.groupby('Curve')["X"].idxmax()][['Curve','Y']].values
index = headmin[:,0]
SimHeadStats = pd.DataFrame({'No Zone': headmin[:,1], 'High-K Zone': headmax[:,1],'Obs. (1998+)': Obs_HEA_median}, index = index)
SimHeadStats = SimHeadStats.reindex(index = wells)

# %% Plotting
colors = [[0,0,1],[0,1,0],[1,0,0]]
fig, ax = plt.subplots(1,2)
fig.sca(ax[0])
SimHeadStats.plot.bar(rot = 0, edgecolor = 'black', alpha = 0.6, align = 'center', width = 0.8, ax = ax[0], color = colors)
plt.ylabel('Hydraulic Head [mAHD]')
plt.title('Comparison of High-K Zone: Hydraulic Head')

fig.sca(ax[1])
SimMassStats.plot.bar(rot = 0, edgecolor = 'black', alpha = 0.6, align = 'center', width = 0.8,ax = ax[1], color = colors)
plt.ylabel('Mass Concentration [g/L]')
plt.title('Comparison of High-K Zone: Mass Concentration')

plt.gcf().set_size_inches(12,4)
plt.gcf().set_tight_layout('tight')
