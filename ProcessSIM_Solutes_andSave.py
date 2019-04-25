# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:35:23 2018

@author: 264401k
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup index
wells = ['SIM2', 'SIM1', 'SIM3', 'SIM6', 'SIM4']

# EC
Obs_EC_FID = "G:/PROJECTS/PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface/Data/QuinnsRocks_WirBores/WellID_Date_1990-2017_Depth_EC.dat"
Obs_EC = pd.read_csv(Obs_EC_FID, delim_whitespace=False, index_col = 1, parse_dates = True)
Obs_EC.sort_values('WellID', inplace = True)

Obs_WL_FID = "G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\QuinnsRocks_WirBores\WellID_Date_1990-2017_HydHead.dat"
Obs_WL = pd.read_csv(Obs_WL_FID, delim_whitespace=False, index_col = 0, parse_dates = True)
Obs_WL.sort_values('WellID', inplace = True)
#
#writer = pd.ExcelWriter('.\TestWL.xlsx')
#for name, group in Obs_WL.groupby('WellID'):
#    print(name)
#    uniqueDates = group.index.unique()
#    averageWL = np.zeros([len(uniqueDates),1])
#    for i,d in enumerate(uniqueDates):
##        print(d)
#        averageWL[i] = group[group.index == d]['WaterLevel_AHD'].mean()
#
#    test = pd.DataFrame(averageWL, uniqueDates)
#    test.to_excel(writer, name)
#writer.save()    
#
#writer = pd.ExcelWriter('.\Test.xlsx')
#for name, group in Obs_EC.groupby('WellID'):
#    print(name)
#    uniqueDates = group.index.unique()
#    averageConds = np.zeros([len(uniqueDates),1])
#    for i,d in enumerate(uniqueDates):
##        print(d)
#        averageConds[i] = group[group.index == d]['Cond'].mean()
#
#    test = pd.DataFrame(averageConds, uniqueDates)
#    test.to_excel(writer, name)
#writer.save()    

testMerge = pd.merge(Obs_WL,Obs_EC, how='outer', left_index=True, right_index=True, on = 'WellID')

writer = pd.ExcelWriter('.\TestMerge.xlsx')
for name, group in testMerge.groupby('WellID'):
    print(name)
    uniqueDates = group.index.unique().values.astype('datetime64[D]')
    averageConds = np.zeros([len(uniqueDates),1])
    averageWL = np.zeros([len(uniqueDates),1])

    for i,d in enumerate(uniqueDates):
#        print(d)
        averageConds[i] = group[group.index == d]['Cond'].mean()
        averageWL[i] = group[group.index == d]['WaterLevel_AHD'].mean()

    test = pd.DataFrame(data = np.concatenate((averageConds,averageWL), axis = 1), index = uniqueDates)
    test.to_excel(writer, name)
writer.save()    