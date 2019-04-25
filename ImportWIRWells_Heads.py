# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:58:44 2018

@author: 264401k

Incoming data file needs to be filtered
e.g
WELL, DATETIME, STATIC, WL

"""
import pandas as pd
fid = r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\QuinnsRocks_WirBores\115375\WaterLevelsDiscreteForSiteCrossTab.xlsx"

data = pd.read_excel(fid)

for name, group in data.groupby('SiteRef'):
    print(name)
    group.to_csv(str(name)+'.txt', index = None, header = None)
