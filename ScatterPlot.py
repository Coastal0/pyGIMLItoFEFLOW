# -*- coding: utf-8 -*-

import numpy as np
import pygimli as pg
import glob
import matplotlib.pyplot as plt

data = pg.load(glob.glob('*.data')[0])
resp = np.loadtxt('response.vector')
obs = np.asarray(data.get('rhoa'))

plt.scatter(obs, calc, marker="+")
fig = plt.gcf()
ax = plt.gca()
plt.grid(True)
ax.set_xlim(min(min(obs),min(resp)),max(max(obs),max(resp)))
ax.set_ylim(min(min(obs),min(resp)),max(max(obs),max(resp)))

plt.plot([0,max(resp)],[0,max(obs)], color='k')
