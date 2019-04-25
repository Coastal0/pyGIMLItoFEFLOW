# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:59:17 2018

@author: 264401k
"""

import numpy as np
from tqdm import tqdm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


#fIN = r"K:\results\QR_Quad_10kCells_ASCIIFEM.fem"
#fIN = r"K:\results\QR_Quad_120kCells_ASCIIFEM.fem"

tk.Tk().withdraw()
fIN = tk.filedialog.askopenfilename(filetypes=[("ASCII FEFLOW files", "*.fem")])
print(fIN)

with open(fIN) as file:
    array = file.readlines()


ProblemTitle = array[0]
ClassHeader = array[1]
Class = [int(x) for x in array[2].split()]

DimensHeader = array[3]
tLine = [int(x) for x in array[4].split()]
nNodes = tLine[0]
nEle = tLine[1]
nDims = tLine[2]  # e.g. 3 for triangles, 4 for quads.

# Elements
elements = []
for i in tqdm(array[8:nEle+7]):
    elements.append([int(x) for x in i.split()])

# Nodes
indices = [i for i, s in enumerate(array) if 'COOR' in s]
nodesXY_1D = []
for i in tqdm(array[indices[0]+1:indices[1]]):
    nodesXY_1D.extend([float(x) for x in list(filter(None, i.strip('\n').split(',')))])
nodeX = nodesXY_1D[0:nNodes]
nodeY = nodesXY_1D[nNodes:]

#%% Plot Elements
patches = []
for i, n in tqdm(enumerate(elements)):
    p1 = nodeX[n[0]-1], nodeY[n[0]-1]
    p2 = nodeX[n[1]-1], nodeY[n[1]-1]
    p3 = nodeX[n[2]-1], nodeY[n[2]-1]
    p4 = nodeX[n[3]-1], nodeY[n[3]-1]
    pts = np.stack([p1, p2, p3, p4])

    polygon = Polygon(pts, True)
    patches.append(polygon)

p = PatchCollection(patches, alpha=0.4)
p.set_edgecolor('k')
p.set_linewidth(0.2)
p.set_facecolor('w')
plt.rc('font', family='Arial', size = 12)
fig, ax = plt.subplots()
ax.add_collection(p)
ax.set_aspect(1)
ax.autoscale(True)
plt.xlim(-110,710)
plt.ylim(-35,45)
plt.ylabel('')
plt.xlabel('')
ax.tick_params(axis=u'both', which=u'both',length=0)
plt.axis('off')
ax.tick_params(labelleft=False, labelbottom = False)
#fig.set_size_inches(5, 5)
fig.set_tight_layout('tight')
plt.show()
#%%
filename = fIN[:-4]+'_MESH_refine2_zoom.svg'
fig.savefig(fname = filename, bbox_inches='tight', format = 'svg', dpi = 600)
saveEMF(filename)

#%%
from subprocess import call
def saveEMF(filename):
    path_to_inkscape = "C:\Program Files\Inkscape\inkscape.exe"
    call([path_to_inkscape, "--file", filename,  "--export-emf",  filename[:-4]+".emf" ])


#%% Draw MassConcentration (Experimental)
massIdx = array.index('INIT_I_TRANS\n')
massNodes = np.zeros((nNodes,2))
n = 0
break_ = 0
while n < nNodes-1 and break_ != 1:
    print(n)
    for i in array[massIdx+1:]:
        if  'MAT_I_TRAN\n' in i:
            break_ = 1
            break
        try:
            mn = [float(f) for f in i.split()]
        except:
            print('Non-numeric character detected...')
            mn = [(f) for f in i.split()]
            mni = ['-' in m for m in mn]
            mnf = [m_.replace('-','') for m_ in mn]
            mn = [float(f) for f in mnf]
            s_ = np.array(mn)[mni.index(True)]
            s__ = np.array(mn)[mni.index(True)+1]
            s = np.linspace(s_, s__, s__+1 - s_)
            mn = np.insert(mn, mni.index(True)+1,s[1:-1])

        if len(mn) == 2:
            massNodes[n] = mn
            n += 1
        if len(mn) > 2:
            for mn_ in mn[1:]:
                massNodes[n,0] = mn[0]
                massNodes[n,1] = mn_
                n += 1
                
massNodes = [m.split() for m in massNodes]


massNodes_mass, massNodes_nodes = zip(*massNodes)
flat_list = [item for sublist in l for item in sublist]
 
massNodes = [float(f_) for f in massNodes for f_ in f]
