# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:47:22 2018

@author: 264401k
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class Node(object):
    name = ""
    time = []
    val = []
    def __init__(self):
        self.time = []
        self.val = []
    
def makeNode(name, time, val):
    node = Node()
    node.name = name
    node.time = time
    node.val = val
    
fIN1 = 'HydraulicHead_Tides_730d+.pow'
        
with open(fIN1, 'r+', newline = '\r\n' ) as f:
    powIN = f.read()
powIN = powIN.splitlines()

nodes = {};

i = 0;
while i < len(powIN) :
    line = powIN[i];
    if "#" in line :
        print(line); 
        name = powIN[i+2].split()[1];
        i = i + 4;
        line = powIN[i];        
        currentNode = Node();
        nodes[name] = currentNode;
        currentNode.name = name;
        
        while not "END" in line:
            split = line.split();
            time = float(split[0]);
            val = float(split[1]);
            currentNode.time.append(time);
            currentNode.val.append(val);
            i = i + 1;
            line = powIN[i];
    i += 1;
    
plt.figure();
for key, value in nodes.items():
#    plt.semilogy(nodes[key].time,nodes[key].val, label = key)
    plt.plot(nodes[key].time,nodes[key].val, label = key)
    plt.legend(loc = 4)
ax = plt.gca()
ax.grid(b = 1, which = 'major', linestyle = '-', color = 'black', alpha = 0.5)
ax.grid(b = 1, which = 'minor', linestyle = '--', color = 'grey', alpha = 0.5)
ax.set_xlabel("Time [d]")
ax.set_ylabel("Hydraulic Head [mAHD]")
ax.tick_params(axis='y', which='minor')
ax.tick_params(axis='x', which='minor')

xminorLocator = MultipleLocator(50)
ax.xaxis.set_minor_locator(xminorLocator)

yminorLocator = MultipleLocator(0.1)
ax.yaxis.set_minor_locator(yminorLocator)
ax.set_title("Hydraulic Head for Synthetic Tidal Model")

fig = plt.gcf()
fig.set_tight_layout('tight')
fig.savefig('HydHeadTides', dpi = 300)
#%%
fIN2 = 'MassConc_Tides_730d+.pow'
  
with open(fIN2, 'r+', newline = '\r\n' ) as f:
    powIN = f.read()
powIN = powIN.splitlines()

nodes = {};

i = 0;
while i < len(powIN) :
    line = powIN[i];
    if "#" in line :
        print(line); 
        name = powIN[i+2].split()[1];
        i = i + 4;
        line = powIN[i];        
        currentNode = Node();
        nodes[name] = currentNode;
        currentNode.name = name;
        
        while not "END" in line:
            split = line.split();
            time = float(split[0]);
            val = float(split[1]);
            currentNode.time.append(time);
            currentNode.val.append(val);
            i = i + 1;
            line = powIN[i];
    i += 1;
    
plt.figure();
for key, value in nodes.items():
    plt.plot(nodes[key].time,nodes[key].val, label = key)
    plt.legend(loc = 4)
ax = plt.gca()
ax.grid(b = 1, which = 'major', linestyle = '-', color = 'black', alpha = 0.5)
ax.grid(b = 1, which = 'minor', linestyle = '--', color = 'grey', alpha = 0.5)
ax.set_xlabel("Time [d]")
ax.set_ylabel("Mass Concentration [mg/L]")
ax.tick_params(axis='y', which='minor')
ax.tick_params(axis='x', which='minor')

xminorLocator = MultipleLocator(50)
ax.xaxis.set_minor_locator(xminorLocator)

yminorLocator = MultipleLocator(2500)
ax.yaxis.set_minor_locator(yminorLocator)
ax.set_title("Mass Concentration for Synthetic Tidal Model")

fig = plt.gcf()
fig.set_tight_layout('tight')
fig.savefig('MassConcTides', dpi = 300)
