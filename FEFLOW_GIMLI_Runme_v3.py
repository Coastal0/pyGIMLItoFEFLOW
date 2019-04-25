# -*- coding: utf-8 -*-
import workbook as w
import matplotlib.pyplot as plt
from matplotlib import colors,ticker
import pandas as pd
import pygimli as pg
import pygimli.mplviewer as pgmpl
import numpy as np
from tkinter import filedialog, Tk

root = Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
fNames = filedialog.askopenfilenames(title = "Select FEFLOW Output (Mass/etc)",
                filetypes = (("DAT file","*.dat"),("all files","*.*")))
print(fNames)

dataDict, data, coords = w.loadData(fNames)

# %% Mesh and coordinate geometry
#if 'fBounds' not in locals():
fBounds = filedialog.askopenfilenames(title = "Select MATLAB boundary and nodes (as *.mat)",
                 filetypes = (("matlab files","*.mat"),("all files","*.*")))
    
hull, bPoly, nodes = w.loadHull(fBounds[0])
#if len(nodes) != len(data):
#    raise ValueError('Length of data and number of nodes in boundary are not equal.')
bPolyMesh = w.fillPoly(bPoly, coords, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull

# %% Data geometry
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    data = dataDict[d][0]
#    if 'SINIT' in data:
#        print('Mass and Saturation present')
#        data['dataCol'] = data['MINIT'] * data['SINIT']
#    elif 'MINIT' in data and 'SINIT' not in data:
#        print('Mass present')
#        data['dataCol'] = data['MINIT']
#    elif 'MINIT' or 'SINIT' not in data:
#        print('Neither mass or solute present in data')
#        data['dataCol'] = data.iloc[:,-1]
#    
    dInterpDict[d], times = w.makeInterpVector(data, bPolyMesh)  # Add data to the nodes
 
#%% Formatting
def plotWells(rLim = 400):
    # Plot Wells
    plt.sca(plt.gcf().get_axes()[0])
    wellFin = r"G:\PROJECTS\PAPER -- Quinns Rocks Urban Expansion & Saline Water Interface\Data\FEFLOW_Supplementary\SIMWell_Coords.dat"
    WellCoords = pd.read_csv(wellFin, sep = '\t')
    
    def getElev(array, value):
      val = array[abs(array[:,0] - value).argmin()][1]
      return val
    
    for well in WellCoords.iterrows():
    #    print(well)
        if well[1].X < rLim:
            plt.plot([well[1]['X'],well[1]['X']],
                     [getElev(topoArray,well[1]['X']),well[1]['Y']], 'k')
            plt.annotate(s = well[1]['LABEL'], 
                         xy = [well[1]['X'],8+getElev(topoArray,well[1]['X'])],
                         ha = 'center', fontsize = 12)
        else:
            print('Well outside viewing area')
    plt.plot([min(topoArray[:,0]), max(topoArray[:,0])],[0,0], 'k--')

def formatActiveFig(lLim = -50, rLim = 400, uLim = 50, bLim = -30, title = True, plotwells = True):
    from os import path
    fig = plt.gcf()
    fig.set_size_inches(12, 3.5)
    
    plt.sca(fig.get_axes()[0])
    plt.plot(topoArray[:, 0], topoArray[:, 1], 'k')
    ax = plt.gca()
    ax.tick_params(which = 'both', direction = 'in')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlim(left=lLim, right=rLim)
    ax.set_ylim(top=uLim, bottom=bLim)
    ax.set_aspect(aspect=1)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (mASL)')
    if title == True:
        ax.set_title(path.basename(fNames[j]), loc = 'left', fontsize = 12)
    fig.set_tight_layout('tight')
    
    if plotwells == 1:
        plotWells(rLim = rLim)

    
def plotPESTNodes():
    nodeFile = "K:\FePest_Nodes_Output#3.dat"
#        nodeFile = "K:\QR_Quad_20-07-1993_FePEST_NoMass_\NODES.dat"
    PESTNodes = pd.read_csv(nodeFile, sep = '\t')
    plt.scatter(PESTNodes.X, PESTNodes.Y, marker = '+', c = 'white', s = 1)

def plotPressure():
    pressureFile = r"G:\PROJECTS\EXP_ABST -- Random Seawater fields\test.lin"
    pFile = np.loadtxt(pressureFile, skiprows = 1, comments="END")    
    plt.plot(pFile[:,0],pFile[:,1], 'k--')

def setAboveWT(mesh, dataVec, val):
    for c in mesh.cells():
        if c.center()[1] > 0:
            dataVec[c.id()] = val
    return dataVec

def fixZeroes(data, dataVec,val):
    dataVec[dataVec == 0] = val
    return dataVec

# Custom Cmap
C = np.loadtxt(r"F:/testCmap3.txt")
ccmap = colors.ListedColormap(C/255)

plt.rc('font', family='Arial', size = 12)
#%% 
# Plot difference - Uncomment as req'd
#w.showDifference(dInterpDict, bPolyMesh)

# Load and Show data
for j in dInterpDict.keys():
    print(j)
    mesh = bPolyMesh
    dataVec = dInterpDict[j]

    # Create point data for plotting contours
    nData = pg.cellDataToPointData(mesh, dataVec)

    fig, ax = plt.subplots()
    if 'MINIT' in data.columns:
        print('Plotting Mass Concentration data...')
        dataVec = setAboveWT(mesh, dataVec, 1e-13)
        _ , cbar = pg.show(mesh, dataVec/1000, label="C [g/l]", ax = ax, 
                           cMap=ccmap, cMin=0.358, cMax=35.8, extend="both", 
                           logScale=True, colorBar = True, orientation="vertical")
        pg.mplviewer.drawField(ax, mesh, nData/1000, levels=[1.0,10.0,30.0], 
                               fillContour=False, colors = 'k', 
                               linewidths=1, alpha=1, linestyles = 'solid')
        plt.sca(plt.gcf().get_axes()[0])
        cbar.ax.xaxis.set_ticklabels(np.ceil(cbar.get_ticks()).astype(int))
    
    elif 'COND' in data.columns:
        print('Plotting Hydraulic Conductivity data...')
        im, cbar = pg.show(mesh, dataVec, label="K (m/day)",
                         cMap='gist_ncar',   cMin=None, cMax=None,
                         extend="both", logScale=False, ax = ax, 
                         colorBar=True, orientation="vertical")
        cbar.ax.xaxis.set_ticklabels(np.ceil(cbar.get_ticks()).astype(int))
        
    elif 'PORO' in data.columns:
        print('Plotting porosity data... ')
        im, cbar = pg.show(mesh, dataVec, label=r"Poro. $\phi$", ax = ax,
                         cMap='viridis',  colorBar=True, cMin=None, cMax=None,
                         extend="both", logScale=False, orientation="vertical")
    else:
        print('Plotting undefined variable...')
        im, cb = pg.show(ax = ax, mesh = mesh, data = dataVec, 
                         colorBar=True, orientation="vertical")        

    formatActiveFig(bLim = -30, uLim = 50, rLim = 600, lLim = -50,
                    title = False, plotwells = True)
#    plotPESTNodes()
#    plotPressure()
    filename = fNames[j][:-4]+'.png'
    fig.savefig(fname = filename, bbox_inches='tight', format = 'png', dpi = 600)

#%% DrawStreams (Experimental)
plotStreamLines = 0

def add_arrow(line, position=None, direction='right', size=15, color=None, xlim = None):
    """
    add an arrow to a line.
    Modified from: https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        if xlim is not None:
            position = xdata.mean()
        else:
            position = xdata[xdata < 400].mean()
 
   # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )
    
if plotStreamLines == 1:
    streamlineFID = filedialog.askopenfilenames(title = "Select streamline output",
                filetypes = (("DAT file","*.dat"),("all files","*.*")))
    for s in streamlineFID:
        streamlines = pd.read_csv(s, sep = '\t')
        sl = streamlines
        n = len(sl.FMAX.unique())
        sortedUniques = np.sort(sl.FMAX.unique())
        sortedUniquesMax = np.max(sortedUniques)
        if ('Fresh' or 'fresh') in s:
            print('fresh streamlines')
            colors_ = plt.cm.Blues(np.linspace(0,1,n))
            direction = 'left'

        elif ('Saline' or 'saline') in s:
            print('saline streamlines')
            colors_ = plt.cm.Reds(np.linspace(0,1,n))
            direction = 'right'
        else:
            colors_ = plt.cm.gist_yarg(np.linspace(0,1,n))
    
#        fig, ax = plt.subplots()
        for name, grp in sl.groupby(sl.PathLine):
        #    print(name)
            if len(grp.PathLine) > 100:
                i = np.where(grp.FMAX.unique() == sortedUniques)[0][0]
                line = ax.plot(grp.X,grp.Y, color = colors_[i], linewidth = 0.5)
                add_arrow(line[0], xlim = 400, direction=direction)

    formatActiveFig(bLim = -30, uLim = 60, rLim = 950, lLim = -50, title = False, plotwells = False)
    filename2 = fNames[j][:-4]+'StreamLines.png'
    fig.savefig(fname = filename2, bbox_inches='tight', format = 'png', dpi = 600)
    
#%% Plot Hydraulic Head Shapefile Contours
plotHydraulicHeadShape = 0
if plotHydraulicHeadShape == 1:
    import shapefile as shp
    shpFileIn = r"K:\QR_Quad_Simple_FEPEST_Homogeneous_200md_K0p1-15deg_HEAD_Contours.shp"
    cntrShpf = shp.Reader(shpFileIn)
    plt.figure()
    for shape in cntrShpf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y)
