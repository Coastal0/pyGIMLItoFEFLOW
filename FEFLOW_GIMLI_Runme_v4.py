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

dataDict = w.loadData(fNames)

# %% Mesh and coordinate geometry
#if 'fBounds' not in locals():
fBounds = filedialog.askopenfilenames(title = "Select MATLAB boundary and nodes (as *.mat)",
                 filetypes = (("matlab files","*.mat"),("all files","*.*")))
    
hull = w.loadHull(fBounds[0])
bPolyMesh = w.fillPoly(dataDict, hull)  # Fill boundary mesh with nodes
w.checkMesh(bPolyMesh)  # Check each node has a value.
topoArray = w.getTopo(hull)  # Extract the topography from the concave hull

# %% Data geometry
dInterpDict = {}
for d in dataDict.keys():
    print(d)
    dInterpDict[d] = w.makeInterpVector(dataDict[d][0], bPolyMesh)  # Add data to the nodes
 
#%% 
#w.showDifference(dInterpDict, bPolyMesh)
# Custom Cmap
C = np.loadtxt(r"F:/testCmap3.txt")
ccmap = colors.ListedColormap(C/255)

plt.rc('font', family='Arial', size = 12)
# Load and Show data
for j in dataDict.keys():
    print(j)
    mesh = bPolyMesh
    data = dataDict[j][0]
    dataVec = w.makeInterpVector(data, mesh)

    # Create point data for plotting contours
    nData = pg.cellDataToPointData(mesh, dataVec)

    fig, ax = plt.subplots()
    if any([m in data.columns for m in ['MINIT','EXP_N']]):
        print('Plotting Mass Concentration data...')
        # dataVec = w.setAboveWT(mesh, dataVec, 1e-13)
        nData = pg.cellDataToPointData(mesh, dataVec)
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
                         cMap='jet',   cMin=None, cMax=None,
                         extend="both", logScale=True, ax = ax, 
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

    w.formatActiveFig(topoArray, bLim = -30, uLim = 50, rLim = 300, lLim = -50, title = None)
    #w.plotWells(topoArray, rLim = 600)
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
