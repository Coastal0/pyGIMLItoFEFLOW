# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:34:38 2019

@author: 264401k
"""

import sys
sys.path.append('C:\\Program Files\\DHI\2019\\FEFLOW 7.2\\bin64')
import ifm
print("FEFLOW IFM Kernal:", ifm.getKernelVersion())
from tkinter import filedialog, Tk

root = Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
fName = filedialog.askopenfilename(title = "Select FEM",
                filetypes = (("FEM file","*.fem"),("all files","*.*")))
print(fName)

doc = ifm.loadDocument(fName)
print(doc.getFileTitle)
nNodes = doc.getNumberOfNodes()
nEle = doc.getNumberOfElements()
nDims = doc.getNumberOfDimensions()
print("nNodes: {} \n nEle: {} \n NDims: {}".format(nNodes, nEle, nDims))

for obs in range(doc.getNumberOfValidObsPoints()):
    print(obs)
    print(doc.getObsLabel(obs))
    doc.getMassValueOfObsIdAtCurrentTime(obs)
    
# Assign scalar quantities of vector components to nodal distributions
def postTimeStep(doc):
    for nNode in range(0,nNodes):
        doc.setNodalRefDistrValue(rID_velX, nNode, doc.getResultsXVelocityValue(nNode))
        doc.setNodalRefDistrValue(rID_velY, nNode, doc.getResultsYVelocityValue(nNode))
        doc.setNodalRefDistrValue(rID_velZ, nNode, doc.getResultsZVelocityValue(nNode))

    #print "PostTimeStep at t=" + str(doc.getAbsoluteSimulationTime())

try:

    # Enable reference distribution recording
    bEnable = 1 # disable = 0, enable = 1

    # Create and enable distribution recording
    doc.createNodalRefDistr("Velocity_X")
    rID_velX = doc.getNodalRefDistrIdByName("Velocity_X")
    doc.enableNodalRefDistrRecording(rID_velX,bEnable)

    doc.createNodalRefDistr("Velocity_Y")
    rID_velY = doc.getNodalRefDistrIdByName("Velocity_Y")
    doc.enableNodalRefDistrRecording(rID_velY,bEnable)

    doc.createNodalRefDistr("Velocity_Z")
    rID_velZ = doc.getNodalRefDistrIdByName("Velocity_Z")
    doc.enableNodalRefDistrRecording(rID_velZ,bEnable)

    nNodes = doc.getNumberOfNodes()

except Exception as err:
    print>>sys.stderr,'Error: '+str(err)+'!'
    sys.exit(-1);