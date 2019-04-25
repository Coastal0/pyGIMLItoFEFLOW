# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:59:32 2017
Find linear gradient between two points and evaluate several other points alnog the line

@author: 264401k
"""
import numpy as np

eval_points_x = np.array([-100,0,600])
node_points_xy = np.array([[250,-24],[450,-25]])

grad_f = (node_points_xy[1,1]-node_points_xy[0,1])/(node_points_xy[1,0]-node_points_xy[0,0])
y_intercept = node_points_xy[0,1]-grad_f *node_points_xy[0,0]

eval_points_y = (grad_f * eval_points_x) + y_intercept
points = np.append(np.stack((eval_points_x,eval_points_y), axis = 0).T,node_points_xy,axis = 0)
print(points[points[:,0].argsort()])
