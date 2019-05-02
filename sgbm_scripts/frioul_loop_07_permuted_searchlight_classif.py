# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:45:11 2012

@author: takerkart
"""


import commands
import os
import os.path as op
import numpy as np

coderoot_dir = '/riou/echange/auzias/sgbm_code/sgbm_abide_jbhi'



n_sl_points_list = [2500]
hemispheres_list = ['lh','rh']
graph_type = 'radius'
graph_params_list = np.arange(30,92,5)
n_permuts = 5000
experiment = 'abide_jbhi_pits01'

for graph_param in graph_params_list:
    for hem in hemispheres_list:
        for n_sl_points in n_sl_points_list:
            cmd = "frioul_batch 'python %s/07_permuted_searchlight_classif.py %s %s %s %d %d %d'" % (coderoot_dir, experiment, hem, graph_type, graph_param, n_sl_points, n_permuts)
            a = commands.getoutput(cmd)
            print(cmd)


