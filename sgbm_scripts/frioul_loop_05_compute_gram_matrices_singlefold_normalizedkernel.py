# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:45:11 2012

@author: auzias
"""


import commands
import os
import os.path as op
import numpy as np

coderoot_dir = '/riou/echange/auzias/sgbm_code/sgbm_abide_jbhi'



n_sl_points_list = [2500]
graph_type = 'radius'
n_folds = 10
graph_params_list = np.arange(90,28,-5)

hemispheres_list = ['lh', 'rh']
experiment = 'abide_jbhi_pits01'
for graph_param in graph_params_list:
    for hem in hemispheres_list:
        for n_sl_points in n_sl_points_list:
            for fold_ind in range(n_folds):
                cmd = "frioul_batch -w 168 'python %s/05_compute_gram_matrices_singlefold_normalizedkernel.py %s %s %s %d %d %d'" \
                    % (coderoot_dir, experiment, hem, graph_type, graph_param, n_sl_points, fold_ind)
                a = commands.getoutput(cmd)
                print(cmd)

