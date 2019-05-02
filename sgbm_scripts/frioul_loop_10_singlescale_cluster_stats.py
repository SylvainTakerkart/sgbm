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
n_permuts = 5000
# threshold used on the point statistics to define clusters; since the statistics we use is a z-score, we use a 1.645 threshold (eq to p<0.05)
#threshold_list = [1.282, 1.645, 2.326, 2.576, 2.878, 3.090, 3.290, 3.719]
#one_sided_p_values = [0.1, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
threshold_list = [1.645, 2.326, 2.576, 3.090]
one_sided_p_values = [0.05, 0.01, 0.005, 0.001]
cortex_scaling_list = ['no']
pointstat = "classifscorezprobafullcortex"
thresholdtype = "pointstat"
if thresholdtype == "permutedproba":
    threshold_list = 1. - np.array(one_sided_p_values)


hemispheres_list = ['rh','lh']
experiment = 'abide_jbhi_pits01'
for cortex_scaling in cortex_scaling_list:
    for threshold in threshold_list:
        for hem in hemispheres_list:
            for n_sl_points in n_sl_points_list:
                cmd = "frioul_batch 'python %s/10_singlescale_cluster_stats.py %s %s %s %d %d %.3f %s %s %s'" % (coderoot_dir, experiment, hem, graph_type, n_sl_points, n_permuts, threshold, cortex_scaling, pointstat, thresholdtype)
                a = commands.getoutput(cmd)
                print(cmd)

