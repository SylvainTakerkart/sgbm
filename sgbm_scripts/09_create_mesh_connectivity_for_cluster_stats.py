import os.path as op
import os
import numpy as np
import sys
import joblib
from sklearn.neighbors import kneighbors_graph
import nibabel.gifti as ng


###############################
###############################
# gender right
###############################
###############################

# read parameters and subjects lists
root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'
experiment = 'abide_jbhi_pits01'
analysis_dir = op.join(root_analysis_dir, experiment)

n_sl_points = 2500

hemispheres_list = ['rh','lh']

for hem in hemispheres_list:

    # reading points where the searchlight will be performed
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hem,n_sl_points))
    [sl_points,sl_points_inds] = joblib.load(fibopoints_path)

    ###############################
    # compute neighbors for all points of the searchlight
    ###############################
    connMatrix = kneighbors_graph(sl_points, n_neighbors=6, include_self=False)
    connMatrix = connMatrix.todense()

# read mask to exclude pole points from the clusters
# read texture of the pole mask, to be excluded from further analyses... (cluster stats & co)
    masktex_path = op.join(spheresampling_dir,'%s.fsaverage.polemask.%dpoints.gii' % (hem,n_sl_points))
    masktex_gii = ng.read(masktex_path)
    masktex_data = masktex_gii.darrays[0].data
    mask_inds = np.where(masktex_data)[0]

    connMatrix[mask_inds,:] = 0
    connMatrix[:,mask_inds] = 0

    conn_path = op.join(spheresampling_dir,'%s.conn_matrix_without_mask.%dpoints.jl' % (hem,n_sl_points))
    joblib.dump(connMatrix.astype(bool),conn_path,compress=3)


