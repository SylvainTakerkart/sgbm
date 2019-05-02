'''
jl version, STt 2015/03/03
- the differences with the previous version is that at each searchlight point, we get the permuted distribution of average classification scores,
which should be centered around 0.5, and transform it by z-scoring... those z-scores are the ones that will be used in the cluster
stats
- we still save the cdf values computed from the permuted distribution, but we don't use them
'''

import os.path as op
import os
import numpy as np
import scipy.stats as st
import sys
import joblib
import nibabel.gifti as ng
from sklearn.preprocessing import StandardScaler
import time

# read parameters and subjects lists
root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'
n_vertex_per_sphere = 50


def point_stat(experiment, hem, graph_type, graph_param, n_sl_points, n_permuts, n_folds, C=1e-2, diagnorm_option=True):


    analysis_dir = op.join(root_analysis_dir, experiment)

    ###############################
    # load permuted classif results
    ###############################
    permuted_skf_list = []
    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'permuted_classif_res', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
    res_path = op.join(res_dir,'%s.classif_res_%dpermuts_xval%02dfolds_diagnorm%s_C1e%d.jl' % (hem,n_permuts,n_folds,str(diagnorm_option).lower(),np.int(np.log10(C))))
    print('Reading permuted classif results from %s' % res_path)
    skf_scores_permuted = joblib.load(res_path)

    # compute avg classif scores across folds
    avg_skf_scores_permuted = skf_scores_permuted.mean(0)

    # read mask to exclude pole points from the calculation of the fullcortex permuted distrib of classif scores
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    masktex_path = op.join(spheresampling_dir,'%s.fsaverage.polemask.%dpoints.gii' % (hem,n_sl_points))
    masktex_gii = ng.read(masktex_path)
    masktex_data = masktex_gii.darrays[0].data
    fullcortex_inds = np.where(masktex_data==0)[0]
    # for each value of c, normalize the permuted distribution of average classif scores (computed over all the full cortex excluding the pole mask)
    point_cdf_fullcortex_permuted = np.zeros([n_sl_points,n_permuts])
    #avg_skf_scores_permuted = avg_skf_scores_permuted.round(2)
    permuted_avg_scores_fullcortex = avg_skf_scores_permuted[fullcortex_inds,:].flatten()
    n_permuts_fullcortex = len(permuted_avg_scores_fullcortex)
    sub_inds = np.random.permutation(n_permuts_fullcortex)[0:500000]
    #sub_inds = np.random.permutation(n_permuts_fullcortex)[0:200]
    permuted_avg_scores_fullcortex = permuted_avg_scores_fullcortex[sub_inds]
    n_permuts_fullcortex = len(permuted_avg_scores_fullcortex)
    #t1 = time.time()
    #scores_list = []
    #lenghts_list = []
    #for point_ind in range(n_sl_points):
    for point_ind in fullcortex_inds:
        t1 = time.time()
        print point_ind
        for perm_ind in range(n_permuts):
        #for perm_ind in range(3):
            #print point_ind, perm_ind
            this_score = avg_skf_scores_permuted[point_ind,perm_ind]
            point_cdf_fullcortex_permuted[point_ind,perm_ind] = \
                len(np.where(permuted_avg_scores_fullcortex>=this_score)[0])
        t2 = time.time()
        print t2-t1

    point_cdf_fullcortex_permuted = 1. - (point_cdf_fullcortex_permuted.astype(np.float) / n_permuts_fullcortex)


    # convert fullcortex cdf to z-scores, we call them zprobasfullcortex!
    # this is the single scale z-score introduced in the paper submitted to MEDIA 
    point_zproba_fullcortex_permuted = st.norm.ppf(point_cdf_fullcortex_permuted)



    data = np.copy(point_zproba_fullcortex_permuted)
    # deal with z_scores that have infinite value (because of the discrete trunkated permuted distribution used with classifscorezprobafullcortex
    inf_z_inds = (data == np.inf)
    real_max_z = np.max(data[~inf_z_inds])
    data[inf_z_inds] = real_max_z + 0.1
    inf_z_inds = (data == -np.inf)
    real_min_z = np.min(data[~inf_z_inds])
    data[inf_z_inds] = real_min_z - 0.1
    point_zproba_fullcortex_permuted = data


    '''
    # save summary stuff: probas (estimated with permutations), zscores and avg classif scores (all 3: only for true labels, i.e permutation number 0)
    point_probas = 1. - point_cdf_permuted[:,:,0]
    proba_path = op.join(res_dir,'%s.true_pointproba_zscore_avgclassifscore_%dpermuts.jl' % (hem,n_permuts))
    print 'Saving all results in %s' % proba_path
    joblib.dump([point_probas,point_zscore_permuted[:,:,0],avg_skf_scores_permuted[:,:,0],C_list],proba_path,compress=3)
    '''

    ################################
    # save unthresholded point stats (z-map)
    ################################
    # define output directory
    pointres_dir = op.join(analysis_dir,
                           'searchlight',
                           '%s_based' % graph_type,
                           'pointstats',
                           'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
    try: 
        os.makedirs(pointres_dir)
        print('Creating new directory: %s' % pointres_dir)
    except:
        print('Output directory is %s' % pointres_dir)

    # save zprobasfullcortex (without cortex-z-scaling) including the surrogate ones
    fullzproba_path = op.join(pointres_dir,'%s.pointstat_classifscorezprobafullcortex_cortexnoscaling_permuted.%s_%d.jl' % (hem,graph_type,graph_param))
    print 'Saving all point z-probas with fullcortex distribution in %s' % fullzproba_path
    joblib.dump(point_zproba_fullcortex_permuted,fullzproba_path,compress=3)

    # read texture of the pole mask, to be excluded from further analyses... (cluster stats & co)
    masktex_path = op.join(spheresampling_dir,'%s.fsaverage.polemask.%dpoints.gii' % (hem,n_sl_points))
    masktex_gii = ng.read(masktex_path)
    masktex_data = masktex_gii.darrays[0].data
    mask_inds = np.where(masktex_data)[0]
    # first texture: the raw pointstat map...
    clustertex1_path = op.join(pointres_dir,'%s.pointzscoremap_classifscorezprobafullcortex_cortexnoscaling.%s_%d.gii' % (hem,graph_type,graph_param))
    data = np.copy(point_zproba_fullcortex_permuted[:,0])
    data[mask_inds] = -5.
    tex1Data = np.tile(data,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darrays_list = [ ng.GiftiDataArray().from_array(tex1Data.astype(np.float32),intent) ]
    gii = ng.GiftiImage(darrays=darrays_list)
    ng.write(gii, clustertex1_path)



def main():
    '''hemispheres_list = ['rh']
    graph_type = 'radius'
    graph_params_list = [35]
    n_sl_points = 2500
    n_permuts = 2000
    '''
    
    args = sys.argv[1:]
    if len(args) < 5:
	print "Wrong number of arguments"
	#usage()
	sys.exit(2)
    else:
        experiment = args[0]
        hem = args[1]
        graph_type = args[2]
        graph_param = int(args[3])
        n_sl_points = int(args[4])
        n_permuts = int(args[5])


    n_folds = 10
    C=1.
    diagnorm_option=True
    '''for hem in hemispheres_list:
        for graph_param in graph_params_list:
            point_stat(graph_type, graph_param, hem, n_sl_points, n_permuts, n_folds)
    '''

    point_stat(experiment, hem, graph_type, graph_param, n_sl_points, n_permuts, n_folds, C=C, diagnorm_option=diagnorm_option)


if __name__ == "__main__":
    main()





