'''
jl version, STt 2015/03/03
the differences with the previous version is that we directly read the zscores computed at the point level;
the threshold used here is a z_value, to be chosen accordingly!
'''


import os.path as op
import os
import numpy as np
import sys
import joblib
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csgraph
import scipy.stats as st
import nibabel.gifti as ng

from pitslearn import get_clusters



root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'
n_vertex_per_sphere = 50
graph_param_list = np.arange(30,92,5)

#def cluster_stat(graph_type, graph_param_list, hem, n_sl_points, n_folds, n_permuts):

def multiscale_cluster_stat(experiment, hem, graph_type, n_sl_points, n_permuts, threshold, n_scales, cortex_scaling = 'no', pointstat='classifscorezproba'):

    analysis_dir = op.join(root_analysis_dir, experiment)
    '''
    graph_type = 'radius'
    graph_param = 50
    hem = 'rh'
    n_sl_points = 2500
    n_permuts = 2000
    n_folds = 10
    n_scales = 3
    threshold = 2.576
    '''
    '''
    ###############################
    # load permuted classif results
    ###############################
    all_pointstat_permuted = []
    for graph_param in graph_param_list:
        print graph_param
        res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'permuted_classif_res', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
        pointstat_path = op.join(res_dir,'%s.pointstat_%s_cortex%sscaling_%dpermuts.jl' % (hem,pointstat,cortex_scaling,n_permuts))
        [pointstat_permuted,C_list] = joblib.load(pointstat_path)
        # reduce to only one value of C
        all_pointstat_permuted.append(pointstat_permuted[:,c_ind,:])

    all_pointstat_permuted = np.array(all_pointstat_permuted)
    # deal with z_scores that have infinite value (because of the discrete trunkated permuted distribution used with classifscorezprobafullcortex
    inf_z_inds = (all_pointstat_permuted == np.inf)
    real_max_z = np.max(all_pointstat_permuted[~inf_z_inds])
    all_pointstat_permuted[inf_z_inds] = real_max_z + 0.1
    '''

    ###############################
    # load permuted classif results
    ###############################
    all_pointstat_permuted = []
    for graph_param in graph_param_list:
        pointres_dir = op.join(analysis_dir,
                               'searchlight',
                               '%s_based' % graph_type,
                               'pointstats',
                               'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
        pointstat_path = op.join(pointres_dir,'%s.pointstat_%s_cortex%sscaling_permuted.%s_%d.jl' % (hem,pointstat,cortex_scaling,graph_type,graph_param))
        print pointstat_path
        pointstat_permuted = joblib.load(pointstat_path)
        all_pointstat_permuted.append(pointstat_permuted)


    all_pointstat_permuted = np.array(all_pointstat_permuted)


    # transform cdf value to z-score
    #all_stats_permuted = st.norm.ppf(np.array(all_cdfs_permuted))

    '''max_size = 3
    all_means = permuted_skf_list.mean(1) - 0.5
    all_stds = permuted_skf_list.std(1)
    
    all_t_stats = np.divide(all_means, all_stds) / np.sqrt(n_folds)
    '''
    all_mean_stats = np.zeros([len(graph_param_list) - n_scales + 1,n_sl_points,n_permuts])
    for scale in range(len(graph_param_list) - n_scales + 1):
        all_mean_stats[scale,:,:] = all_pointstat_permuted[scale:(scale+n_scales),:,:].mean(0)
    
    all_max_stats = np.nanmax(all_mean_stats,0)
    print all_max_stats


    #### there is a bug here!!!!!!!! this overevaluates the optimal scale by 5mm systematically for n_scales impair!! (dunno what happens for n_scales pair)
    bestTrueScales = np.argmax(all_mean_stats[:,:,0],0) + 0.5 * (n_scales+1)
    #### it should probably be the following (but this needs to be validated fully)
    # bestTrueScales = np.argmax(all_mean_stats[:,:,0],0) + 0.5 * (n_scales-1)

    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    conn_path = op.join(spheresampling_dir,'%s.conn_matrix_without_mask.%dpoints.jl' % (hem,n_sl_points))
    connMatrix = joblib.load(conn_path)


    clusterMassesList = []
    maxClusterStatList = []
    for perm_ind in range(n_permuts):
    #for perm_ind in range(3):
        print perm_ind
        clusterLabels, clusterMasses = get_clusters(all_max_stats[:,perm_ind], connMatrix, threshold)
        if len(clusterMasses):
            if perm_ind == 0:
                trueClusterLabels = clusterLabels.copy()
                trueClusterMasses = np.array(clusterMasses)
            clusterMassesList.extend(clusterMasses)
            maxClusterStatList.append(np.max(clusterMasses))
    if 'trueClusterMasses' not in locals():
        trueClusterMasses = []
        trueClusterLabels = []

    # compute corrected probabilities of the cluster using the empirical distribution of the max cluster masses
    correctedClusterProbas = []
    for mass in trueClusterMasses:
        thisProba = float(len(np.where(maxClusterStatList>=mass)[0])) / len(maxClusterStatList)
        correctedClusterProbas.append(thisProba)
    correctedClusterProbas = np.array(correctedClusterProbas)
        

    # compute critical cluster mass
    print len(maxClusterStatList)
    correctedThreshold = 0.05
    criticalInd = int(np.floor(correctedThreshold*len(maxClusterStatList))) + 1
    print "criticalInd", criticalInd
    sortedMaxClusterStatsList = list(maxClusterStatList)
    sortedMaxClusterStatsList.sort(reverse=True)
    if len(sortedMaxClusterStatsList):
        criticalMass = sortedMaxClusterStatsList[criticalInd-1] # with -1 because indexing starts at zero
    else:
        criticalMass = np.inf

    print np.unique(trueClusterLabels)
    print len(trueClusterMasses)
    print trueClusterMasses
    print correctedClusterProbas

    print criticalMass

    significantClusterInds = np.where(correctedClusterProbas<correctedThreshold)[0]
    significantClusterProbas = correctedClusterProbas[significantClusterInds]
    if len(significantClusterInds):
        significantClusterMasses = trueClusterMasses[significantClusterInds]
    else:
        significantClusterMasses = []

    
    nbrTrueClusters = len(trueClusterMasses)
    nbrSignificantClusters = len(significantClusterMasses)
    correctedpvaluesClusterMap = 0.06 * np.ones(n_sl_points)
    maxzscoresClusterMap = -5. * np.ones(n_sl_points)
    #classifscoresClusterMap = 0.45 * np.ones(n_sl_points)
    significantClusterLabelMap = np.zeros(n_sl_points)
    significantClusterBestScaleMap = np.zeros(n_sl_points)
    signficantLabel = 0
    clusterSizes = []
    for label in range(nbrTrueClusters):
        labelInds = np.where(trueClusterLabels==label)[0]
        clusterSizes.append(len(labelInds))
        if correctedClusterProbas[label] < correctedThreshold:
            correctedpvaluesClusterMap[labelInds] = correctedClusterProbas[label]
            #classifscoresClusterMap[labelInds] = avg_skf_scores_permuted[labelInds]
            maxzscoresClusterMap[labelInds] = all_max_stats[labelInds,0]
            signficantLabel = signficantLabel + 1
            significantClusterLabelMap[labelInds] = signficantLabel
            significantClusterBestScaleMap[labelInds] = bestTrueScales[labelInds]

    clusterSizes = np.array(clusterSizes)
    significantClusterSizes = clusterSizes[significantClusterInds]

    ################################
    # save multiscale stats and cluster results
    ################################
    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'multiscale_clusterstats_%s_cortex%sscaling_threshpointstat' % (pointstat,cortex_scaling), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
    if not(op.exists(res_dir)):
        os.makedirs(res_dir)
        print('Creating new directory: %s' % res_dir)
    
    # read texture of the pole mask, to be excluded from further analyses... (cluster stats & co)
    masktex_path = op.join(spheresampling_dir,'%s.fsaverage.polemask.%dpoints.gii' % (hem,n_sl_points))
    masktex_gii = ng.read(masktex_path)
    masktex_data = masktex_gii.darrays[0].data
    mask_inds = np.where(masktex_data)[0]

    # first texture: z-score at each point
    clustertex1_path = op.join(res_dir,'%s.zscoremap.%s_allscales_max%dmeans.gii' % (hem,graph_type,n_scales))
    # DIFFERENCE with monoscale
    data = np.copy(all_max_stats[:,0])
    data[mask_inds] = -5.
    tex1Data = np.tile(data,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex1Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    ng.write(gii, clustertex1_path)
    
    # second texture: best-scale at each point
    clustertex2_path = op.join(res_dir,'%s.bestscale.%s_allscales_max%dmeans.gii' % (hem,graph_type,n_scales))
    # DIFFERENCE with monoscale
    data = np.copy(bestTrueScales)
    data[mask_inds] = -5.
    tex2Data = np.tile(data,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex2Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    ng.write(gii, clustertex2_path)
    
    '''
    #####################
    # save cluster stats!
    #####################
    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'multiscale_clusterstats_%s_cortex%sscaling_threshpointstat' % (pointstat,cortex_scaling), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
    if not(op.exists(res_dir)):
        os.makedirs(res_dir)
        print('Creating new directory: %s' % res_dir)
    '''
    
    # save infos and stats about the cluster thresholding process
    info_path = op.join(res_dir,'%s.cluster_infos_thresh%.3f_%s_allscales_max%dmeans.jl' % (hem,threshold,graph_type, n_scales))
    joblib.dump([correctedClusterProbas,
                 significantClusterProbas,
                 trueClusterMasses,
                 significantClusterMasses,
                 clusterSizes,
                 significantClusterSizes,
                 nbrTrueClusters,
                 nbrSignificantClusters,
                 maxClusterStatList,
                 criticalMass,
                 significantClusterBestScaleMap,
                 bestTrueScales,
                 significantClusterLabelMap],info_path,compress=3)

    # first texture: constant value, the corrected cluster p value, within each cluster; nothing outside
    clustertex1_path = op.join(res_dir,'%s.cluster_correctedpvalues_thresh%.3f_%s_allscales_max%dmeans.gii' % (hem,threshold,graph_type,n_scales))
    tex1Data = np.tile(correctedpvaluesClusterMap,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex1Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    print('Writing gifti texture: %s' % clustertex1_path)
    ng.write(gii, clustertex1_path)
    
    # second texture: z-score of point proba displayed within each cluster; nothing outside
    clustertex2_path = op.join(res_dir,'%s.cluster_zscores_thresh%.3f_%s_allscales_max%dmeans.gii' % (hem,threshold,graph_type,n_scales))
    tex2Data = np.tile(maxzscoresClusterMap,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex2Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    print('Writing gifti texture: %s' % clustertex2_path)
    ng.write(gii, clustertex2_path)

    # third texture: the label of significant clusters within each cluster; nothing outside
    clustertex3_path = op.join(res_dir,'%s.cluster_labels_thresh%.3f_%s_allscales_max%dmeans.gii' % (hem,threshold,graph_type,n_scales))
    tex3Data = np.tile(significantClusterLabelMap,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex3Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    print('Writing gifti texture: %s' % clustertex3_path)
    ng.write(gii, clustertex3_path)
    
    # fourth texture: z-score of point proba displayed within each cluster; nothing outside
    clustertex4_path = op.join(res_dir,'%s.cluster_bestscale_thresh%.3f_%s_allscales_max%dmeans.gii' % (hem,threshold,graph_type,n_scales))
    tex4Data = np.tile(significantClusterBestScaleMap,[n_vertex_per_sphere,1]).T.flatten()
    intent = 0
    darray = ng.GiftiDataArray().from_array(tex4Data.astype(np.float32),intent)
    gii = ng.GiftiImage(darrays=[darray])
    print('Writing gifti texture: %s' % clustertex4_path)
    ng.write(gii, clustertex4_path)




def main():
    args = sys.argv[1:]
    if len(args) < 6:
	print "Wrong number of arguments"
	#usage()
	sys.exit(2)
    else:
        experiment = args[0]
        hem = args[1]
        graph_type = args[2]
        n_sl_points = int(args[3])
        n_permuts = int(args[4])
        threshold = float(args[5])
        n_scales = int(args[6])
        cortex_scaling = args[7] #should be no or z
        pointstat = args[8] # should be "classifscorezscaling" ou "classifscorezproba"


    multiscale_cluster_stat(experiment, hem, graph_type, n_sl_points, n_permuts, threshold, n_scales, cortex_scaling, pointstat)


if __name__ == "__main__":
    main()



