import os.path as op
import os
import numpy as np
import sys
import joblib
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
import scipy.stats as st
import nibabel.gifti as ng

from pitslearn import get_clusters




root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'
n_vertex_per_sphere = 50
graph_param_list = np.arange(30,92,5)
#graph_param_list = np.arange(30,41,5)
#graph_param_list = np.array([50])
#def cluster_stat(graph_type, graph_param_list, hem, n_sl_points, n_folds, n_permuts):
#graph_param_list = np.arange(40,52,10)

def singlescale_cluster_stat(experiment, hem, graph_type, n_sl_points, n_permuts, threshold, cortex_scaling='no', pointstat='classifscorezprobafullcortex', thresholdtype='pointstat'):
    '''
    :param experiment:
    :param graph_type:
    :param hem:
    :param n_sl_points:
    :param n_permuts:
    :param threshold:
    :param cortex_scaling:
    :param pointstat:
    :param thresholdtype: by default, use a threshold on the pointstat; if 'permutedproba' is specified, use a threshold on the proba
    :return:
    '''



    analysis_dir = op.join(root_analysis_dir, experiment)
    '''
    graph_type = 'radius'
    graph_param = 60
    hem = 'rh'
    n_sl_points = 2500
    n_permuts = 2000
    n_folds = 10
    threshold = 1.645
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

    '''
    # deal with z_scores that have infinite value (because of the discrete trunkated permuted distribution used with classifscorezprobafullcortex
    inf_z_inds = (all_pointstat_permuted == np.inf)
    real_max_z = np.max(all_pointstat_permuted[~inf_z_inds])
    all_pointstat_permuted[inf_z_inds] = real_max_z + 0.1
    '''


    if thresholdtype == 'pointstat':
        allscales_tothreshold_permuted = all_pointstat_permuted
    elif thresholdtype == 'permutedproba':
        print "Depreciated option thresholdtype = permutedproba"
        exit()
        '''
        allscales_cdf_permuted = []
        for graph_param in graph_param_list:
            print graph_param
            res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'permuted_classif_res', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
            cdf_path = op.join(res_dir,'%s.pointcdf_classifscorenoscaling_%dpermuts.jl' % (hem,n_permuts))
            [point_cdf_permuted,C_list] = joblib.load(cdf_path)
            # reduce to only one value of C
            allscales_cdf_permuted.append(point_cdf_permuted[:,c_ind,:])
        allscales_tothreshold_permuted = np.array(allscales_cdf_permuted)
        print allscales_tothreshold_permuted.max()
        '''


    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    conn_path = op.join(spheresampling_dir,'%s.conn_matrix_without_mask.%dpoints.jl' % (hem,n_sl_points))
    connMatrix = joblib.load(conn_path)


    # scale by scale, with multiscale correction
    trueClusterMassesList = []
    trueClusterLabelsList = []
    allscales_maxClusterMassesList = np.zeros([len(graph_param_list),n_permuts])
    for g_ind, graph_param in enumerate(graph_param_list):
        thisscale_clusterMassesList = []
        thisscale_clusterMassesList = []
        for perm_ind in range(n_permuts):
        #for perm_ind in range(3):
            print g_ind, perm_ind
            # use allscales_tothreshold_permuted to give the clusters (clusterLabels), but do not use automatically the returned clusterMasses
            clusterLabels, clusterMasses = get_clusters(allscales_tothreshold_permuted[g_ind,:,perm_ind], connMatrix, threshold)
            # now, compute the clusterMasses using the all_pointstat_permuted variable! only if they're different!
            '''
            if thresholdtype != 'pointstat':
                # should loop over np.unique(clusterLabels) or something like that...
                print "Not implemented yet"
                exit()
            '''
            if perm_ind == 0:
                print clusterLabels, clusterMasses
                trueClusterLabelsList.append(np.array(clusterLabels))
                trueClusterMassesList.append(np.array(clusterMasses))
            if len(clusterMasses):
                thisscale_clusterMassesList.extend(clusterMasses)
                allscales_maxClusterMassesList[g_ind,perm_ind] = np.max(clusterMasses)
            #else: #useless, so commented out
            #    allscales_maxClusterMassesList[g_ind,perm_ind] = 0
    maxClusterStatList = allscales_maxClusterMassesList.max(0)

    # get rid of the zeros in case there are some...
    maxClusterStatList = maxClusterStatList[np.where(maxClusterStatList)[0]]


    # compute corrected probabilities of the true clusters at each scale using the empirical distribution of the max
    # cluster masses computed across scales!
    MScorrectedClusterProbasList = []
    for g_ind, graph_param in enumerate(graph_param_list):
        thisscale_correctedClusterProbas = []
        for mass in trueClusterMassesList[g_ind]:
            thisProba = float(len(np.where(maxClusterStatList>=mass)[0])) / len(maxClusterStatList)
            thisscale_correctedClusterProbas.append(thisProba)
        MScorrectedClusterProbasList.append(np.array(thisscale_correctedClusterProbas))

    # compute critical cluster mass
    correctedThreshold = 0.05
    criticalInd = int(np.floor(correctedThreshold*len(maxClusterStatList))) + 1
    sortedMaxClusterStatsList = list(maxClusterStatList)
    sortedMaxClusterStatsList.sort(reverse=True)
    criticalMass = sortedMaxClusterStatsList[criticalInd-1] # with -1 because indexing starts at zero

    #print np.unique(trueClusterLabelsList)
    #print len(trueClusterMassesList)

    print trueClusterMassesList
    print MScorrectedClusterProbasList

    print criticalMass

    significantClusterIndsList = []
    significantClusterProbasList = []
    significantClusterMassesList = []
    for g_ind, graph_param in enumerate(graph_param_list):
        thisscale_significantClusterInds = np.where(MScorrectedClusterProbasList[g_ind]<correctedThreshold)[0]
        thisscale_significantClusterProbas = MScorrectedClusterProbasList[g_ind][thisscale_significantClusterInds]
        thisscale_significantClusterMasses = trueClusterMassesList[g_ind][thisscale_significantClusterInds]
        significantClusterIndsList.append(thisscale_significantClusterInds)
        significantClusterProbasList.append(thisscale_significantClusterProbas)
        significantClusterMassesList.append(thisscale_significantClusterMasses)

    # read texture of the pole mask, to be excluded from further analyses... (cluster stats & co)
    masktex_path = op.join(spheresampling_dir,'%s.fsaverage.polemask.%dpoints.gii' % (hem,n_sl_points))
    masktex_gii = ng.read(masktex_path)
    masktex_data = masktex_gii.darrays[0].data
    mask_inds = np.where(masktex_data)[0]

    '''
    ################################
    # save unthresholded point stats (z-map)
    ################################
    pointres_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'singlescale_pointstats_%s_cortex%sscaling_thresh%s' % (pointstat,cortex_scaling,thresholdtype), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
    if not(op.exists(pointres_dir)):
        os.makedirs(pointres_dir)
        print('Creating new directory: %s' % pointres_dir)
    '''

    #####################
    # save cluster stats!
    #####################
    clusterres_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'singlescale_clusterstats_%s_cortex%sscaling_thresh%s' % (pointstat,cortex_scaling,thresholdtype), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))
    if not(op.exists(clusterres_dir)):
        os.makedirs(clusterres_dir)
        print('Creating new directory: %s' % clusterres_dir)

    # second multi-texture: constant value, the corrected cluster p value, within each cluster; nothing outside
    clustertex2_path = op.join(clusterres_dir,'%s.cluster_correctedpvalues_thresh%.3f_%s.gii' % (hem,threshold,graph_type))
    darrays2_list = []
    # third multi-texture: z-score of point proba displayed within each cluster; nothing outside
    clustertex3_path = op.join(clusterres_dir,'%s.cluster_zscores_thresh%.3f_%s.gii' % (hem,threshold,graph_type))
    darrays3_list = []
    # fourth multi-texture: the label of significant clusters within each cluster; nothing outside
    clustertex4_path = op.join(clusterres_dir,'%s.cluster_labels_thresh%.3f_%s.gii' % (hem,threshold,graph_type))
    darrays4_list = []




    nbrSignificantClustersList = []
    nbrTrueClustersList = []
    significantClusterSizesList = []
    clusterSizesList = []
    significantClusterLabelMapList = []
    for g_ind, graph_param in enumerate(graph_param_list):


        '''
        #load classif scores to create textures
        in_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'permuted_classif_res', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
        proba_path = op.join(in_dir,'%s.pointstat_%s_cortexnoscaling_%dpermuts.jl' % (hem,pointstat,n_permuts))
        #print 'Saving all results in %s' % proba_path
        avg_skf_scores_permuted = joblib.load(proba_path)
        avg_skf_scores_permuted = avg_skf_scores_permuted[:,0]
        '''
        # replace classif score by pointzstats
        #avg_skf_scores_permuted = allscales_tothreshold_permuted[g_ind,:,0]

        nbrTrueClustersList.append(len(trueClusterMassesList[g_ind]))
        nbrSignificantClustersList.append(len(significantClusterMassesList[g_ind]))
        correctedpvaluesClusterMap = 0.06 * np.ones(n_sl_points)
        #classifscoresClusterMap = 0.45 * np.ones(n_sl_points)
        zscoresClusterMap = -5. * np.ones(n_sl_points)
        significantClusterLabelMap = np.zeros(n_sl_points)
        significantLabel = 0
        thisscale_clusterSizes = []
        for label in range(nbrTrueClustersList[g_ind]):
            labelInds = np.where(trueClusterLabelsList[g_ind]==label)[0]
            thisscale_clusterSizes.append(len(labelInds))
            if MScorrectedClusterProbasList[g_ind][label] < correctedThreshold:
                correctedpvaluesClusterMap[labelInds] = MScorrectedClusterProbasList[g_ind][label]
                #classifscoresClusterMap[labelInds] = avg_skf_scores_permuted[labelInds]
                zscoresClusterMap[labelInds] = all_pointstat_permuted[g_ind,labelInds,0]
                significantLabel = significantLabel + 1
                significantClusterLabelMap[labelInds] = significantLabel

        clusterSizesList.append(np.array(thisscale_clusterSizes))
        significantClusterSizesList.append(clusterSizesList[g_ind][significantClusterIndsList[g_ind]])
        significantClusterLabelMapList.append(significantClusterLabelMap)




        #####################
        # save cluster stats!
        #####################

        # save infos and stats about the cluster thresholding process
        #info_path = op.join(clusterres_dir,'%s.cluster_infos_thresh%.3f_%s_%d.pck' % (hem,threshold,graph_type, graph_param))
        #f = open(info_path,'w')
        #pickle.dump([MScorrectedClusterProbasList[g_ind], significantClusterProbasList[g_ind], trueClusterMassesList[g_ind], significantClusterMassesList[g_ind], clusterSizes, signficantClusterSizes, nbrTrueClusters, nbrSignificantClusters, maxClusterStatList, criticalMass],f)
        #f.close()

        # second multi-texture: constant value, the corrected cluster p value, within each cluster; nothing outside
        #clustertex2_path = op.join(clusterres_dir,'%s.cluster_correctedpvalues_thresh%.3f_%s_%d.gii' % (hem,threshold,graph_type, graph_param))
        tex2Data = np.tile(correctedpvaluesClusterMap,[n_vertex_per_sphere,1]).T.flatten()
        intent = 0
        darrays2_list.append(ng.GiftiDataArray().from_array(tex2Data.astype(np.float32),intent))
        #gii = ng.GiftiImage(darrays=[darray])
        #print('Writing gifti texture: %s' % clustertex2_path)
        #ng.write(gii, clustertex2_path)

        # third multi-texture: z-score of point proba displayed within each cluster; nothing outside
        #clustertex3_path = op.join(clusterres_dir,'%s.cluster_zscores_thresh%.3f_%s_%d.gii' % (hem,threshold,graph_type, graph_param))
        tex3Data = np.tile(zscoresClusterMap,[n_vertex_per_sphere,1]).T.flatten()
        intent = 0
        darrays3_list.append(ng.GiftiDataArray().from_array(tex3Data.astype(np.float32),intent))
        #gii = ng.GiftiImage(darrays=[darray])
        #print('Writing gifti texture: %s' % clustertex3_path)
        #ng.write(gii, clustertex3_path)

        '''# second texture: classif score displayed within each cluster; nothing outside
        clustertex2_path = op.join(clusterres_dir,'%s.cluster_classifscores_thresh%.3f_%s_%d.gii' % (hem,threshold,graph_type, graph_param))
        tex2Data = np.tile(classifscoresClusterMap,[n_vertex_per_sphere,1]).T.flatten()
        intent = 0
        darray = ng.GiftiDataArray().from_array(tex2Data.astype(np.float32),intent)
        gii = ng.GiftiImage(darrays=[darray])
        print('Writing gifti texture: %s' % clustertex2_path)
        ng.write(gii, clustertex2_path)
        '''

        # fourth multi-texture: the label of significant clusters within each cluster; nothing outside
        #clustertex4_path = op.join(clusterres_dir,'%s.cluster_labels_thresh%.3f_%s_%d.gii' % (hem,threshold,graph_type, graph_param))
        tex4Data = np.tile(significantClusterLabelMap,[n_vertex_per_sphere,1]).T.flatten()
        intent = 0
        darrays4_list.append(ng.GiftiDataArray().from_array(tex4Data.astype(np.float32),intent))
        #gii = ng.GiftiImage(darrays=[darray])
        #print('Writing gifti texture: %s' % clustertex4_path)
        #ng.write(gii, clustertex4_path)


    #gii1 = ng.GiftiImage(darrays=darrays1_list)
    #ng.write(gii1, clustertex1_path)

    gii2 = ng.GiftiImage(darrays=darrays2_list)
    print('Writing gifti texture: %s' % clustertex2_path)
    ng.write(gii2, clustertex2_path)

    gii3 = ng.GiftiImage(darrays=darrays3_list)
    print('Writing gifti texture: %s' % clustertex3_path)
    ng.write(gii3, clustertex3_path)

    gii4 = ng.GiftiImage(darrays=darrays4_list)
    print('Writing gifti texture: %s' % clustertex4_path)
    ng.write(gii4, clustertex4_path)

    '''
    # save infos and stats about the cluster thresholding process
    info_path = op.join(clusterres_dir,'%s.cluster_infos_thresh%.3f_%s.jl' % (hem,threshold,graph_type))
    joblib.dump([MScorrectedClusterProbasList,
                 significantClusterProbasList,
                 trueClusterMassesList,
                 significantClusterMassesList,
                 clusterSizesList,
                 significantClusterSizesList,
                 nbrTrueClustersList,
                 nbrSignificantClustersList,
                 maxClusterStatList,
                 criticalMass],
                info_path,
                compress=3)
    '''

    # save infos and stats about the cluster thresholding process
    info_path = op.join(clusterres_dir,'%s.new_cluster_infos_thresh%.3f_%s.jl' % (hem,threshold,graph_type))
    print('Writing all kinds of info on results in %s' % info_path)
    joblib.dump([MScorrectedClusterProbasList,
                 significantClusterProbasList,
                 trueClusterMassesList,
                 significantClusterMassesList,
                 clusterSizesList,
                 significantClusterSizesList,
                 nbrTrueClustersList,
                 nbrSignificantClustersList,
                 maxClusterStatList,
                 allscales_maxClusterMassesList,
                 criticalMass,
                 significantClusterLabelMapList],
                info_path,
                compress=3)


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
        #graph_param = int(args[2])
        n_sl_points = int(args[3])
        n_permuts = int(args[4])
        threshold = float(args[5])
        cortex_scaling = args[6] #should be no or z
        pointstat = args[7] # should be "classifscorezscaling" or "classifscorezproba" or "classifscorezprobafullcortex"
        thresholdtype = args[8] # should be "pointstat" or "permutedproba"

    singlescale_cluster_stat(experiment,
                             hem,
                             graph_type,
                             n_sl_points,
                             n_permuts,
                             threshold,
                             cortex_scaling,
                             pointstat,
                             thresholdtype)


if __name__ == "__main__":
    main()

