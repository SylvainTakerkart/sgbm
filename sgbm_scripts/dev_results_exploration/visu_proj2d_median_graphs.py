'''
This file is the original file 12_etc, just modified for the pathnames etc.
It serves as a base for the development of other files in this directory!
'''



import os.path as op
import os
import numpy as np
import sys
import joblib
import networkx as nx
import matplotlib.pyplot as plt

import pickle
import nibabel.gifti as ng
import scipy.spatial.distance as sd

from pitslearn import get_center_of_cluster

from stereo_projection import graph_stereo_projection

#import anatomist.direct.api as ana
#from soma import aims

'''
root_analysis_dir = '/riou/work/crise/takerkart/pitslearn'
pits_data_dir = '/riou/work/crise/takerkart/pitslearn/pits_database/oasis'
oasis_fs_dir = '/riou/work/meca/oasis/FS_database_OASIS'
'''

root_analysis_dir = '/hpc/scalp/auzias/sgbm/sgbm_results'
input_data_dir = '/hpc/scalp/auzias/abide_db/abide_pits'
oasis_fs_dir = '/hpc/scalp/auzias/abide_db/freesurfer_db/'
graph_param_list = np.arange(30,92,5)
n_xval_folds = 10
diagnorm_option = True


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def geometricmediansubject_influencepitszones_multiscale_stats(experiment, hemisphere, graph_type, n_sl_points, n_permuts, threshold, n_scales, cortex_scaling = 'no', pointstat='classifscorezproba'):

    ##########################################
    # 0. get cluster and everything associated
    ##########################################

    analysis_dir = op.join(root_analysis_dir, experiment)

    # read golden spiral points
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hemisphere,n_sl_points))
    [sl_spherepoints_coords,sl_points_inds] = joblib.load(fibopoints_path)

    if experiment.find('asymmetry') >= 0:
        xhemi_reference_hem = hemisphere

    if experiment.find('asymmetry') >= 0:
        template_subject = 'fsaverage_sym'
    else:
        template_subject = 'fsaverage'

    sampleslist_path = op.join(analysis_dir,'samples_list.jl')
    [y, samples_subjects_list, samples_hem_list, samples_gender_list] = joblib.load(sampleslist_path)
    y = np.array(y)

    classes = np.unique(y)
    classes_names = []
    if experiment.find('asymmetry') >= 0:
        for c in classes:
            classes_names.append(samples_hem_list[np.where(y==c)[0][0]])
    else:
        for c in classes:
            classes_names.append(samples_gender_list[np.where(y==c)[0][0]])

    print classes_names


    params_path = op.join(analysis_dir,'pits_extraction_parameters.jl')
    [alpha, an, dn, r, area, param_string] = joblib.load(params_path)


    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'multiscale_clusterstats_%s_cortex%sscaling_threshpointstat' % (pointstat,cortex_scaling), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))

    # load infos and stats about the cluster thresholding process
    info_path = op.join(res_dir,'%s.cluster_infos_thresh%.3f_%s_allscales_max%dmeans.jl' % (hemisphere,threshold,graph_type, n_scales))
    [correctedClusterProbas,
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
     significantClusterLabelMap] = joblib.load(info_path)

    out_dir = op.join(res_dir,'%s.cluster_infos_thresh%.3f_%s_allscales_max%dmeans' % (hemisphere,threshold,graph_type, n_scales))
    if not(op.exists(out_dir)):
        os.makedirs(out_dir)
        print('Creating new directory: %s' % out_dir)
    print out_dir



    clusterLabels = np.unique(significantClusterLabelMap[significantClusterLabelMap>0])
    clustercenters_coords_list = []
    clustercenters_inds_list = []
    for label_ind, label in enumerate(clusterLabels):
        ###############################
        # 1. get center point of cluter
        ###############################

        # get center of the cluster
        thisclustercenter_ind = get_center_of_cluster(significantClusterLabelMap, label, sl_spherepoints_coords)

        # get all points of the cluster
        thiscluster_inds = np.where(significantClusterLabelMap==label)[0]

        #########################
        # 2. get associated scale
        #########################

        # get mean scale of all point of the cluster
        mean_scale_ind = np.round(bestTrueScales[thiscluster_inds].mean())
        print mean_scale_ind

        # get corresponding radius = representative radius of what's happening in this cluster! (strong assumption)
        graph_param = graph_param_list[mean_scale_ind]

        print sl_spherepoints_coords[thisclustercenter_ind]


        #################################################################################
        # 3. compute gram matrix (mean of gram matrix computed for each fold in the xval)
        #################################################################################

        gram_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'gram_matrices_xval', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)

        xval_dir = op.join(analysis_dir,'xval')
        xval_path = op.join(xval_dir,'stratified_%02dfold_inds.jl' % n_xval_folds)
        # load all train and test indices
        [train_inds_list,test_inds_list] = joblib.load(xval_path)


        mean_gram_matrix_allfolds = np.zeros([len(y),len(y)])
        counts_gram_matrix_allfolds = np.zeros([len(y),len(y)])

        for fold_ind in range(n_xval_folds):
        #for fold_ind in range(6,7):
            print 'Loading gram matrices for fold %d of %d' % (fold_ind+1,n_xval_folds)
            gram_path = op.join(gram_dir,'K_%s_sigmasheur_fold%02dof%02d.jl' % (hemisphere,fold_ind+1,n_xval_folds))
            [allK_train,allK_test,allK_testdiagonal,sigma_coords_vals,sigma_depth_vals] = joblib.load(gram_path)

            train_inds = train_inds_list[fold_ind]
            test_inds = test_inds_list[fold_ind]
            traintrain_inds = np.ix_(train_inds,train_inds)
            testtrain_inds = np.ix_(test_inds,train_inds)

            K_train = allK_train[thisclustercenter_ind,:,:,0]
            if diagnorm_option:
                # normalisation of the diagonal terms
                diag = np.diagonal(K_train)
                if 0. in diag:
                    zero_inds = (diag == 0)
                    if np.sum(zero_inds) == len(diag):
                        # if all elements of the diagonal are zero, put a fake 1 value
                        diag[:] = 1.
                    else:
                        # else put the min value in place of the zeros...
                        diag[zero_inds] = np.min(diag[~zero_inds])
                #print diag
                diagmat = np.tile(diag,[K_train.shape[0],1])
                normmat = np.multiply(1./np.sqrt(diagmat),1./np.sqrt(diagmat).T)
                K_train = np.multiply(K_train,normmat)
            # extract test "gram" matrix
            K_test = allK_test[thisclustercenter_ind,:,:,0]
            if diagnorm_option:
                # normalisation of the diagonal terms
                Ktestdiagonal = allK_testdiagonal[thisclustercenter_ind,:,0]
                # in case there is a zero on the diagonal, trick it a bit...
                #print Ktestdiagonal
                if 0. in Ktestdiagonal:
                    zero_inds = (Ktestdiagonal == 0)
                    if np.sum(zero_inds) == len(Ktestdiagonal):
                        Ktestdiagonal[:] = 1.
                    else:
                        Ktestdiagonal[zero_inds] = np.min(Ktestdiagonal[~zero_inds])
                diagmat = np.tile(diag,[K_test.shape[0],1])
                testdiagmat = np.tile(Ktestdiagonal,[K_train.shape[0],1]).T
                normmat = np.multiply(1./np.sqrt(diagmat),1./np.sqrt(testdiagmat))
                K_test = np.multiply(K_test,normmat)

            mean_gram_matrix_allfolds[traintrain_inds] = mean_gram_matrix_allfolds[traintrain_inds] + K_train
            counts_gram_matrix_allfolds[traintrain_inds] = counts_gram_matrix_allfolds[traintrain_inds] + 1
            mean_gram_matrix_allfolds[testtrain_inds] = mean_gram_matrix_allfolds[testtrain_inds] + K_test
            counts_gram_matrix_allfolds[testtrain_inds] = counts_gram_matrix_allfolds[testtrain_inds] + 1



        #######################################
        # 4. find median subjects in each class
        #######################################
        n_subj_per_class = 3
        median_sample_inds_list = []
        for c in classes:
            c_inds = np.where(y==c)[0]
            class_K = mean_gram_matrix_allfolds[np.ix_(c_inds,c_inds)]
            sum_K = class_K.sum(1)
            sum_K_sorted = np.argsort(sum_K)
            # the geometric medians are the elements with largest similarities in this class, i.e the last elements)
            argmax_val = sum_K_sorted[-n_subj_per_class:]
            median_sample_ind = c_inds[argmax_val]
            median_sample_inds_list.append(median_sample_ind)

        print median_sample_inds_list


        #############################################
        # 5. project their graphs to 2D planar graphs
        #############################################

        # load local graphs
        print 'Loading local graphs'
        localgraphs_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'local_graphs', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
        localgraphs_path = op.join(localgraphs_dir,'%s.localgraphslist_point%04d.jl' % (hemisphere,thisclustercenter_ind))
        localgraphs_list = joblib.load(localgraphs_path)

        # for each class, extract and represent graph for each median subject
        for c_ind, c in enumerate(classes):
            for sample_ind in median_sample_inds_list[c_ind]:

                g = localgraphs_list[sample_ind]
                subject = samples_subjects_list[sample_ind]
                x_2d = graph_stereo_projection(g.X,sl_spherepoints_coords[thisclustercenter_ind])

                n_parcels = x_2d.shape[0]
                orig_graph = nx.empty_graph(n_parcels)
                for i in range(n_parcels):
                    for j in range(i):
                        if g.A[i,j] != 0:
                            orig_graph.add_edge(i, j)
                            orig_graph.graph['act'] = g.D
                            orig_graph.graph['geo'] = x_2d[:,:2]

                print x_2d.shape, x_2d[:2,:].shape

                geo = orig_graph.graph['geo']
                x = geo[:, 0]
                y = geo[:, 1]
                plt.figure()

                # Plotting the edges
                print n_parcels, x.shape, y.shape
                for n, nbrs in orig_graph.adjacency_iter():
                    for nbr, eattr in nbrs.items():
                        print n,nbr
                        plt.plot([x[n], x[nbr]], [y[n], y[nbr]], 'k', alpha=0.5, zorder=1)

                # Plotting the nodes
                plt.scatter(x, y, c=orig_graph.graph['act'], s=300, cmap='jet', zorder=5)

                plt.colorbar()

    plt.show()






"""
    # 6. plot them with networkx


    # 7. save images





























    analysis_dir = op.join(root_analysis_dir, experiment)

    writer = aims.Writer()

    # read golden spiral points
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hemisphere,n_sl_points))
    [sl_spherepoints_coords,sl_points_inds] = joblib.load(fibopoints_path)

    '''
    subjectslist_path = op.join(analysis_dir,'subjects_list.pck')
    f = open(subjectslist_path,'r')
    [subjects_list, xhemi_reference_hem, hemispheres_list] = pickle.load(f)
    f.close()
    '''

    if experiment.find('asymmetry') >= 0:
        xhemi_reference_hem = hemisphere

    if experiment.find('asymmetry') >= 0:
        template_subject = 'fsaverage_sym'
    else:
        template_subject = 'fsaverage'

    sampleslist_path = op.join(analysis_dir,'samples_list.jl')
    [y, samples_subjects_list, samples_hem_list, samples_gender_list] = joblib.load(sampleslist_path)
    y = np.array(y)

    classes = np.unique(y)
    classes_names = []
    if experiment.find('asymmetry') >= 0:
        for c in classes:
            classes_names.append(samples_hem_list[np.where(y==c)[0][0]])
    else:
        for c in classes:
            classes_names.append(samples_gender_list[np.where(y==c)[0][0]])

    print classes_names


    '''
    # create labels y
    # create input data X with the fullbrain graphs, and labels y 
    y = []
    for subject in subjects_list:
        for group_ind, tmp_hem in enumerate(hemispheres_list):
            y.append(group_ind)
    '''

    #n_samples = len(y)
    #y = np.array(y)

    params_path = op.join(analysis_dir,'pits_extraction_parameters.jl')
    [alpha, an, dn, r, area, param_string] = joblib.load(params_path)


    #def zone_of_influence(point_ind, graph_param, experiment, graph_type, hem, n_sl_points, n_permuts, threshold, cortex_scaling='no', pointstat='classifscorezproba', thresholdtype='pointstat'):
    '''Given the center point of a significant cluster and the associated scale (called graph_param),
    estimates the zone around this point that contributed to the classification,
    by looking at the sulcal basins of all subjects'''

    #hem_letter = hem[0].upper()

    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'multiscale_clusterstats_%s_cortex%sscaling_threshpointstat' % (pointstat,cortex_scaling), 'sl_%s_points_%dpermuts' % (n_sl_points,n_permuts))

    # load infos and stats about the cluster thresholding process
    info_path = op.join(res_dir,'%s.cluster_infos_thresh%.3f_%s_allscales_max%dmeans.jl' % (hemisphere,threshold,graph_type, n_scales))
    [correctedClusterProbas,
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
     significantClusterLabelMap] = joblib.load(info_path)

    out_dir = op.join(res_dir,'%s.cluster_infos_thresh%.3f_%s_allscales_max%dmeans' % (hemisphere,threshold,graph_type, n_scales))
    if not(op.exists(out_dir)):
        os.makedirs(out_dir)
        print('Creating new directory: %s' % out_dir)
   


    ############################################################
    # taken from 11_zone_of_influence (merge attempt of 11 & 12)
    ############################################################

    basins_tex_list = []
    spherereg_coords_list = []
    resampledbasins_tex_list = []
    resampleddilatedpits_tex_list = []
    for subj_ind, subject in enumerate(samples_subjects_list):
        if experiment.find('asymmetry') >= 0:
            hem = samples_hem_list[subj_ind]
        else:
            hem = hemisphere
        #hem = hemispheres_list[y[subj_ind]]
        #hem_letter = hem[0].upper()
        print subject
        #this_graph =  all_localgraphs_list[thisclustercenter_ind][subj_ind]

        # coordinates on the sphere from op.join(input_data_dir,'pits_db','FS_database_OASIS',subject,'surf','%s.sphere.reg.gii' % hem)
        # defined in 01_create_full_hem_pitsgraphs.py, used in 02_searchlight_compute_local_graphs.py
        #this_graph.X
        # we can get the inds of the pits by matching these coords to the subj.sphere.reg.gii mesh, and the basins textures from
        # multitex_path = op.join(input_data_dir,'lucile','pits_database','oasis',subject,'dpfMap',alpha_string,params_string2,'%s_%s_area%dFilteredTexture.gii' % (params_string1,hem_letter,area))


        '''
        # read multitex texture to get basins, pits and basins areato get spatial adjacancy of the pits (one pit = one basin)
        multitex_path = op.join(pits_data_dir,subject,'dpfMap','%s.%s_area%dFilteredTexture.gii' % (hem,param_string,area))
        multitex_gii = ng.read(multitex_path)
        basins_tex = multitex_gii.darrays[0].data
        basins_tex_list.append(basins_tex)
        '''
        # get basins texture (-1 in poles; everything 0 or above is a real basin with one pit)
        basins_tex_gii = ng.read(op.join(input_data_dir,subject,'%s_%s_%s_basinsTexture.gii' % (subject, hem, param_string)))
        basins_tex = basins_tex_gii.darrays[0].data
        # read triangulated spherical mesh and get coordinates of all vertices! this changes depending of the experiment!
        if experiment.find('asymmetry') >= 0 and hem == 'lh':
            spherereg_path = op.join(oasis_fs_dir,subject,'surf','%s.fsaverage_sym.sphere.reg.gii' % xhemi_reference_hem)
        elif experiment.find('asymmetry') >= 0  and  hem == 'rh':
            spherereg_path = op.join(oasis_fs_dir,subject,'xhemi','surf','%s.fsaverage_sym.sphere.reg.gii' % xhemi_reference_hem)
        else:
            spherereg_path = op.join(oasis_fs_dir,subject,'surf','%s.sphere.reg.gii' % hem)
        spherereg_gii = ng.read(spherereg_path)
        spherereg_coords = spherereg_gii.darrays[0].data
        spherereg_coords_list.append(spherereg_coords)

        '''
        n_local_pits = len(this_graph.D)
        # compute the distances from the nodes of the graphs (pits) to all vertices of the mesh
        nodes_distances = sd.cdist(this_graph.X,mesh_coords/100).squeeze()
        # find the vertex indices of the pits
        pits_inds_nativemesh = np.argmin(nodes_distances,axis=1)
        # get the corresponding basins labels
        basins_inds = basins_tex[pits_inds_nativemesh]
        '''

        # now, switch to the reinterpolated textures!!
        if experiment.find('asymmetry') >= 0:
            resampledtex_path = op.join(input_data_dir,subject,'dpfMap','xhemi.%s.%s.%s_area%dFilteredTexture.gii' % (hem,template_subject,param_string,area))
            dilatedpitstex_path = op.join(input_data_dir,subject,'dpfMap','xhemi.%s.%s.%s_area50Filtered_dilatedpits.gii' % (hem,template_subject,param_string))
        else:
            resampledtex_path = op.join(input_data_dir,subject,'dpfMap','%s.%s.%s_area%dFilteredTexture.gii' % (hem,template_subject,param_string,area))
            dilatedpitstex_path = op.join(input_data_dir,subject,'dpfMap','%s.%s.%s_area50Filtered_dilatedpits.gii' % (hem,template_subject,param_string))
        resampledtex_gii = ng.read(resampledtex_path)
        resampledbasins_tex = resampledtex_gii.darrays[0].data
        resampledbasins_tex_list.append(resampledbasins_tex)
        resampleddilatedpits_gii = ng.read(dilatedpitstex_path)
        resampleddilatedpits_tex = resampleddilatedpits_gii.darrays[0].data
        resampleddilatedpits_tex_list.append(resampleddilatedpits_tex)


    clusterLabels = np.unique(significantClusterLabelMap[significantClusterLabelMap>0])
    clustercenters_coords_list = []
    clustercenters_inds_list = []
    for label_ind, label in enumerate(clusterLabels):
        # get center of the cluster
        thisclustercenter_ind = get_center_of_cluster(significantClusterLabelMap, label, sl_spherepoints_coords)

        # get all points of the cluster
        thiscluster_inds = np.where(significantClusterLabelMap==label)[0]

        # get mean scale of all point of the cluster
        mean_scale_ind = np.round(bestTrueScales[thiscluster_inds].mean())
        print mean_scale_ind

        # get corresponding radius = representative radius of what's happening in this cluster! (strong assumption)
        graph_param = graph_param_list[mean_scale_ind]

        ##########################
        # load all local graphs
        ##########################
        print 'Loading all local graphs'
        localgraphs_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'local_graphs', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
        localgraphs_path = op.join(localgraphs_dir,'all_localgraphs_list_%s.jl' % (hemisphere))
        [all_localgraphs_list,y,sl_points_inds] = joblib.load(localgraphs_path)


        #thisclustercenter_ind = 2300

        gram_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'gram_matrices', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)

        xval_dir = op.join(analysis_dir,'xval')
        xval_path = op.join(xval_dir,'stratified_%02dfold_inds.jl' % n_xval_folds)
        # load all train and test indices
        [train_inds_list,test_inds_list] = joblib.load(xval_path)


        mean_gram_matrix_allfolds = np.zeros([len(y),len(y)])
        counts_gram_matrix_allfolds = np.zeros([len(y),len(y)])

        for fold_ind in range(n_xval_folds):
        #for fold_ind in range(6,7):
            print 'Loading gram matrices for fold %d of %d' % (fold_ind+1,n_xval_folds)
            gram_path = op.join(gram_dir,'K_%s_sigmasheur_fold%02dof%02d.jl' % (hemisphere,fold_ind+1,n_xval_folds))
            [allK_train,allK_test,allK_testdiagonal,sigma_coords_vals,sigma_depth_vals] = joblib.load(gram_path)

            train_inds = train_inds_list[fold_ind]
            test_inds = test_inds_list[fold_ind]
            traintrain_inds = np.ix_(train_inds,train_inds)
            testtrain_inds = np.ix_(test_inds,train_inds)

            K_train = allK_train[thisclustercenter_ind,:,:,0]
            if diagnorm_option:
                # normalisation of the diagonal terms
                diag = np.diagonal(K_train)
                if 0. in diag:
                    zero_inds = (diag == 0)
                    if np.sum(zero_inds) == len(diag):
                        # if all elements of the diagonal are zero, put a fake 1 value
                        diag[:] = 1.
                    else:
                        # else put the min value in place of the zeros...
                        diag[zero_inds] = np.min(diag[~zero_inds])
                #print diag
                diagmat = np.tile(diag,[K_train.shape[0],1])
                normmat = np.multiply(1./np.sqrt(diagmat),1./np.sqrt(diagmat).T)
                K_train = np.multiply(K_train,normmat)
            # extract test "gram" matrix
            K_test = allK_test[thisclustercenter_ind,:,:,0]
            if diagnorm_option:
                # normalisation of the diagonal terms
                Ktestdiagonal = allK_testdiagonal[thisclustercenter_ind,:,0]
                # in case there is a zero on the diagonal, trick it a bit...
                #print Ktestdiagonal
                if 0. in Ktestdiagonal:
                    zero_inds = (Ktestdiagonal == 0)
                    if np.sum(zero_inds) == len(Ktestdiagonal):
                        Ktestdiagonal[:] = 1.
                    else:
                        Ktestdiagonal[zero_inds] = np.min(Ktestdiagonal[~zero_inds])
                diagmat = np.tile(diag,[K_test.shape[0],1])
                testdiagmat = np.tile(Ktestdiagonal,[K_train.shape[0],1]).T
                normmat = np.multiply(1./np.sqrt(diagmat),1./np.sqrt(testdiagmat))
                K_test = np.multiply(K_test,normmat)

            mean_gram_matrix_allfolds[traintrain_inds] = mean_gram_matrix_allfolds[traintrain_inds] + K_train
            counts_gram_matrix_allfolds[traintrain_inds] = counts_gram_matrix_allfolds[traintrain_inds] + 1
            mean_gram_matrix_allfolds[testtrain_inds] = mean_gram_matrix_allfolds[testtrain_inds] + K_test
            counts_gram_matrix_allfolds[testtrain_inds] = counts_gram_matrix_allfolds[testtrain_inds] + 1



        # first code: wrong because argmin instead of argmax!!!
        # median_sample_inds_list = []
        # for c in classes:
        #     c_inds = np.where(y==c)[0]
        #     class_K = mean_gram_matrix_allfolds[np.ix_(c_inds,c_inds)]
        #     sum_K = class_K.sum(1)
        #     # reject outliers before taking the geometric median
        #     min_val = np.min(reject_outliers(sum_K,5))
        #     # the geometric median is the element with min similarity with all others in this class
        #     argmin_val = np.where(sum_K==min_val)[0]
        #     median_sample_ind = c_inds[argmin_val]
        #     median_sample_inds_list.append(median_sample_ind)
        #     #median_samples_list.append(subjects_list[median_sample_ind])

        median_sample_inds_list = []
        for c in classes:
            c_inds = np.where(y==c)[0]
            class_K = mean_gram_matrix_allfolds[np.ix_(c_inds,c_inds)]
            sum_K = class_K.sum(1)
            argmax_val = np.argsort(sum_K)[-1]
            # the geometric median is the element with MAX similarity with all others in this class
            median_sample_ind = c_inds[argmax_val]
            median_sample_inds_list.append(median_sample_ind)
            #median_samples_list.append(subjects_list[median_sample_ind])



        graphs_inflated_windows_list = []
        texturedmesh_obj_list = []
        parcels_white_windows_list = []
        parcelstexturedmesh_obj_list = []
        mult_coeff = 1

        '''# one assertions for compatibility purposes; need to be checked!!!!
        allsubjects_list = list(subjects_list)
        allsubjects_list.extend(allsubjects_list)
        '''
        # for each class, extract and represent graph for each median subject
        for class_ind, sample_ind in enumerate(median_sample_inds_list):
            
            if experiment.find('asymmetry') >= 0:
                hem = samples_hem_list[median_sample_inds_list[class_ind]]
            else:
                hem = hemisphere
            #hem_letter = hem[0].upper()


            g = all_localgraphs_list[thisclustercenter_ind][sample_ind]
            subject = samples_subjects_list[sample_ind]
            #subject_ind = subject_ind + 1

            # read multitex texture to get basins, pits and basins areato get spatial adjacancy of the pits (one pit = one basin)
            multitex_path = op.join(input_data_dir,subject,'dpfMap','%s.%s_area%dFilteredTexture.gii' % (hem,param_string,area))
            multitex_gii = ng.read(multitex_path)

            # get pits texture (0 everywhere except single vertex with one where the pits are)
            pits_tex = multitex_gii.darrays[1].data
            pits_inds = np.where(pits_tex)[0]
            basins_tex = multitex_gii.darrays[0].data

            # read triangulated spherical mesh and get coordinates of all vertices! this changes depending of the experiment!
            # these are only used to go from the graph nodes coordinates to the vertex indices on the native meshes
            if experiment.find('asymmetry') >= 0 and hem == 'lh':
                mesh_path = op.join(oasis_fs_dir,subject,'surf','%s.fsaverage_sym.sphere.reg.gii' % xhemi_reference_hem)
            elif experiment.find('asymmetry') >= 0 and hem == 'rh':
                mesh_path = op.join(oasis_fs_dir,subject,'xhemi','surf','%s.fsaverage_sym.sphere.reg.gii' % xhemi_reference_hem)
            else:
                mesh_path = op.join(oasis_fs_dir,subject,'surf','%s.sphere.reg.gii' % hem)
            mesh_gii = ng.read(mesh_path)
            mesh_coords = mesh_gii.darrays[0].data
            pits_3dcoords = mesh_coords[pits_inds,:]
            n_pits = len(pits_inds)

            nodes_pits_distances = sd.cdist(g.X*100,pits_3dcoords)
            # which pit corresponds to each node of the graph?
            nodes_pitsinds = nodes_pits_distances.argmin(1)
            # which vertex of the mesh corresponds to each node of the graph?
            nodes_meshinds = pits_inds[nodes_pitsinds]


            # create other basins tex with only the concerned basins for this graph
            maskedbasins_tex = np.copy(basins_tex)
            maskedbasins_tex[:] = 0.
            for node_ind, node in enumerate(nodes_meshinds):
                maskedbasins_tex[basins_tex == basins_tex[node]] = node_ind + 1.
            # save this texture in a temp file
            #localbasins_path = op.join(analysis_dir,'tmptex.gii')
            if experiment.find('asymmetry') >= 0:
                localbasins_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_local_basins.gii' % (label,graph_type, graph_param,samples_hem_list[median_sample_inds_list[class_ind]],subject))
            else:
                localbasins_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_local_basins.gii' % (label,graph_type, graph_param,samples_gender_list[median_sample_inds_list[class_ind]],subject))
            intent = 0
            darrays = ng.GiftiDataArray().from_array(maskedbasins_tex.astype(np.float32),intent)
            gii = ng.GiftiImage(darrays=[darrays])
            ng.write(gii, localbasins_path)
            # # load it back!! (yes, I know, this is ridiculous!!)
            # maskedbasins_obj = a.loadObject(localbasins_path)
            # maskedbasins_obj.setPalette('Blue-Red-Fusion')
            # maskedbasins_obj.glSetTexRGBInterpolation(True)
            # maskedbasins_obj.notifyObservers()

            # read inflated mesh
            whitemesh_path = op.join(oasis_fs_dir,subject,'surf','%s.white.gii' % hem)

            # read inflated mesh
            inflatedmesh_path = op.join(oasis_fs_dir,subject,'surf','%s.inflated.gii' % hem)
            inflatedmesh_gii = ng.read(inflatedmesh_path)
            inflatedmesh_coords = inflatedmesh_gii.darrays[0].data
            nodes_3dcoords = inflatedmesh_coords[nodes_meshinds]

            #xform = inflatedmesh_gii.darrays[0].

            X = nodes_3dcoords

            #X = g.X
            A = g.A
            D = g.D
            D = ( D - D.min() ) / ( D.max() - D.min() )


            # graphs_inflated_windows_list.append(a.createWindow( '3D' ))
            # w = graphs_inflated_windows_list[-1]
            # parcels_white_windows_list.append(a.createWindow( '3D' ))
            # w2 = parcels_white_windows_list[-1]
            # #w = a.createWindow( '3D' )
            # #myEPI = a.loadObject( 'data/uagrabbr_10_func05_0020.nii')
            # #w.addObjects(myEPI)

            # manually add links to visualization window
            graph_texture = []
            surf = aims.SurfaceGenerator;
            meshLinks = aims.AimsSurfaceTriangle()
            # edge_list = []
            for i in range(A.shape[0]):
                for j in range(A.shape[0]):
                    if A[i,j]:
                        edge = surf.cylinder(X[i,:]*mult_coeff,X[j,:]*mult_coeff,1,1,100,1)
                        aims.SurfaceManip.meshMerge(meshLinks, edge)
                        graph_texture.extend(-0.1*np.ones(np.array(edge.vertex()).shape[0]))
                        # edge_list.append(a.toAObject(surf.cylinder(X[i,:]*mult_coeff,X[j,:]*mult_coeff,1,1,100,1)))
                        # w.addObjects(edge_list[-1])


            # manually add nodes
            #node_list = []
            for i in range(A.shape[0]):
                node = aims.SurfaceGenerator.sphere( X[i,:]*mult_coeff, 3, 10 )
                graph_texture.extend(D[i]*np.ones(np.array(node.vertex()).shape[0]))
                aims.SurfaceManip.meshMerge(meshLinks, node)
                # node_list.append(a.toAObject(aims.SurfaceGenerator.sphere( X[i,:]*mult_coeff, 3, 10 )))
                # #assign color to each node
                # #material=a.Material(diffuse=[D[i], D[i], D[i],1])
                # color_ind = int(round(D[i]*15))
                # material=a.Material(diffuse=[colormap[color_ind,0], colormap[color_ind,1], colormap[color_ind,2],1])
                # node_list[-1].setMaterial(material)
                # w.addObjects(node_list[-1])
                # #aims.SurfaceManip.meshMerge(meshLinks, node)

            graph_texture = np.array(graph_texture)

            # # get rid of the cursor
            # a.execute( 'WindowConfig', windows=[w], cursor_visibility=0 )
            # # set the proper point of view
            # w.camera( view_quaternion=winf[ 'view_quaternion' ], zoom=winf['zoom'], slice_quaternion=winf['slice_quaternion'], observer_position=winf['observer_position'], force_redraw=True )

            # color = 0.8
            # opacity = 0.8
            # inflatedmesh_obj = a.loadObject(inflatedmesh_path)
            # whitemesh_obj = a.loadObject(whitemesh_path)
            # #material=a.Material(diffuse=[color,color,color,opacity])
            # #mesh_obj.setMaterial(material)
            # #w.addObjects(mesh_obj) 

            # #curvtex_path = op.join(root_dir,'..','pits_db','FS_database_OASIS',subject,'surf','%s.curv.gii' % hem)
            # #curvtex_obj = a.loadObject(curvtex_path)
            # basins_obj = a.loadObject(multitex_path)
            # basins_obj.setPalette('pastel-256')
            # basins_obj.glSetTexRGBInterpolation(True)
            # basins_obj.notifyObservers()
            # texturedmesh_obj_list.append(a.fusionObjects([inflatedmesh_obj , basins_obj], "FusionTexSurfMethod"))
            # texturedmesh_obj = texturedmesh_obj_list[-1]
            # material=a.Material(diffuse=[color,color,color,opacity])
            # texturedmesh_obj.setMaterial(material)
            # w.addObjects(texturedmesh_obj)

            # parcelstexturedmesh_obj_list.append(a.fusionObjects([whitemesh_obj , maskedbasins_obj], "FusionTexSurfMethod"))
            # parcelstexturedmesh_obj = parcelstexturedmesh_obj_list[-1]
            # #material=a.Material(diffuse=[color,color,color,opacity])
            # #parcelstexturedmesh_obj.setMaterial(material)
            # w2.addObjects(parcelstexturedmesh_obj)


            if experiment.find('asymmetry') >= 0:
                graphmesh_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_graph_mesh.gii' % (label,graph_type,graph_param,samples_hem_list[median_sample_inds_list[class_ind]],subject))
                graphtex_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_graph_texture.gii' % (label,graph_type,graph_param,samples_hem_list[median_sample_inds_list[class_ind]],subject))
            else:
                graphmesh_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_graph_mesh.gii' % (label,graph_type,graph_param,samples_gender_list[median_sample_inds_list[class_ind]],subject))
                graphtex_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_%s_graph_texture.gii' % (label,graph_type,graph_param,samples_gender_list[median_sample_inds_list[class_ind]],subject))

            writer.write(meshLinks, graphmesh_path)
            intent = 0
            darrays = ng.GiftiDataArray().from_array(graph_texture.astype(np.float32),intent)
            gii = ng.GiftiImage(darrays=[darrays])
            ng.write(gii, graphtex_path)



        '''
        clusterLabels = np.unique(significantClusterLabelMap[significantClusterLabelMap>0])
        clustercenters_coords_list = []
        clustercenters_inds_list = []
        clusterLabels = np.unique(significantClusterLabelMap[significantClusterLabelMap>0])
        clustercenters_coords_list = []
        clustercenters_inds_list = []
        for label_ind, label in enumerate(clusterLabels):
            # get center of the cluster
            thisclustercenter_ind = get_center_of_cluster(significantClusterLabelMap, label, sl_spherepoints_coords)
        for label_ind, label in enumerate(clusterLabels):
            # get center of the cluster
            thisclustercenter_ind = get_center_of_cluster(significantClusterLabelMap, label, sl_spherepoints_coords)

            # get all points of the cluster
            thiscluster_inds = np.where(significantClusterLabelMap==label)[0]

            # get mean scale of all point of the cluster
            mean_scale_ind = np.round(bestTrueScales[thiscluster_inds].mean())
            print mean_scale_ind

            # get corresponding radius = representative radius of what's happening in this cluster! (strong assumption)
            graph_param = graph_param_list[mean_scale_ind]

            ##########################
            # load all local graphs
            ##########################
            print 'Loading all local graphs'
            localgraphs_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'local_graphs', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
            localgraphs_path = op.join(localgraphs_dir,'all_localgraphs_list_%s.jl' % (hemisphere))
            [all_localgraphs_list,y,sl_points_inds] = joblib.load(localgraphs_path)
        '''

        #subj_ind = 28
        #subject = subjects_list[subj_ind]

        #thisclustercenter_ind = 2300


        #####################
        #no need to reload all textures for each cluster!!!
        #####################

        influence_tex = np.zeros(163842)
        classinfluence_tex_list = []
        pits_influence_tex = np.zeros(163842)
        pits_classinfluence_tex_list = []
        for c in classes:
            classinfluence_tex_list.append(np.zeros(163842))
            pits_classinfluence_tex_list.append(np.zeros(163842))
        for subj_ind, subject in enumerate(samples_subjects_list):
            print subject
            this_graph =  all_localgraphs_list[thisclustercenter_ind][subj_ind]

            # coordinates on the sphere from op.join(input_data_dir,'pits_db','FS_database_OASIS',subject,'surf','%s.sphere.reg.gii' % hem)
            # defined in 01_create_full_hem_pitsgraphs.py, used in 02_searchlight_compute_local_graphs.py
            this_graph.X
            # we can get the inds of the pits by matching these coords to the subj.sphere.reg.gii mesh, and the basins textures from
            # multitex_path = op.join(input_data_dir,'lucile','pits_database','oasis',subject,'dpfMap',alpha_string,params_string2,'%s_%s_area%dFilteredTexture.gii' % (params_string,hem_letter,area))


            '''
            # read multitex texture to get basins, pits and basins areato get spatial adjacancy of the pits (one pit = one basin)
            multitex_path = op.join(input_data_dir,'lucile','pits_database','oasis',subject,'dpfMap',alpha_string,params_string2,'%s_%s_area%dFilteredTexture.gii' % (params_string,hem_letter,area))
            multitex_gii = ng.read(multitex_path)
            basins_tex = multitex_gii.darrays[0].data
            '''
            basins_tex = basins_tex_list[subj_ind]
            '''
            # read triangulated spherical mesh and get coordinates of all vertices
            mesh_path = op.join(oasis_fs_dir,subject,'surf','%s.sphere.reg.gii' % hem)
            mesh_gii = ng.read(mesh_path)
            mesh_coords = mesh_gii.darrays[0].data
            '''
            spherereg_coords = spherereg_coords_list[subj_ind]

            n_local_pits = len(this_graph.D)
            # compute the distances from the nodes of the graphs (pits) to all vertices of the mesh
            nodes_distances = sd.cdist(this_graph.X,spherereg_coords/100).squeeze()
            # find the vertex indices of the pits
            pits_inds_nativemesh = np.argmin(nodes_distances,axis=1)
            # get the corresponding basins labels
            basins_inds = basins_tex[pits_inds_nativemesh]

            '''
            # now, switch to the reinterpolated textures!!
            resampledtex_path = op.join(input_data_dir,'lucile','pits_database','oasis',subject,'dpfMap',alpha_string,params_string2,'%s_%s_area%dFilteredTexture_nn.reg.gii' % (params_string,hem_letter,area))
            resampledtex_gii = ng.read(resampledtex_path)
            resampledbasins_tex = resampledtex_gii.darrays[0].data
            '''
            resampledbasins_tex = resampledbasins_tex_list[subj_ind]
            resampleddilatedpits_tex = resampleddilatedpits_tex_list[subj_ind]

            for basin in basins_inds:
                influence_tex[resampledbasins_tex==basin] += 1.
                classinfluence_tex_list[y[subj_ind]][resampledbasins_tex==basin] += 1.
                pits_influence_tex[resampledbasins_tex==basin] += resampleddilatedpits_tex[resampledbasins_tex==basin]
                pits_classinfluence_tex_list[y[subj_ind]][resampledbasins_tex==basin] += resampleddilatedpits_tex[resampledbasins_tex==basin]
            print subject

        zonetex_path = op.join(out_dir,'cluster%02d_%s_%d_allclasses_influencezone.gii' % (label, graph_type, graph_param))
        intent = 0
        darrays = ng.GiftiDataArray().from_array(influence_tex.astype(np.float32),intent)
        gii = ng.GiftiImage(darrays=[darrays])
        ng.write(gii, zonetex_path)
        for c_ind,c in enumerate(classes):
            zonetex_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_influencezone.gii' % (label, graph_type, graph_param,classes_names[c_ind]))
            intent = 0
            darrays = ng.GiftiDataArray().from_array(classinfluence_tex_list[c_ind].astype(np.float32),intent)
            gii = ng.GiftiImage(darrays=[darrays])
            ng.write(gii, zonetex_path)
        pits_zonetex_path = op.join(out_dir,'cluster%02d_%s_%d_allclasses_influencepits.gii' % (label, graph_type, graph_param))
        intent = 0
        darrays = ng.GiftiDataArray().from_array(pits_influence_tex.astype(np.float32),intent)
        gii = ng.GiftiImage(darrays=[darrays])
        ng.write(gii, pits_zonetex_path)
        for c_ind,c in enumerate(classes):
            pits_zonetex_path = op.join(out_dir,'cluster%02d_%s_%d_%sclass_influencepits.gii' % (label, graph_type, graph_param,classes_names[c_ind]))
            intent = 0
            darrays = ng.GiftiDataArray().from_array(pits_classinfluence_tex_list[c_ind].astype(np.float32),intent)
            gii = ng.GiftiImage(darrays=[darrays])
            ng.write(gii, pits_zonetex_path)
        #b = input()

    #return graphs_inflated_windows_list, parcels_white_windows_list

"""



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
        pointstat = args[8] # should be "classifscorezscaling" ou "classifscorezproba" ou "classifscorezprobafullcortex"


    geometricmediansubject_influencepitszones_multiscale_stats(experiment, hem, graph_type, n_sl_points, n_permuts, threshold, n_scales, cortex_scaling, pointstat)


if __name__ == "__main__":
    main()
