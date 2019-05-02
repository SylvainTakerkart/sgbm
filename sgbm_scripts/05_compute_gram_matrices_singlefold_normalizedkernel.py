import os.path as op
import os
import numpy as np
import sys
import joblib
from sklearn.cross_validation import StratifiedKFold
import scipy.spatial.distance as sd
from pitskernel import sxdnewkernel


root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'


def searchlight_compute_gram_matrices_singlefold_normalizedkernel(experiment, hem, graph_type, graph_param, n_sl_points, fold_ind, n_folds=10, subkernels_option=False):
    analysis_dir = op.join(root_analysis_dir, experiment)

    if subkernels_option:
        n_kernels = 7
    else:
        n_kernels = 1

    if graph_type == 'nn':
        n_neighbors = graph_param
    elif graph_type == 'radius':
        radius = graph_param
    elif graph_type == 'conn':
        connlength = graph_param


    # reading points where the searchlight will be performed
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hem,n_sl_points))
    [sl_points,sl_points_inds] = joblib.load(fibopoints_path)

    localgraphs_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'local_graphs', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)

    #########################################################
    # Load the reference xval computed on the original labels
    #########################################################
    xval_dir = op.join(analysis_dir,'xval')
    xval_path = op.join(xval_dir,'stratified_%02dfold_inds.jl' % n_folds)
    print('Loading cross-validation indices from %s' % xval_path)
    [train_inds_list,test_inds_list] = joblib.load(xval_path)


    ###################################
    # Compute gram matrix for this fold
    ###################################


    # work on this fold only!
    train_inds = train_inds_list[fold_ind]
    test_inds = test_inds_list[fold_ind]
    print 'Working on fold %d of %d' % (fold_ind+1,n_folds)

    sigma_depth_vals = []
    sigma_coords_vals = []
    allK_train = np.zeros([n_sl_points,len(train_inds),len(train_inds),n_kernels])
    allK_test = np.zeros([n_sl_points,len(test_inds),len(train_inds),n_kernels])
    allK_testdiagonal = np.zeros([n_sl_points,len(test_inds),n_kernels])
    for (center_ind, point_ind) in enumerate(sl_points_inds):
        ##
        # Loading local graphs
        ##
        localgraphs_path = op.join(localgraphs_dir,'%s.localgraphslist_point%04d.jl' % (hem, point_ind))
        print('Loading %s' % localgraphs_path)
        localgraphs_list = joblib.load(localgraphs_path)
        ##
        # Estimating sigma values
        ##
        print 'Estimating sigma values for point %d of %d' % (center_ind+1,n_sl_points)
        for i_ind, i in enumerate(train_inds):
            if i_ind == 0:
                all_depth = localgraphs_list[i].D
                all_coords = localgraphs_list[i].X
            else:
                all_depth = np.vstack([all_depth, localgraphs_list[i].D])
                all_coords = np.vstack([all_coords, localgraphs_list[i].X])
        dists_depth = sd.cdist(all_depth,all_depth)
        sigma_depth_vals.append(np.median(dists_depth.flatten()))
        dists_coords = sd.cdist(all_coords,all_coords)
        sigma_coords_vals.append(np.median(dists_coords.flatten()))
        ##
        # Computing gram matrix
        ##
        print 'Computing gram matrix for point %d of %d' % (center_ind+1,n_sl_points)
        sigma_coords = sigma_coords_vals[-1]
        sigma_depth = sigma_depth_vals[-1]
        kernel = sxdnewkernel(x_sigma = sigma_coords, d_sigma = sigma_depth, subkernels=subkernels_option)
        print '     searchlight location %d of %d' % (point_ind,n_sl_points)
        for i_ind,i in enumerate(train_inds):
            for j_ind,j in enumerate(train_inds[:i_ind+1]):
                allK_train[point_ind,i_ind,j_ind,:] = kernel.evaluate(localgraphs_list[i],localgraphs_list[j])
                allK_train[point_ind,j_ind,i_ind,:] = allK_train[point_ind,i_ind,j_ind,:]
                #print localgraphs_list[i].D
                #print localgraphs_list[j].D
                #print i,j,allK_train[point_ind,i_ind,j_ind,:]
            for j_ind, j in enumerate(test_inds):
                #print i,j
                #print localgraphs_list[i].D
                #print localgraphs_list[j].D
                allK_test[point_ind,j_ind,i_ind,:] = kernel.evaluate(localgraphs_list[i],localgraphs_list[j])
        for j_ind, j in enumerate(test_inds):
            allK_testdiagonal[point_ind,j_ind,:] = kernel.evaluate(localgraphs_list[j],localgraphs_list[j])


    gram_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'gram_matrices_xval', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
    try:
        os.makedirs(gram_dir)
        print('Creating new directory: %s' % gram_dir)
    except:
        print('Output directory is %s' % gram_dir)
    gram_path = op.join(gram_dir,'K_%s_sigmasheur_fold%02dof%02d.jl' % (hem,fold_ind+1,n_folds))
    joblib.dump([allK_train,allK_test,allK_testdiagonal,sigma_coords_vals,sigma_depth_vals],gram_path,compress=3)



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
        graph_param = int(args[3])
        n_sl_points = int(args[4])
        #n_folds = int(args[5])
        fold_ind = int(args[5])

    searchlight_compute_gram_matrices_singlefold_normalizedkernel(experiment, hem, graph_type, graph_param, n_sl_points, fold_ind)


if __name__ == "__main__":
    main()


