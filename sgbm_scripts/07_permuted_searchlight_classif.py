import os.path as op
import os
import numpy as np
import sys
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'


'''
C_list = 10. ** np.arange(-4,2)
C_list = 10. ** np.arange(-3,-2)
C_list = 10. ** np.arange(-4,-3)
C_list = 10. ** np.arange(-2,-1)
C_list = 10. ** np.arange(-1,0)
C_list = 10. ** np.arange(0,1)
C_list = 10. ** np.arange(1,2)
'''

def permuted_searchlight_classif(experiment, hem, graph_type, graph_param, n_sl_points, n_folds, n_permuts, C=1e-2, subkernels_option=False, diagnorm_option=True):

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

    #############################
    # load permutations and xvals
    #############################
    permutations_dir = op.join(analysis_dir,'permutations')
    permutations_path = op.join(permutations_dir,'permuted_labels_xval%02dfolds.jl' % n_folds)
    print 'Reading all permuted labels from %s' % permutations_path
    [y_train_permuted_list,y_test_permuted_list] = joblib.load(permutations_path)

    # test if there are enough permutations in this file as compared to what is asked here
    # with the value of n_permuts
    if n_permuts > len(y_train_permuted_list[0]):
        print('Not enough permutations available in %s: you should re-run 06_create_permutations.py' % permutations_path)
        exit()

    gram_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'gram_matrices_xval', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
    skf_scores_permuted = np.zeros([n_folds,n_sl_points,n_permuts])

    ############
    # start xval
    ############
    for fold_ind in range(n_folds):
        print 'Starting fold %d of %d' % (fold_ind+1,n_folds)
        ##################################
        # load gram matrices for this fold
        ##################################
        #gram_path = op.join(gram_dir,'K_%s_autosigmas_fold%02dof%02d.jl' % (hem,fold_ind+1,n_folds))
        gram_path = op.join(gram_dir,'K_%s_sigmasheur_fold%02dof%02d.jl' % (hem,fold_ind+1,n_folds))
        print '   Reading gram matrices in %s' % gram_path
        [allK_train,allK_test,allK_testdiagonal,sigma_coords_vals,sigma_depth_vals] = joblib.load(gram_path)


        #####################
        # do permuted classif
        #####################
        print '   Starting SVM with permuted labels'
        for point_ind in range(n_sl_points):
            print '     Searchligh location %d of %d' % (point_ind+1, n_sl_points)
            #t1 = time.time()
            # extract training gram matrix
            K_train = allK_train[point_ind,:,:,0]
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
            K_test = allK_test[point_ind,:,:,0]
            if diagnorm_option:
                # normalisation of the diagonal terms
                Ktestdiagonal = allK_testdiagonal[point_ind,:,0]
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
            for perm_ind in range(n_permuts):
                #print '     Permutation %d of %d' % (perm_ind+1, n_permuts)
                y_train_permuted = y_train_permuted_list[fold_ind][perm_ind]
                y_test_permuted = y_test_permuted_list[fold_ind][perm_ind]
                g_svc = SVC(kernel='precomputed',C=C)
                g_svc.fit(K_train,y_train_permuted)
                guesses = g_svc.predict(K_test)
                skf_scores_permuted[fold_ind, point_ind, perm_ind] = accuracy_score(y_test_permuted,guesses)
            #t2 = time.time()
            #print t2-t1


    ################
    # save stuff...
    ################
    res_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'permuted_classif_res', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
    try:
        os.makedirs(res_dir)
        print('Creating new directory: %s' % res_dir)
    except:
        print('Output directory is %s' % res_dir)
    res_path = op.join(res_dir,'%s.classif_res_%dpermuts_xval%02dfolds_diagnorm%s_C1e%d.jl' % (hem,n_permuts,n_folds,str(diagnorm_option).lower(),np.int(np.log10(C))))
    print 'Saving all results in %s' % res_path
    joblib.dump(skf_scores_permuted,res_path,compress=3)




def main():
    args = sys.argv[1:]
    if len(args) < 4:
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
    permuted_searchlight_classif(experiment, hem, graph_type, graph_param, n_sl_points, n_folds, n_permuts, C=1., subkernels_option = False, diagnorm_option = True)


    '''
    #graph_type = 'nn'
    #graph_param = 4
    #graph_type = 'radius'
    #graph_param = 50
    
    graph_type = 'conn'
    graph_param = 1
    hem = 'rh'
    n_sl_points = 1000
    
    area = 100
    
    n_permuts = 100
    
    run_searchlight(graph_type, graph_param, area, hem, n_sl_points, n_permuts)
    '''
    

if __name__ == "__main__":
    main()


