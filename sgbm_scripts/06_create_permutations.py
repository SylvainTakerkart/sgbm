import os.path as op
import os
import numpy as np
import sys
import joblib



# read parameters and subjects lists
root_analysis_dir = '/hpc/nit/users/takerkart/sgbm_bip'
experiment = 'nsbip_dev01'
analysis_dir = op.join(root_analysis_dir, experiment)

sampleslist_path = op.join(analysis_dir,'samples_list.jl')
[y, samples_subjects_list, samples_hem_list, samples_group_list] = joblib.load(sampleslist_path)

n_examples = len(y)
y = np.array(y)

        
def create_permuted_labels_within_xval_folds(n_permuts,n_folds):


    #########################################################
    # Load the reference xval computed on the original labels
    #########################################################
    xval_dir = op.join(analysis_dir,'xval')
    xval_path = op.join(xval_dir,'stratified_%02dfold_inds.jl' % n_folds)
    print('Loading cross-validation indices from %s' % xval_path)
    [train_inds_list,test_inds_list] = joblib.load(xval_path)

    # To make sure that the hyper-parameters used to compute the
    # gram matrices at each fold are valid, make sure to permute
    # labels separately within the training set and testing set
    # of each fold of the cross validation
    y_train_permuted_list = []
    y_test_permuted_list = []
    for fold_ind, train_inds in enumerate(train_inds_list):
        test_inds = test_inds_list[fold_ind]
        y_train = y[train_inds]
        y_test = y[test_inds]
        y_train_permuted_list_thisfold = []
        y_test_permuted_list_thisfold = []
        y_train_permuted_list_thisfold.append(y_train)
        y_test_permuted_list_thisfold.append(y_test)
        for perm_ind in range(n_permuts-1):
            current_train_permut = np.random.permutation(len(train_inds))
            current_test_permut = np.random.permutation(len(test_inds))
            y_train_permuted_list_thisfold.append(y_train[current_train_permut])
            y_test_permuted_list_thisfold.append(y_test[current_test_permut])
        y_train_permuted_list.append(y_train_permuted_list_thisfold)
        y_test_permuted_list.append(y_test_permuted_list_thisfold)
            
    # Save only the permuted labels! (including the original non permuted ones
    # as the first item of the lists)
    permutations_dir = op.join(analysis_dir,'permutations')
    try:
        os.makedirs(permutations_dir)
        print('Creating new directory: %s' % permutations_dir)
    except:
        print('Output directory is %s' % permutations_dir)
    permutations_path = op.join(permutations_dir,'permuted_labels_xval%02dfolds.jl' % n_folds)
    if op.exists(permutations_path):
        print('Warning: overwriting %s' % permutations_path)
    joblib.dump([y_train_permuted_list,y_test_permuted_list],permutations_path,compress=3)


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        n_permuts = 50
        n_folds = 2
    else:
        n_permuts = int(args[0])
        n_folds = int(args[1])

    # run it without any arguments if the default 10-fold is chosen!

    create_permuted_labels_within_xval_folds(n_permuts,n_folds)


if __name__ == "__main__":
    main()


