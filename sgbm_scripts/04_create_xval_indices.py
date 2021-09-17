import os.path as op
import os
import numpy as np
import sys
import graph
import joblib
from sklearn.model_selection import StratifiedKFold


# read parameters and subjects lists
root_analysis_dir = '/hpc/nit/users/takerkart/sgbm_bip'
experiment = 'nsbip_dev01'
analysis_dir = op.join(root_analysis_dir, experiment)

sampleslist_path = op.join(analysis_dir,'samples_list.jl')
[y, samples_subjects_list, samples_hem_list, samples_group_list] = joblib.load(sampleslist_path)

n_examples = len(y)
y = np.array(y)

        
def create_xval_indices(n_folds):


    # Compute (or load) the reference xval computed on the original labels
    xval_dir = op.join(analysis_dir,'xval')
    try:
        os.makedirs(xval_dir)
        print('Creating new directory: %s' % xval_dir)
    except:
        print('Output directory is %s' % xval_dir)
    xval_path = op.join(xval_dir,'stratified_%02dfold_inds.jl' % n_folds)
        # create xval object and save all train and test indices
    skf_xval_orig = StratifiedKFold(n_splits=n_folds)
    train_inds_list = []
    test_inds_list = []
    #for (train_inds, test_inds) in skf_xval_orig:
    #    train_inds_list.append(train_inds)
    #    test_inds_list.append(test_inds)
    for train_inds, test_inds in skf_xval_orig.split(y, y):
        train_inds_list.append(train_inds)
        test_inds_list.append(test_inds)
        
    print('Saving cross-validation indices to %s' % xval_path)
    joblib.dump([train_inds_list,test_inds_list],xval_path,compress=3)



def main():
    args = sys.argv[1:]
    n_folds = int(args[0])

    create_xval_indices(n_folds)


if __name__ == "__main__":
    main()


