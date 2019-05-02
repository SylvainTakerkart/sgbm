import nibabel.gifti as ng
import os.path as op
import os
import numpy as np
import sys
import graph
import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.spatial.distance as sd

import matplotlib.pyplot as plt


# read parameters and subjects lists
root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'
experiment = 'abide_jbhi_pits01'
analysis_dir = op.join(root_analysis_dir, experiment)



def compute_localgraphs(graph_type, graph_param, hem, n_sl_points):

    if graph_type == 'nn':
        n_neighbors = graph_param
    elif graph_type == 'radius':
        radius = graph_param
    elif graph_type == 'conn':
        connlength = graph_param

    sampleslist_path = op.join(analysis_dir,'samples_list.jl')
    [y, samples_subjects_list, samples_hem_list, samples_group_list] = joblib.load(sampleslist_path)

    ##################
    # reading all info
    ##################

    print 'Reading things from disk (positions of searchlight points, full pits-graphs etc.)'

    # reading points where the searchlight will be performed
    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hem,n_sl_points))
    [sl_points,sl_points_inds] = joblib.load(fibopoints_path)

    # read the full-hemisphere pits graphs for all subjects of the database
    fullgraphs_dir = op.join(analysis_dir,'full_hemisphere_pitgraphs')
    pitgraphs_path = op.join(fullgraphs_dir,'full_%s_pitgraphs.jl' % hem)
    pitgraphs_dict = joblib.load(pitgraphs_path)

    # create input data X with the fullbrain graphs, and labels y 
    X = []
    for subject in samples_subjects_list:
        X.append(pitgraphs_dict[subject])

    n_examples = len(y)
    y = np.array(y)

    ##########################
    # compute all local graphs
    ##########################
    localgraphs_dir = op.join(analysis_dir,'searchlight','%s_based' % graph_type,'local_graphs', '%s_%d' % (graph_type, graph_param), 'sl_%s_points' % n_sl_points)
    try:
        os.makedirs(localgraphs_dir)
        print('Creating new directory: %s' % localgraphs_dir)
    except:
        print('Output directory is %s' % localgraphs_dir)

    print 'Compute all local graphs'

    for point_ind in sl_points_inds:
        current_center = sl_points[point_ind,:] * 100
        print point_ind
        localgraphs_list = []
        for n, subject in enumerate(samples_subjects_list):
            fullgraph = pitgraphs_dict[subject]
            if graph_type == 'nn':
                # 1. find the n_neigbhors closest pits from the current center
                pits_distances = sd.cdist(current_center[np.newaxis,:],fullgraph.X).squeeze()
                nearby_pits_inds = np.argsort(pits_distances)[range(n_neighbors)]
            elif graph_type == 'radius':
                # 1. find the closest pits from  the current center
                pits_distances = sd.cdist(current_center[np.newaxis,:],fullgraph.X).squeeze()
                nearby_pits_inds = np.where(pits_distances < radius)[0]
                # if there are no pits in the neighborhood, take only the closest pit!
                if len(nearby_pits_inds)==0:
                    nearby_pits_inds = np.array([np.argmin(pits_distances)])
            elif graph_type == 'conn':
                n_pits = fullgraph.A.shape[0]
                # 1. find the line of the connectivity matrix for which the mean of the distances
                # of the connected pits is the smallest (and not the sum, which would favor small graphs);
                # this identifies the "center pit"; then grow the graph from there...
                # first of all, add identity so that the current pit gets included in the sum of distances
                A_tmp = fullgraph.A + np.identity(n_pits)
                mean_of_distances = np.zeros(n_pits)
                for pit_ind in range(n_pits):
                    nearby_pits_inds = np.where(A_tmp[pit_ind,:])[0]
                    local_pits_distances = sd.cdist(current_center[np.newaxis,:],fullgraph.X[nearby_pits_inds,:]).squeeze()
                    mean_of_distances[pit_ind] = np.mean(local_pits_distances)
                closest_pit_ind = mean_of_distances.argmin()
                # now, grow the graphs from this pit!
                pits_inds_list = [closest_pit_ind]
                for c in range(connlength-1):
                    pits_inds_list.extend(np.where(A_tmp[np.array(pits_inds_list),:])[0])
                unique_pits_inds = np.unique(pits_inds_list)
                nearby_pits_inds = np.where(A_tmp[unique_pits_inds,:].sum(0))[0]
            # 2. define graph
            adjacency = fullgraph.A[np.ix_(nearby_pits_inds,nearby_pits_inds)]
            pits_coords = fullgraph.X[nearby_pits_inds,:]
            pits_depth = fullgraph.D[nearby_pits_inds]
            localgraph = graph.pitsgraph(adjacency,pits_coords/100.,pits_depth,pits_depth,pits_depth,pits_depth)
            localgraphs_list.append(localgraph)

        localgraphs_path = op.join(localgraphs_dir,'%s.localgraphslist_point%04d.jl' % (hem, point_ind))
        print('Saving %s' % localgraphs_path)
        joblib.dump(localgraphs_list,localgraphs_path,compress=3)

def main():
    args = sys.argv[1:]


    
    if len(args) < 2:
	print "Wrong number of arguments, run it as: %run 03_searchlight_compute_local_graphs lh radius 2500"
	sys.exit(2)
    else:
        hem = args[0]
        graph_type = args[1]
        n_sl_points = int(args[2])

    graph_param_list = np.arange(30,92,5)

    for graph_param in graph_param_list:
        print graph_param
        compute_localgraphs(graph_type, graph_param, hem, n_sl_points)
    

if __name__ == "__main__":
    main()


