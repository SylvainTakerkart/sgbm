import numpy as np
from scipy.sparse import csgraph
import scipy.spatial.distance as sd

def get_clusters(statMap, connMatrix, threshold):

    n_sl_points = len(statMap)

    subthresholdPointInds = np.where(statMap < threshold)[0]
    if len(subthresholdPointInds) <n_sl_points:
        
        clusterGraph = connMatrix.copy()
        clusterGraph[subthresholdPointInds,:] = False
        clusterGraph[:,subthresholdPointInds] = False
        clusterGraph = np.array(clusterGraph)
        
        [n_clusters,clusterLabels] = csgraph.cs_graph_components(clusterGraph)
        
        # get all cluster masses
        clusterMasses = []
        for i in range(n_clusters):
            clusterMasses.append(statMap[np.where(clusterLabels==i)[0]].sum())

        # probably important to keep, or re-introduce somewhere at some point!
        '''
        if len(clusterMasses):
            # sanity check in cas of all supra-threshold points were in the pole mask
            max_cluster_masses_list.append(np.nanmax(cluster_mass_list))
            all_cluster_masses_list.extend(cluster_mass_list)
        '''

    else:
        # if there are no clusters
        clusterLabels, clusterMasses = [], []

    return clusterLabels, clusterMasses

def get_center_of_cluster(clusterLabelMap, clusterLabel, sl_spherepoints_coords):
    '''to be put back into pits_lib/pitslearn.py when operational'''

    thiscluster_inds = np.where(clusterLabelMap==clusterLabel)[0]
    thisclustercenter_coords = sl_spherepoints_coords[thiscluster_inds,:].mean(0)
    distances = sd.cdist(np.atleast_2d(thisclustercenter_coords),sl_spherepoints_coords).squeeze()
    thisclustercenter_ind = distances.argmin()

    return thisclustercenter_ind

