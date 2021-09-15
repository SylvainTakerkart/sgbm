import nibabel.gifti as ng
import os
import os.path as op
import sys
import numpy as np
#from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph
import graph
import joblib


# read parameters and subjects lists
input_data_dir = '/hpc/scalp/data/BIPS_database/data_neurospin/processed/sulcal_pits'
root_analysis_dir = '/hpc/nit/users/takerkart/sgbm_bip'
experiment = 'nsbip_dev01'
analysis_dir = op.join(root_analysis_dir, experiment)

subjectslist_path = op.join(analysis_dir,'subjects_list.jl')
[groups_list, subjects_list] = joblib.load(subjectslist_path)

params_path = op.join(analysis_dir,'pits_extraction_parameters.jl')
[alpha, an, dn, r, area, param_string] = joblib.load(params_path)


def compute_fullgraphs(hem):

    pitgraphs_dict = dict()
    for subject in subjects_list:

        print(subject)

        # get basins texture (-1 in poles; everything 0 or above is a real basin with one pit)
        basins_tex_gii = ng.read(op.join(input_data_dir,subject,'%s_%s_%s_basinsTexture.gii' % (subject, hem, param_string)))
        basins_tex = basins_tex_gii.darrays[0].data

        # get pits texture (0 everywhere except single vertex with one where the pits are)
        pits_tex_gii = ng.read(op.join(input_data_dir,subject,'%s_%s_%s_pitsTexture.gii' % (subject, hem, param_string)))#0050002_rh_D20R1.5A50_pitsTexture.gii
        pits_tex = pits_tex_gii.darrays[0].data
        pits_inds = np.where(pits_tex)[0]

        # get area of basins
        area_tex_gii = ng.read(op.join(input_data_dir,subject,'%s_%s_%s_areasTexture.gii' % (subject, hem, param_string)))#0050002_rh_D20R1.5A50_areasTexture.gii
        area_tex = area_tex_gii.darrays[0].data
        basins_area = area_tex[pits_inds]
        # STt 2015/08/24 (first experiment with this: oasis_asymmetries_allgenders, abide_nyu), i.e after MEDIA submission
        # convert vector to matrix
        basins_area = np.atleast_2d(basins_area).T

        # read triangulated spherical mesh and get coordinates of all vertices
        mesh_path = op.join(input_data_dir,subject,'%s.sphere.reg.gii' % hem)
        mesh_gii = ng.read(mesh_path)
        mesh_coords = mesh_gii.darrays[0].data
        # convert cartesian coordinates into spherical coordinates
        ro = np.sqrt(np.sum(mesh_coords*mesh_coords,1))
        phi = np.arctan(mesh_coords[:,1]/mesh_coords[:,0])
        theta = np.arccos(mesh_coords[:,2]/ro)
        spherical_coords = np.vstack([phi,theta]).T

        # compute connectivity of the triangulated mesh
        mesh_connectivity = kneighbors_graph(mesh_coords, n_neighbors=6, include_self=False)

        # get coordinates of the pits on the sphere and in 3d space
        pits_spherecoords = spherical_coords[pits_inds,:]
        pits_3dcoords = mesh_coords[pits_inds,:]
        n_pits = len(pits_inds)



        # get basins labels for each pit, and put them in the same order as the pits!
        basins_labels = []
        for pit_ind in range(n_pits):
            basins_labels.append(basins_tex[pits_inds[pit_ind]])

        # sanity check on basins and pits textures (one pit per basin; same number of pits and basins etc.)
        basins_tmp1_labels = np.unique(basins_tex)
        # get rid of the nodes which have negative labels (-1 labels for the poles)
        basins_tmp1_labels = basins_tmp1_labels[np.where(basins_tmp1_labels >= 0)[0]]
        basins_tmp2_labels = basins_labels[:]
        basins_tmp2_labels.sort()
        if ( (len(basins_tmp1_labels) != len(basins_tmp2_labels)) or np.max(np.abs(np.array(basins_tmp1_labels)-np.array(basins_tmp2_labels)))):
            print("Error: there's something weird with the pits and/or basins textures: %s and %s " % (pits_path, basins_path))



        # build connectivity matrix of the basin-based region adjacancy graph
        basins_submask = []
        basins_size = []
        n_basins = np.size(basins_labels)
        for basin_ind, basin_label in enumerate(basins_labels):
            basins_submask.append(np.array(np.nonzero(basins_tex == basin_label))[0])
            basins_size.append(len(basins_submask[-1]))
        #print(basins_size)
        adjacency = np.zeros([n_basins, n_basins])
        for i in range(n_basins):
            for j in range(i):
                adjacency[i,j] = mesh_connectivity[basins_submask[i],:][:,basins_submask[j]].sum()
                adjacency[j,i] = adjacency[i,j]
            adjacency[adjacency!=0] = 1
        # STt 2015/08/24 (first experiment with this: oasis_asymmetries_allgenders, abide_nyu), i.e after MEDIA submission
        # add ones on the diagonal (every node is connected to itself) to avoid problems in kernel computation
        np.fill_diagonal(adjacency,1.)

        # read depth of the pits
        depth_path = op.join(input_data_dir,subject,'%s_%s_dpf_%s.gii' % (subject,hem,alpha) )#0050002_rh_dpf_0.03.gii
        depth_gii = ng.read(depth_path)
        depth_tex = depth_gii.darrays[0].data
        pits_depth = depth_tex[pits_inds]
        # STt 2015/08/24 (first experiment with this: oasis_asymmetries_allgenders, abide_nyu), i.e after MEDIA submission
        # convert vector to matrix
        pits_depth = np.atleast_2d(pits_depth).T

        # read thickness and compute mean thickness in basin
        thickness_path = op.join(input_data_dir,subject,'%s.thickness.gii' % hem)
        thickness_gii = ng.read(thickness_path)
        thickness_tex = thickness_gii.darrays[0].data
        basins_thickness = np.zeros(n_pits)
        for basin_ind, basin_label in enumerate(basins_labels):
            basin_inds = np.where(basins_tex == basin_label)[0]
            basins_thickness[basin_ind] = np.mean(thickness_tex[basin_inds])
        # STt 2015/08/24 (first experiment with this: oasis_asymmetries_allgenders, abide_nyu), i.e after MEDIA submission
        # convert vector to matrix
        basins_thickness = np.atleast_2d(basins_thickness).T

        # now, we have everything, we can construct the pits graph!
        g = graph.pitsgraph(adjacency, pits_3dcoords, pits_depth, basins_area, basins_thickness, pits_spherecoords)

        # put all this in the pits dictionnary
        pitgraphs_dict[subject] = g


    fullgraphs_dir = op.join(analysis_dir,'full_hemisphere_pitgraphs')
    try:
        os.makedirs(fullgraphs_dir)
        print('Creating new directory: %s' % fullgraphs_dir)
    except:
        print('Output directory is %s' % fullgraphs_dir)


    pitgraphs_path = op.join(fullgraphs_dir,'full_%s_pitgraphs.jl' % hem)
    joblib.dump(pitgraphs_dict,pitgraphs_path,compress=3)


def main():
    args = sys.argv[1:]


    # for both hemispheres, just run it as 'python 01_create_full_hemisphere_pitgraphs.py'
    # for just one hemisphere, run it as 'python 01_create_full_hemisphere_pitgraphs.py lh'

    if len(args) < 1:
        hemispheres_list = ['lh','rh']
    else:
        hem = args[0]
        hemispheres_list = [hem]

    for hem in hemispheres_list:
        compute_fullgraphs(hem)
    

if __name__ == "__main__":
    main()
