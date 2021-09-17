import os.path as op
import os
import numpy as np
import sys
import joblib
import scipy.spatial.distance as sd

# read parameters and subjects lists
root_analysis_dir = '/hpc/nit/users/takerkart/sgbm_bip'
experiment = 'nsbip_dev01'
analysis_dir = op.join(root_analysis_dir, experiment)


def fibonacci(N):
    # pseudo regular sampling of a sphere
    inc = np.pi * (3 - np.sqrt(5))
    off = 2. / N
    r2d = 180./np.pi
    k = np.arange(0,N)
    y = k*off - 1. + 0.5*off
    r = np.sqrt(1 - y*y)
    phi = k * inc
    x = np.cos(phi)*r
    z = np.sin(phi)*r
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    lats = 90.-r2d*theta
    lons = r2d*phi
    return x,y,z


subjectslist_path = op.join(analysis_dir,'subjects_list.jl')
[groups_list, subjects_list] = joblib.load(subjectslist_path)

def define_fibonacci_points(hem, n_sl_points, radius=100):

    fullgraphs_dir = op.join(analysis_dir,'full_hemisphere_pitgraphs')
    pitgraphs_path = op.join(fullgraphs_dir,'full_%s_pitgraphs.jl' % hem)
    pitgraphs_dict = joblib.load(pitgraphs_path)

    # get fibonacci points on sphere
    #n_points = 50
    xx, yy, zz = fibonacci(n_sl_points)
    sl_points = np.vstack([xx,yy,zz]).T

    # keep only the points that have one pit close-by (radius)
    points_count = np.zeros(n_sl_points)
    for subj_ind, subject in enumerate(subjects_list):
        #print(subject)
        pits_3dcoords = pitgraphs_dict[subject].X
        for point_ind in range(n_sl_points):
            current_point = sl_points[point_ind,:] * 100
            # compute distances between this point and all the pits
            distances = sd.cdist(current_point[np.newaxis,:],pits_3dcoords).squeeze()
            #print(distances.shape)
            point_pits_min_distance = distances.min()
            #print(point_pits_min_distance)
            if point_pits_min_distance > radius:
                # this is a bad point!
                points_count[point_ind] = points_count[point_ind] + 1
    
    # keep only the points for which a pit is close by!
    sl_points_inds = np.where(points_count==0)[0]
    print(len(sl_points_inds))

    # if the radius is large enough, all points should be valid! in that case, do not mention
    # the radius in the filename
    # if some points have been discarded because there is no pit within the given radius, then
    # use the radius in the filename!

    spheresampling_dir = op.join(analysis_dir,'searchlight','sphere_sampling')
    try:
        os.makedirs(spheresampling_dir)
        print('Creating new directory: %s' % spheresampling_dir)
    except:
        print('Output directory is %s' % spheresampling_dir)
    if len(sl_points_inds) == n_sl_points:
        fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints.jl' % (hem,n_sl_points))
    else:
        fibopoints_path = op.join(spheresampling_dir,'%s.sphere_sampling_%dpoints_%dradius.jl' % (hem,n_sl_points,radius))
    joblib.dump([sl_points,sl_points_inds],fibopoints_path,compress=3)



def main():
    args = sys.argv[1:]
    print(args)
    
    if len(args) < 2:
        print("Wrong number of arguments, run it as: %run 02_define_sphere_sampling.py lh 2500")
        #usage()
        sys.exit(2)
    else:
        hem = args[0]
        n_sl_points = int(args[1])

    define_fibonacci_points(hem, n_sl_points)

if __name__ == "__main__":
    main()

