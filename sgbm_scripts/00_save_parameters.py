import os
import os.path as op
import joblib


'''
Create two files in which we store i) the parameters used for extracting the pits and ii) the list of subjects and groups
'''


root_analysis_dir = '/hpc/nit/users/takerkart/sgbm_bip'

experiment = 'nsbip_dev01'

# parameters used for the extraction of sulcal pits
alpha = '0.03'
an = '0'
dn = '20'
r = '1.5'
area = 50
param_string = 'D%sR%sA%s' % (dn,r,area)

# the subjects lists
bip = ['794288333904', '358764900547']

ctrl = ['975557055860', '582268889106']


groups_list = []
groups_list.append(asd)
groups_list.append(ctrl)

groupnames_list = ['bip','cntrl']

subjects_list = []
for group in groups_list:
    subjects_list.extend(group)



# create directory where all the analyses will be performed
analysis_dir = op.join(root_analysis_dir, experiment)
try:
    os.makedirs(analysis_dir)
    print('Creating new directory: %s' % analysis_dir)
except:
    print('Output directory is %s' % analysis_dir)


subjectslist_path = op.join(analysis_dir,'subjects_list.jl')
joblib.dump([groups_list, subjects_list],subjectslist_path,compress=3)


params_path = op.join(analysis_dir,'pits_extraction_parameters.jl')
joblib.dump([alpha, an, dn, r, area, param_string],params_path,compress=3)


# create labels y and other variables
y = []
samples_hem_list = []
samples_group_list = []
samples_subjects_list = []
for group_ind in range(len(groups_list)):
    for subject in groups_list[group_ind]:
        y.append(group_ind)
        samples_hem_list.append('')
        samples_group_list.append(groupnames_list[group_ind])
        samples_subjects_list.append(subject)

sampleslist_path = op.join(analysis_dir,'samples_list.jl')
joblib.dump([y, samples_subjects_list, samples_hem_list, samples_group_list],sampleslist_path,compress=3)





