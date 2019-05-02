import os
import os.path as op
import joblib


'''
Create two files in which we store i) the parameters used for extracting the pits and ii) the list of subjects and groups
'''


root_analysis_dir = '/riou/work/scalp/hpc/auzias/sgbm'

experiment = 'abide_jbhi_pits01'

# parameters used for the extraction of sulcal pits
alpha = '0.03'
an = '0'
dn = '20'
r = '1.5'
area = 50
param_string = 'D%sR%sA%s' % (dn,r,area)

# the subjects lists
asd = ['0050475','0050478','0050481','0050482','0050484','0050485','0050486','0050490','0050496','0050498','0050499','0050500','0050501','0050506','0050508','0050510','0050511','0050512','0050515','0050516','0050519','0050520','0050521','0050522','0050523','0050524','0050527','0050531','0050686','0050689','0050690','0050693','0050694','0050695','0050696','0050697','0050702','0050704','0050705','0050708','0050711','0050745','0050746','0050747','0050748','0050750','0050753','0050754','0050755','0050756','0050757','0050964','0050966','0050972','0050973','0050974','0050975','0050976','0050983','0050985','0050990','0050991','0050994','0050995','0051007','0051014','0051015','0051016','0051017','0051018','0051021','0051025','0051028','0051029','0051034']


ctrl = ['0050437','0050438','0050440','0050441','0050442','0050443','0050445','0050446','0050447','0050448','0050449','0050450','0050451','0050453','0050454','0050458','0050459','0050460','0050461','0050462','0050463','0050464','0050467','0050470','0050471','0050472','0050473','0050474','0050682','0050683','0050685','0050687','0050688','0050691','0050692','0050698','0050699','0050701','0050703','0050707','0050709','0050710','0050724','0050726','0050728','0050731','0050732','0050733','0050734','0050737','0050738','0050739','0050740','0050741','0050742','0051067','0051074','0051077','0051089','0051091','0051094','0051096','0051099','0051103','0051104','0051105','0051108','0051110','0051111','0051112','0051113','0051115','0051116','0051118','0051120','0051121','0051126','0051128','0051130','0051148','0051150','0051153','0051154','0051156']


groups_list = []
groups_list.append(asd)
groups_list.append(ctrl)

groupnames_list = ['asd','cntrl']

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





