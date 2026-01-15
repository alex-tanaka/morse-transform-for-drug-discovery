print('Classification script started\n')

# Packages
from classification import (HelperFunctions,
                            Results, 
                            TuningHybridGB) 
import argparse 
import pandas as pd
import torch
from ray import tune

# Parse command-line arguments
parser = argparse.ArgumentParser(
                    prog='FeatureClassifier',
                    description='Classifies chemically-enchanced Morse features and saves the results',
                    epilog='Enjoy classifying features')

parser.add_argument('--targets', nargs='+', type=str,  default='ampc') # nargs='+' concatenates multiples args, requires at least one
parser.add_argument('--dataset', type=str, default='dude', ) # Optional argument
parser.add_argument('--baseline_dataset', type=str, default='dude_baseline_q9', ) # Optional argument
parser.add_argument('--results_prefix', type=str, default='icos') # Optional argument

args = parser.parse_args()

# Assign command-line arguments
target_names = args.targets
dataset = args.dataset
baseline_dataset = args.baseline_dataset
results_name = args.results_prefix
print(results_name)

print('targets =', target_names, '\n')

# File path names
# May need to be altered depending on your file structure
file_path_root = '../data/'
file_path_share_root = file_path_root  + 'parameters/'
save_folder = 'results/' + dataset + '/'

# Parameters
# results_name = 'icos' # CHANGE this to prevent overwriting old results
desired_crit_categories = [0,1,2]
top_values = [20]# [20, 3, 7, 1, 5, 15, 13, 10]
num_directions = 32# 100 #(12, 20) # 1 # 12
unaug_train = False #
weight_positives = False #
pos_weight_factor = 1

print("Using torch", torch.__version__)

torch.manual_seed(2023) # Setting the seed

gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

if torch.cuda.is_available():
    print(torch.cuda.device_count(), 'x GPU')

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Helper functions
h = HelperFunctions()

# Load parameter file
# Could generate from the point cloud dictionary directly?
parameter_name = dataset + '_parameters.csv'
parameter = pd.read_csv(file_path_root + 'parameters/' + dataset + '_parameters.csv',
                        sep=',',
                        dtype={'Targets': 'string',
                                'Actives': 'int64',
                                'Inactives': 'int64',
                                'Inactives/Actives': 'float64'}
                        )

# MAGIC
# Reading 5D features and labels
calc_target_names = target_names 

features = {}
baseline_features = {}
labels = {}

for i in calc_target_names:
    features[i] = pd.read_parquet(file_path_root + 'features/' + dataset + '/' + i + '_rand_pentakis_top_20_features.parquet')

    labels[i] = pd.read_parquet(file_path_root + 'features/' + dataset + '/' + i + '_rand_pentakis_top_20_labels.parquet')

    # convert parquet columns from strings to int to allow for easier indexing later
    features[i].columns = features[i].columns.astype(int)

    labels[i].columns = labels[i].columns.astype(int)

    # Get repeated indices for augmented molecules
    p = h.get_parameters(i, parameter)
    N_inactive = labels[i].size - labels[i].sum()[0]
    N_active = int(labels[i].sum()[0]/p[2]) #len(ID_5D[i]['all']) - N_inactive

    g = h.make_groups(N_active,
                      N_inactive,
                      active_aug_multi=p[2])
    
    print('X:', len(features[i]),
          ', label:', labels[i].size,
          ', # aug actives: ', labels[i].sum()[0],
          ', # actives:', labels[i].sum()[0]/p[2],
          ', unaug tot:', labels[i].size - labels[i].sum()[0] + labels[i].sum()[0]/p[2],
          ', groups', len(g),
          ', parameter', p
          )

print('\n')
print('Baseline features')
for i in calc_target_names:
    baseline_features[i] = pd.read_parquet(file_path_root + 'features/' + baseline_dataset + '/' + i + '_rand_pentakis_top_20_features.parquet')

    # labels[i] = pd.read_parquet(file_path_root + 'features/' + dataset + '/' + i + '_rand_pentakis_top_20_labels.parquet')

    # convert parquet columns from strings to int to allow for easier indexing later
    baseline_features[i].columns = baseline_features[i].columns.astype(int)

    # labels[i].columns = labels[i].columns.astype(int)

    # Get repeated indices for augmented molecules
    p = h.get_parameters(i, parameter)
    N_inactive = labels[i].size - labels[i].sum()[0]
    N_active = int(labels[i].sum()[0]/p[2]) #len(ID_5D[i]['all']) - N_inactive

    g = h.make_groups(N_active,
                      N_inactive,
                      active_aug_multi=p[2])
    
    print('X:', len(baseline_features[i]),
          ', label:', labels[i].size,
          ', # aug actives: ', labels[i].sum()[0],
          ', # actives:', labels[i].sum()[0]/p[2],
          ', unaug tot:', labels[i].size - labels[i].sum()[0] + labels[i].sum()[0]/p[2],
          ', groups', len(g),
          ', parameter', p
          )

# Create results files
r = Results()
r.create_results_dfs(results_name, 
                     file_path_root=file_path_root, 
                     folder=save_folder,
                     target_names=target_names)

# Training
targets = target_names

# LightGBM parameters
params = {'device_type': 'cpu',
          'num_threads': 1,#
          'objective': 'binary',
          # 'scale_pos_weight': 1.0,#1.0 default
          'num_leaves': 31,#5,31
          'min_data_in_leaf': 20,#20, 200
          'bagging_freq': 1,#0
          'bagging_fraction': 1.0,#1
          'min_sum_hessian_in_leaf': 1e-3,
          'max_depth': 100,#-1, 100
          'num_iterations': 100,#100
          'learning_rate': 0.1,#0.1
          # 'lambda_l1': 0.0,#0 default
          # 'lambda_l2': 0,#0 default
          'min_gain_to_split': 0,#0
          'feature_fraction': 1.0, #1.0
          'max_bin': 255,#255 default
          # 'extra_trees': True,#False default
          # 'boosting': 'gbdt',#'dart', 'gbdt' default
          'verbosity': -1}

# Hyperparameter search space
config = {'num_leaves': tune.lograndint(2, 4095),# (2, 4095)
          'min_data_in_leaf': tune.lograndint(20, 2000), #
          'max_depth': tune.randint(2, 100),
          # 'max_bin': tune.randint(2, 1000),
          # 'path_smooth': tune.loguniform(1e-5, 1e5),
          # 'learning_rate': tune.loguniform(1e-7, 1e2), (1e-3, 10)
          # 'num_iterations': tune.lograndint(10, 1000),
          'lambda_l1': tune.loguniform(1e-7, 1e5),
          'lambda_l2': tune.loguniform(1e-7, 1e5),
          'bagging_fraction': tune.uniform(0.3, 1.0),
          'min_sum_hessian_in_leaf': tune.loguniform(1e-7, 1e5),# (1e-5, 20)
          'feature_fraction': tune.uniform(0.3, 1.0),
          # 'min_gain_to_split': tune.loguniform(1e-6, 1e8),
          # 'scale_pos_weight': tune.loguniform(1, 50000)
          }
          
t = TuningHybridGB()
for top in top_values:
    t.cv_tune_results_multi_target_one_top_save_csv(results_name,
                                             top,
                                             num_directions,
                                             targets,
                                             features,
                                             labels,
                                             baseline_features,
                                             n_original_properties=17,#
                                             num_quantiles=9,#
                                             desired_properties=[3, 4, 5],######
                                             crit_categories=13,
                                             desired_crit_categories=desired_crit_categories,
                                             k_folds=5,
                                             random_seed=2023,
                                             add_num_crit_pts=True,#
                                             add_euler_char=False,#
                                             sum_over_all_crit_pts=True,#
                                             projected_heights=True,#
                                             bin_count_betti=False,#
                                             groups=None,
                                             num_boost_round=100,
                                             params=params,
                                             folder=save_folder,
                                             parameter_file_name=parameter_name,
                                             file_path_root=file_path_root,
                                             file_path_share_root=file_path_share_root,
                                             unaug_train=unaug_train,
                                             weight_positives=weight_positives,
                                             pos_weight_factor=pos_weight_factor,
                                             config=config,
                                             num_samples=200
                                             )