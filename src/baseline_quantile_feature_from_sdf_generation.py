# Packages
from magic import Baseline, FeatureSave
import numpy as np
import pandas as pd
import argparse 
from multiprocessing import active_children
from joblib.externals.loky import get_reusable_executor

# Parse command-line arguments
parser = argparse.ArgumentParser(
                    prog='ChemicalFeatureGenerator',
                    description='Generates and saves chemical features for a given target',
                    epilog='Enjoy generating features')

parser.add_argument('target') # Positional, compulsory argumet
parser.add_argument('--top', default=20, type=int) # Vestigial optional argument
parser.add_argument('--file_path_root', default='data/') # Optional argument
parser.add_argument('--file_path_sdf', default='data/sdf/dude/') # Optional argument
parser.add_argument('--file_path_save_feature', default='data/features/dude/') # Optional argument
parser.add_argument('--dataset', default='dude') # Optional argument

args = parser.parse_args()

# Change the file_path_root to something reflecting your own directory
file_path_root = args.file_path_root
file_path_sdf = args.file_path_sdf 
file_path_save_feature = args.file_path_save_feature
dataset = args.dataset

# Target depends on the command-line argument
name = args.target

# Load feature vector dicitonary of property quantiles
baseline = Baseline(file_path_sdf, name)
feature_vector = baseline.quantiles()

# Create a dictionary of feature vectors
# using the top critical points for each direction and data augmentation.

# Number of actives
n_actives = len(feature_vector[name]['active'])

# Number of inactives
n_inactives = len(feature_vector[name]['inactive'])

# Rounded-up ratio of inactives to actives in the target
# For aligned features this is set to 1 as a hack
ratio_decimal = 1 # n_inactives / n_actives 
ratio = int(np.ceil(ratio_decimal))

# Number of inactives
n_inactives = len(feature_vector[name]['inactive'])

# Number of top critical values to calculate
top = int(args.top)

# Record parameters in a df
parameter = {}
parameter['Targets'] = pd.Series(np.array([name]), dtype='str')
parameter['Actives'] = pd.Series(np.array([n_actives]), dtype='Int64')
parameter['Inactives'] = pd.Series(np.array([n_inactives]), dtype='Int64')
parameter['Inactives/Actives'] = pd.Series(np.array([ratio_decimal]), dtype='float')

df = pd.DataFrame(parameter)
df.to_csv(file_path_root + 'parameters/' + dataset + '/' + name + '_parameter.csv',
          index=False)  

print('target:', name, 
      ', actives:', n_actives, 
      ', inactives:', n_inactives, 
      ', augmented ratio:', ratio,
      ', top:', top)

print('actives:', len(feature_vector[name]['active']), 
      ', inactives:', len(feature_vector[name]['inactive']))

# Load then save the features and labels as a parquet table
s = FeatureSave()
s.feature_to_parquet(target=name, 
                     file_path_save_feature=file_path_save_feature, 
                     feature_dict=feature_vector,
                     top=top)

# Shutdown all active processes
print('number of active processes:', len(active_children()))
get_reusable_executor().shutdown(wait=True)
print('number of active processes:', len(active_children()), '\n')