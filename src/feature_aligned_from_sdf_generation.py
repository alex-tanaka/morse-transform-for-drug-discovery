# Packages
from magic import SdfReader, MoleculeComplex, DataAlignment, FeatureSave
import numpy as np
import pandas as pd
import argparse 
from multiprocessing import active_children
from joblib.externals.loky import get_reusable_executor

# Parse command-line arguments
parser = argparse.ArgumentParser(
                    prog='MorseFeatureGenerator',
                    description='Generates and saves Morse transform output for a given target',
                    epilog='Enjoy generating features')

parser.add_argument('target') # Positional, compulsory argumet
parser.add_argument('--top', default=20, type=int) # Optional argument
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

# Load dictionary of 3D points clouds + partial charges + lipophilicity
# + symbol + atomic number + vdw radius 
reader = SdfReader(file_path_sdf, name)
targets_dict = reader.reading()

# Create a dictionary of feature vectors
# using the top critical points for each direction and data augmentation.

# Number of actives
n_actives = len(targets_dict[name]['active'])

# Number of inactives
n_inactives = len(targets_dict[name]['inactive'])

# Rounded-up ratio of inactives to actives in the target
# For aligned features this is set to 1 as a hack
ratio_decimal = 1 # n_inactives / n_actives 
ratio = int(np.ceil(ratio_decimal))

# Number of inactives
n_inactives = len(targets_dict[name]['inactive'])

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

# Directions to analyse the complexes
m = MoleculeComplex()
directions1 = m.getDirectionVectors(N=12, method='regularPolyhedron')
directions2 = m.getDirectionVectors(N=20, method='regularPolyhedron')
directions3 = m.getDirectionVectors(N=68, method='gaussian', random_seed=2024)
directions = np.concatenate((
                              directions1, 
                              directions2, 
                              # directions3
                             ), axis=0)

# Augment only the actives
a = DataAlignment()
aug = {}
aug[name] = {}

aug[name]['inactive'] = a.align(targets_dict[name]['inactive'])
aug[name]['active'] = a.align(targets_dict[name]['active'])

aug[name]['both'] = aug[name]['inactive'] + aug[name]['active']

# Calculate feature vectors
# Loky documentation: The order of the outputs always matches the 
# order the inputs have been submitted with. 
# return_as: str in {‘list’, ‘generator’, ‘generator_unordered’}, default=’list’
# If ‘list’, calls to this instance will return a list, only when all results 
# have been processed and retrieved. 
feature_vector = {}
feature_vector[name] = {}

feature_vector_both = m.build_getFeatures(aug[name]['both'],
                                          directions,
                                          top=top,
                                          parallel_calc=True,
                                          backend='loky',
                                          progress_bar=True)

feature_vector[name]['inactive'] = feature_vector_both[:n_inactives]
feature_vector[name]['active'] = feature_vector_both[n_inactives:]

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