import numpy as np
import pandas as pd
import pickle 

class FeatureSave:
    """
    A class for converting dictionaries or pickled feature vectors 
    to data frames in parquet format.

    Examples
    ========
        >>> a
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """

    def feature_to_parquet(self,
                           target='ampc',
                           feature_dict=False,
                           top=20,
                           file_path_save_feature=''):
        """
        Converts features to parquet files.

        Parameters
        ===========
        list_df: A list
            A list of panda data frames of point clouds and VdW radii
        rotation: A string

        Returns
        ========
        """

        # Load feature vectors
        #try:
        #    import pickle5 as pickle
        #except ImportError:  # Python 3.x
        #    import pickle

        # with open(file_path_save_feature + 'features_7D/' + target + '_rand_pentakis_top_20_features_7D.p', 'rb') as fp:
        #     feature = pickle.load(fp, encoding="bytes")

        if feature_dict:
            feature = feature_dict
        else:
            with open(file_path_save_feature + target + '_rand_pentakis_top_' + str(top) + '_features.p', 'rb') as fp:
                feature = pickle.load(fp, encoding="bytes")

        N_active = len(feature[target]['active'])

        # Features
        features = feature[target]['active']
        features.extend(feature[target]['inactive'])
        x = np.stack(features, axis=0) # memory intensive

        # Labels
        N = np.shape(features)[0]
        y = np.zeros(N)
        y[0:N_active] = np.ones(N_active)
        y = np.array([int(i) for i in y], dtype='int32')

        x_df = pd.DataFrame(x)
        y_df = pd.DataFrame(y)

        x_df.columns = x_df.columns.astype(str)
        y_df.columns = y_df.columns.astype(str)

        x_df.to_parquet(file_path_save_feature + target + '_rand_pentakis_top_' + str(top) + '_features.parquet')
        y_df.to_parquet(file_path_save_feature + target + '_rand_pentakis_top_' + str(top) + '_labels.parquet')

