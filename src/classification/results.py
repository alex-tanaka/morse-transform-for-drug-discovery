import pandas as pd
import numpy as np

class Results:
    """
    A class for creating empty .csv files to store
    classification results

    Examples
    ========
        >>> a
    """

    def __init__(self):
        """
        Default constructor

        Parameters
        ----------
        """
    
    def create_results_dfs(self,
                           direction_name,
                           file_path_root='data/',
                           folder='results/',
                           target_names=['abl1', 'ace', 'adrb1', 'adrb2', 'ampc',
                                         'andr', 'cp2c9', 'cxcr4', 'fa7', 'grik1',
                                         'hdac8', 'kith', 'mcr', 'pgh1', 'reni'],
                           top_values=[20, 15, 13, 10, 7, 5, 3, 1],
                            ):
        """
        Create the dataframes to store the results of the cross validated
        generalisation errors.
        """
        results = {}
        results['target'] = pd.Series(sorted(target_names), dtype='string')
        N = len(target_names)

        for top in top_values:
            results['top %d' %(top)] = pd.Series(np.zeros(N), dtype='float')
            results['top %d STD' %(top)] = pd.Series(np.zeros(N), dtype='float')
            results['top %d low' %(top)] = pd.Series(np.zeros(N), dtype='float')
            results['top %d up' %(top)] = pd.Series(np.zeros(N), dtype='float')
            results['top %d min' %(top)] = pd.Series(np.zeros(N), dtype='float')
            results['top %d max' %(top)] = pd.Series(np.zeros(N), dtype='float')

        df = pd.DataFrame(results)

        # Metrics
        metrics = ['roc', 'brier', 'logl',
                   'bedroc_alpha5', 'bedroc_alpha10', 'bedroc_alpha20',
                   'ef_1', 'abs_ef_1',
                   'pr_auc',
                   'accuracy']
        
        for m in metrics:
            df.to_csv(file_path_root + folder + '%s_%s_results.csv' %(direction_name, m))