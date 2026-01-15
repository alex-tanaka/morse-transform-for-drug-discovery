from sklearn.utils.validation import _num_samples
from .helper import HelperFunctions
import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from scipy.stats import bootstrap
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             roc_auc_score,
                             average_precision_score,
                             brier_score_loss,
                             log_loss)
import ray
from ray import tune
from ray.air import Checkpoint, session
# from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.ax import AxSearch
from ray.tune.search.hebo import HEBOSearch

class TuningGB:
    """
    A class for classifying features.

    Examples
    ========
        >>> 
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """
    def cv_tune_results_multi_target_one_top_save_csv(self,
                                                      direction_name,
                                                      top,
                                                      n_directions,
                                                      targets,
                                                      x_dict,
                                                      y_dict,
                                                      crit_categories=3,
                                                      desired_crit_categories=[0,1,2],
                                                      original_top=20,
                                                      k_folds=5,
                                                      num_boost_round=100,
                                                      random_seed=2023,
                                                      groups=True,
                                                      folder='results/',
                                                      add_num_crit_pts=False,
                                                      add_euler_char=False,
                                                      sum_over_all_crit_pts=True,
                                                      pure_ect=False,
                                                      projected_heights=True,
                                                      bin_count_betti=False,
                                                      moments=False,
                                                      params={},
                                                        config={},
                                                        num_samples=100,
                                                      graph=False,
                                                      file_path_root='data/',
                                                      file_path_share_root='data/',
                                                      parameter_file_name=None,
                                                      unaug_train=False,
                                                      weight_positives=False,
                                                      pos_weight_factor=1
                                                    ):
        """
        CV results for each target saving the results to csv files
        """

        # Helper functions
        h = HelperFunctions()

        # Load results df
        metrics = ['roc', 'brier', 'logl',
                   'bedroc_alpha5', 'bedroc_alpha10', 'bedroc_alpha20',
                   'ef_1', 'abs_ef_1',
                   'pr_auc',
                   'accuracy']
        
        results = {}
        
        for m in metrics:
            results[m] = pd.read_csv(file_path_root + folder + '%s_%s_results.csv' %(direction_name, m),
                                     index_col=0,
                                     dtype={'target': 'string'}
                                     )
            # print(results[m][['target', 'top 10']].dtypes)

        # Load parameters for targets
        if parameter_file_name is None:
            parameter = pd.read_csv(file_path_share_root + 'parameters.csv',
                                    sep=',',
                                    dtype={'Targets': 'string',
                                           'Actives': 'int64',
                                           'Inactives': 'int64',
                                           'Inactives/Actives': 'float64'}
                                    )
        else:
            parameter = pd.read_csv(file_path_share_root + parameter_file_name,
                                    sep=',',
                                    dtype={'Targets': 'string',
                                           'Actives': 'int64',
                                           'Inactives': 'int64',
                                           'Inactives/Actives': 'float64'}
                                    )

        # Print which top feature
        print('\n')
        print('====================================')
        print('TOP = ', top)
        print('\n')

        for target in targets:
            print('\n')
            print('--------------------------------')
            print(target)
            # print('Python type:', type(target))
            print('\n')

            # Dimensionally reduce the feature vector
            x = h.dim_reduction(x_dict[target], original_top,
                                top,
                                n_directions,
                                crit_categories=crit_categories,
                                desired_crit_categories=desired_crit_categories,
                                projected_heights=projected_heights)#True

            # Summarise the dimensionally reduced feature vector
            x, integer_indices = h.summary(x,
                                           top, 
                                           n_directions,
                                           crit_categories=len(desired_crit_categories),
                                           integer_categories=[0,1,2],
                                           projected_heights=projected_heights,#True
                                           add_num_crit_pts=add_num_crit_pts,
                                           add_euler_char=add_euler_char,
                                           pure_ect=pure_ect,
                                           sum_over_all_crit_pts=sum_over_all_crit_pts,
                                           bin_count_betti=bin_count_betti,
                                           moments=moments)
            y = y_dict[target]

            # Get target details and groups
            target_details = h.get_parameters(target, parameter)
            p = target_details
            if groups is not None:
                groups = h.make_groups(p[0], p[1], active_aug_multi=p[2])

            summary = self.cv_tune_results_one_target(
                        target,
                        x,
                        y,
                        target_details,
                        groups,
                        top=top,
                        crit_categories=len(desired_crit_categories),
                        n_directions=n_directions,
                        k_folds=k_folds,
                        random_seed=random_seed,
                        num_boost_round=num_boost_round,
                        categorical_feature=integer_indices,
                        params=params,
                        graph=graph,
                        unaug_train=unaug_train,
                        weight_positives=weight_positives,
                        pos_weight_factor=pos_weight_factor,
                          config=config,
                          num_samples=num_samples
                        ) 
            
            # Metrics
            # print('summary dictionaries:')
            for m in metrics:
                result = results[m]
                # print(target, type(target))
                # print(summary[m])
                # print(result['target'].dtypes) 
                # print(result)
                # print(result['target']==target)
                result.loc[result['target']==target, 'top %d' %(top)] = summary[m]['mean']
                result.loc[result['target']==target, 'top %d STD' %(top)] = summary[m]['std']
                result.loc[result['target']==target, 'top %d low' %(top)] = summary[m]['low']
                result.loc[result['target']==target, 'top %d up' %(top)] = summary[m]['up']
                result.loc[result['target']==target, 'top %d min' %(top)] = summary[m]['min']
                result.loc[result['target']==target, 'top %d max' %(top)] = summary[m]['max']
                # print(result, '\n')
            
        # Save results df
        # print('results df:')
        for m in metrics:
            result = results[m]
            # print(result, '\n')
            result.to_csv(file_path_root + folder + '%s_%s_results.csv' %(direction_name, m))

    def cv_tune_results_one_target(self,
                                    target,
                                    x,
                                    y,
                                    target_details,
                                    groups,
                                    top=10,
                                    crit_categories=3,
                                    n_directions=12,
                                    k_folds=5,
                                    random_seed=2023,
                                    num_boost_round=100,
                                    categorical_feature=[],
                                    params={'num_leaves': 31},
                                      config={},
                                      num_samples=100,#_num_samples,
                                    graph=False,
                                    unaug_train=False,
                                    weight_positives=False,
                                    pos_weight_factor=1):
        """
        Output the cross-validation metrics for one target
        """

        # Helper functions
        h = HelperFunctions()

        # Metrics
        metrics = ['roc', 'brier', 'logl',
                   'bedroc_alpha5', 'bedroc_alpha10', 'bedroc_alpha20',
                   'ef_1', 'abs_ef_1',
                   'pr_auc',
                   'recall', 'precision', 'accuracy']

        # For fold results
        fold_results = {}

        for m in metrics:
            fold_results[m] = np.zeros(k_folds)

        # Set fixed random number seed
        random_seed = 2023

        # Define the K-folds
        outer_cv = StratifiedGroupKFold(n_splits=k_folds,
                                        shuffle=True,
                                        random_state=random_seed)

        # Remove augmented copies from the test splits
        if groups is not None:
            if unaug_train:
                outer_cv = h.un_aug_train_and_test_splits(outer_cv, x, y, groups, target_details)
                print('unaug train')
            else:
                outer_cv = h.un_aug_test_splits(outer_cv, x, y, groups, target_details)
                print('aug train')
        elif groups is None:
            outer_cv = StratifiedKFold(n_splits=k_folds,
                                       shuffle=True,
                                       random_state=random_seed)
            outer_cv = outer_cv.split(x, y)
            # outer_cv = outer_cv.split(x, y, groups=np.arange(len(y)))#np.arange(len(y))
            print('No GROUPS')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(outer_cv):
            #
            # TUNING
            #

            scheduler = ASHAScheduler(
                                      max_t=num_boost_round,#?
                                      grace_period=1,
                                      reduction_factor=2
                                      )

            # Resources
            gpus_per_trial = 0.0#0.016666
            cpus_per_trial = 1#0.2

            ray.shutdown()
            ray.init(configure_logging=False,
                     log_to_driver=False
                     ) #address='auto', num_gpus=2

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                                        self.train_gb,
                                        target=target,
                                        x=x,
                                        y=y,
                                        target_details=target_details,
                                        groups=groups,
                                        train_idx=train_ids,
                                        top=top,
                                        crit_categories=crit_categories,
                                        n_directions=n_directions,
                                        k_folds=k_folds,
                                        random_seed=random_seed,
                                        num_boost_round=num_boost_round,
                                        params=params,
                                        categorical_feature=categorical_feature,
                                        graph=graph,
                                        unaug_train=False,
                                        weight_positives=False,
                                        pos_weight_factor=1
                                        ),
                    resources={#"gpu": gpus_per_trial,
                               "cpu": cpus_per_trial
                               }
                ),
                tune_config=tune.TuneConfig(
                    metric="logl",#"logl", 'roc_auc', "bedroc_5"
                    mode="min",#"min", 'max'
                    #search_alg=HEBOSearch(metric='roc_auc', mode='max'),#BasicVariantGenerator(), AxSearch()
                    scheduler=scheduler,
                    num_samples=num_samples,
                ),
                param_space=config,
                run_config=ray.train.RunConfig(verbose=0, log_to_file=False)#
                # or
                #rung_config=ray.air.RunConfig(verbose=0)
            )

            tuning_results = tuner.fit()
            
            # Defaults to the metric specified in your Tuner’s TuneConfig.
            # Defaults to the mode specified in your Tuner’s TuneConfig.
            # best_result = tuning_results.get_best_result("logl", "min")
            # best_result = tuning_results.get_best_result("roc_auc", "max")
            best_result = tuning_results.get_best_result()

            print("Best trial config: {}".format(best_result.config))
            print("Best trial final validation loss: {}".format(
                best_result.metrics["logl"]))
            print("Best trial final validation ROCAUC: {}".format(
                best_result.metrics["roc_auc"]))
            print("Best trial final validation BEDROC-5: {}".format(
                best_result.metrics["bedroc_5"]))
            print("Best trial final validation 1% Enrichment Factor: {}".format(
                best_result.metrics["ef_1"]))

            #
            # TRAINING
            #

            # Ratio of inactives to actives in the training data 
            # Weight the positive class
            labels = y.values#np.array(y.values)
            labels = np.squeeze(labels)

            N_actives = np.sum(labels[train_ids])
            N_inactives = len(labels[train_ids]) - N_actives

            pos_weight = N_inactives / N_actives

            if weight_positives:
                params['scale_pos_weight'] = pos_weight

            print('pos_weight =', pos_weight, 
                  ', actives =', N_actives,
                  ', inactives = ', N_inactives)

            # Training data
            try:
                # x is a pandas dataframe
                X = x.values
            except:
                # x is a numpy array
                X = x
            train_labels = labels[train_ids]
            train_data = lgb.Dataset(X[train_ids], 
                                      label=train_labels,
                                      categorical_feature=[])
            # Add tuned parameters
            print(params)
            tuned_params = {**params, **best_result.config}
            print(tuned_params)

            # Train LightGBM classifier
            bst = lgb.train(tuned_params, train_data#, 
                            #num_boost_round=num_boost_round
                            )

            #
            # TESTING:
            #

            # Testing data
            y_true = labels[test_ids]
            X_test = X[test_ids]
            # test_data = lgb.Dataset(X_test, 
            #                           label=y_true,
            #                           categorical_feature=[], 
            #                           reference=train_data)
            # Evaluation for this fold
            y_score = bst.predict(X_test)

            # Metrics
            # with np.printoptions(threshold=np.inf):
            #     print(y_score)
            fold_results['roc'][fold] = roc_auc_score(y_true, y_score)
            fold_results['brier'][fold] = brier_score_loss(y_true, y_score)
            fold_results['logl'][fold] = log_loss(y_true, y_score)
            fold_results['recall'][fold] = recall_score(y_true, (np.array(y_score) > 0.5))
            fold_results['precision'][fold] = precision_score(y_true, (np.array(y_score) > 0.5))
            fold_results['accuracy'][fold] = accuracy_score(y_true, (np.array(y_score) > 0.5))
            fold_results['pr_auc'][fold] = average_precision_score(y_true, y_score)
            # Rdkit implemented metrics:
            score_mat = np.column_stack((y_score, y_true))
            score_mat = score_mat[score_mat[:, 0].argsort()[::-1]] # descending order
            fold_results['bedroc_alpha5'][fold] = CalcBEDROC(score_mat, 1, 5)
            fold_results['bedroc_alpha10'][fold] = CalcBEDROC(score_mat, 1, 10)
            fold_results['bedroc_alpha20'][fold] = CalcBEDROC(score_mat, 1, 20)
            fold_results['ef_1'][fold] = CalcEnrichment(score_mat, 1, [0.01])[0]
            fold_results['abs_ef_1'][fold] = (CalcEnrichment(score_mat, 1, [0.01])[0]
                                * np.sum(score_mat[:,1])/len(score_mat[:,1]))

            print("The {}-fold ROC AUC test score: {}".format(fold, fold_results['roc'][fold]))
            print("The {}-fold EF 1% test score: {}".format(fold, fold_results['ef_1'][fold]))

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION fold_results FOR {k_folds} FOLDS:')
        print('\n')

        summary = {}

        for m in metrics:
            summary[m] = {}
            summary[m]['mean'] = np.mean(fold_results[m])
            summary[m]['std'] = np.std(fold_results[m])
            summary[m]['min'] = np.min(fold_results[m])
            summary[m]['max'] = np.max(fold_results[m])
            # Bootstrapped 95% CI
            # Must be a sequence of samples for this function
            bs = bootstrap((fold_results[m],), 
                           np.mean,
                           confidence_level=0.95,
                           n_resamples=1000)
            summary[m]['low'], summary[m]['up'] = bs.confidence_interval
            print(m, ': mean = %.3f' %(summary[m]['mean']),
                  ', std = %.2g' %(summary[m]['std']),
                  ', lower 95%% CI = %.3f' %(summary[m]['low']),
                  ', uppper 95%% CI = %.3f' %(summary[m]['up']),
                  ', min = %.3f' %(summary[m]['min']),
                  ', max = %.3f' %(summary[m]['max'])
                  )

        return summary
    
    def train_gb(
                  self,
                  config,
                  target='ampc',
                  x=0,
                  y=0,
                  target_details=[],
                  groups=[],
                  top=10,
                  crit_categories=3,
                  n_directions=12,
                  k_folds=5,
                    train_idx=[],
                  random_seed=2023,
                  num_boost_round=100,
                  categorical_feature=[],
                  params={'num_leaves': 31},
                  num_samples=_num_samples,
                  graph=False,
                  unaug_train=False,
                  weight_positives=False,
                  pos_weight_factor=1
                  ):
        """
        Output the cross-validation metrics for one target
        """

        # Helper functions
        h = HelperFunctions()

        # Slice data and labels to training data
        print('tuning loop: train idices', train_idx)
        try:
            # x is a pandas dataframe
            X = x.to_numpy()
        except:
            # x is a numpy array
            X = x
        try:
            # y is a pandas dataframe
            y = y.to_numpy()
        except:
            # y is a numpy array
            y = y
        # x_train = x.iloc[train_idx]#
        # y_train = y.iloc[train_idx]#
        x_train = X[train_idx]#CHECK#
        y_train = y[train_idx]

        # Slice groups to training data
        if groups is not None:
            groups_train = np.array(groups)[train_idx]

        # Metrics
        metrics = ['roc_auc', 'brier', 'logl',
                'bedroc_5', 'bedroc_10', 'bedroc_20',
                'ef_1', 'abs_ef_1',
                'pr_auc',
                'recall', 'precision', 'accuracy']

        # For fold results
        tune_fold_results = {}

        for m in metrics:
            tune_fold_results[m] = np.zeros(k_folds)

        # Set fixed random number seed
        random_seed = 2023

        # Define the K-folds
        outer_cv = StratifiedGroupKFold(n_splits=k_folds,
                                        shuffle=True,
                                        random_state=random_seed)

        # Remove augmented copies from the test splits
        if groups is not None:
            outer_cv = h.un_aug_inner_val_splits(outer_cv,
                                                x_train, 
                                                y_train,
                                                groups_train, 
                                                target_details,
                                                train_idx)
        elif groups is None:
            outer_cv = StratifiedKFold(n_splits=k_folds,
                                       shuffle=True,
                                       random_state=random_seed)
            outer_cv = outer_cv.split(x_train, y_train)
            # x = x_train
            # y = y_train
            # outer_cv = outer_cv.split(x, y, groups=np.arange(len(y)))#np.arange(len(y))
            print('NO GROUPS')

        # Add parameters to be tuned into the LightGBM parameter set
        # print(params)
        trial_params = {**params, **config}
        # print(trial_params)

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(outer_cv):
            #print('--------------------------------')
            #print(f'FOLD {fold}')
            #print('--------------------------------')

            #
            # TRAINING
            #

            # Ratio of inactives to actives in the training data 
            # Weight the positive class
            labels = y_train
            labels = np.squeeze(labels)

            N_actives = np.sum(labels[train_ids])
            N_inactives = len(labels[train_ids]) - N_actives

            pos_weight = N_inactives / N_actives

            if weight_positives:
                trial_params['scale_pos_weight'] = pos_weight

            print('pos_weight =', pos_weight, 
                  ', actives =', N_actives,
                  ', inactives = ', N_inactives)

            # Training data
            train_labels = labels[train_ids]
            train_data = lgb.Dataset(x_train[train_ids], 
                                      label=train_labels,
                                      categorical_feature=[])

            # Train LightGBM classifier
            bst = lgb.train(trial_params, train_data#, 
                            #num_boost_round=num_boost_round
                            )

            #
            # TESTING:
            #

            # Testing data
            y_true = labels[test_ids]
            X_test = x_train[test_ids]
            # test_data = lgb.Dataset(X_test, 
            #                           label=y_true,
            #                           categorical_feature=[], 
            #                           reference=train_data)
            # Evaluation for this fold
            y_score = bst.predict(X_test)

            # Metrics
            tune_fold_results['roc_auc'][fold] = roc_auc_score(y_true, y_score)
            tune_fold_results['brier'][fold] = brier_score_loss(y_true, y_score)
            tune_fold_results['logl'][fold] = log_loss(y_true, y_score)
            tune_fold_results['recall'][fold] = recall_score(y_true, (np.array(y_score) > 0.5))
            tune_fold_results['precision'][fold] = precision_score(y_true, (np.array(y_score) > 0.5))
            tune_fold_results['accuracy'][fold] = accuracy_score(y_true, (np.array(y_score) > 0.5))
            tune_fold_results['pr_auc'][fold] = average_precision_score(y_true, y_score)
            # Rdkit implemented metrics:
            score_mat = np.column_stack((y_score, y_true))
            score_mat = score_mat[score_mat[:, 0].argsort()[::-1]] # descending order
            tune_fold_results['bedroc_5'][fold] = CalcBEDROC(score_mat, 1, 5)
            tune_fold_results['bedroc_10'][fold] = CalcBEDROC(score_mat, 1, 10)
            tune_fold_results['bedroc_20'][fold] = CalcBEDROC(score_mat, 1, 20)
            tune_fold_results['ef_1'][fold] = CalcEnrichment(score_mat, 1, [0.01])[0]
            tune_fold_results['abs_ef_1'][fold] = (CalcEnrichment(score_mat, 1, [0.01])[0]
                                * np.sum(score_mat[:,1])/len(score_mat[:,1]))

            # Report running mean
            #session.report({key: np.mean(tune_fold_results[key][:fold+1]) for key in metrics})
            #tune.report({key: np.mean(tune_fold_results[key][:fold+1]) for key in metrics})

        # Print fold tune_fold_results
        #print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS:')
        #print('\n')

        summary = {}

        for m in metrics:
            summary[m] = {}
            summary[m]['mean'] = np.mean(tune_fold_results[m])

        # Report metrics to Ray
        session.report({key: summary[key]['mean'] for key in metrics})
        # tune.report({key: summary[key]['mean'] for key in metrics})