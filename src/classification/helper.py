import numpy as np
from scipy.stats import skew
from itertools import islice
import torch

class HelperFunctions:
    """
    A class containing helper functions.

    Examples
    ========
        >>> a
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """

    def get_parameters(self, 
                       target, 
                       parameter):
        """
        Output the number of actives, inactives, and the
        ratio of inactives to actives for a given target
        and parameter table

        Parameters
        ----------
        """

        n_active = parameter[parameter['Targets'] == target]['Actives'].values[0]
        n_inactive = parameter[parameter['Targets'] == target]['Inactives'].values[0]
        active_aug_multi = int(
                                np.ceil(
                                        parameter[parameter['Targets'] == target]['Inactives/Actives'].values[0]
                                        )
                                )

        return n_active, n_inactive, active_aug_multi

    def make_groups(self,
                    n_active,
                    n_inactive,
                    active_aug_multi=1,
                    inactive_aug_multi=1):
        """
        Output a list which denotes the group memebership
        of each augmented molecule. Augmented copies
        of the same molecule features are assigned to the 
        same group. This is used for stratified group
        k-fold generation.

        Parameters
        ----------
        """

        groups = []

        for i in range(n_active):
            rot_group = active_aug_multi * [i]
            groups.extend(rot_group)

        for i in range(n_active, n_active + n_inactive):
            rot_group = inactive_aug_multi * [i]
            groups.extend(rot_group)

        return groups


    def dim_reduction(self,
                      df, 
                      original_top, 
                      desired_top,
                      n_desired_directions,
                      crit_categories=3,
                      desired_crit_categories=[0, 1, 2],
                      projected_heights=True,
                      esd_indices=False):
        """
        Dimensionally reduce the MAGIC feature vector
        by reducing one of or a combination of:

        * Number of analysed directons
        * Number of top critical points
        * Length of the critical point index
        * Projected heights

        Parameters
        ----------
        df: a pandas dataframe

        original top: int

        desired top: int

        n_disired_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally 
            reduced feature 

        desired_crit_categories: list of ints

        projected_heights: bool 

        esd_indices: bool or list of ints
        """
        reduced_df = df.copy()

        top = int(original_top)
        desired_top = int(desired_top)

        if desired_top > original_top:
            desired_top = int(original_top)
        elif desired_top <= 0:
            desired_top = int(1)

        col = []

        try:
            for n in range(n_desired_directions):
                # Truncated projected heights
                if projected_heights:
                    col.extend(list(np.arange(n * top * (1 + crit_categories),
                                            n * top * (1 + crit_categories) + desired_top)))

                # Truncated indices
                for k in desired_crit_categories:
                    col.extend(list(np.arange(n * top * (1 + crit_categories) + (1 + k) * top,
                                            n * top * (1 + crit_categories) + (1 + k) * top
                                            + desired_top)))
        except TypeError:
            for n in range(*n_desired_directions):# unpack tuple args
                # Truncated projected heights
                if projected_heights:
                    col.extend(list(np.arange(n * top * (1 + crit_categories),
                                            n * top * (1 + crit_categories) + desired_top)))

                # Truncated indices
                for k in desired_crit_categories:
                    col.extend(list(np.arange(n * top * (1 + crit_categories) + (1 + k) * top,
                                            n * top * (1 + crit_categories) + (1 + k) * top
                                            + desired_top)))

        if esd_indices:
            col.extend(esd_indices)

        reduced_df = reduced_df[col]

        return reduced_df


    def baseline_dim_reduction(self,
                      df,
                      n_original_properties=17, 
                      desired_properties=[0, 1, 2],
                      num_quantiles=9):
        """
        Dimensionally reduce the baseline quantile feature
        by reducing:

        * Number of desired properties

        Parameters
        ----------
        df: a pandas dataframe

        desired_properties: list of ints

        """
        reduced_df = df.copy()
        col = []

        for q in range(num_quantiles):
            for p in desired_properties:
                if p < n_original_properties:
                    col_index = q * n_original_properties + p
                    col_index = int(col_index)
                    col.append(col_index)
        
        for p in desired_properties:
            if p >= n_original_properties:
                col_index = (num_quantiles - 1) * n_original_properties  
                col_index = col_index + p 
                col_index = int(col_index)
                col.append(col_index)
        
        print('Columns selected:', col)
        reduced_df = reduced_df[col]

        return reduced_df


    def external_dim_reduction(self,
                      df,
                      desired_properties=False,
                              ):
        """
        Dimensionally reduce the external feature
        by reducing:

        * Number of desired properties

        Parameters
        ----------
        df: a pandas dataframe

        desired_properties: list of ints

        """
        reduced_df = df.copy()

        if desired_properties:
            col = [int(p) for p in desired_properties]
            print('Columns:', reduced_df.columns)
            print('Columns selected:', col)
            reduced_df = reduced_df[col]
        else:
            print('Columns:', reduced_df.columns)
            print('Columns selected: all')

        return reduced_df


    def summary(self,
                df, 
                top, 
                n_directions,
                crit_categories=3,
                integer_categories=[0,1,2],
                projected_heights=True,
                add_num_crit_pts=True,
                add_euler_char=True,
                pure_ect=False,
                sum_over_all_crit_pts=True,
                bin_count_betti=False,
                moments=False):
        """
        Summarise the MAGIC feature vector by taking summary statistics of the
        feature vector across either all directions or all critical points.

        Parameters
        ----------
        df: a pandas dataframe

        top: int

        n_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally 
            reduced feature 

        num_crit_pts: bool 

        euler_char: bool 

        sum_over_all_crit_pts: bool

        bin_count_betti: bool
            Only to be used if there are projected heights and all three
            Betti numbers!  
        """
        # All the input arguments to the function
        # except 'self' and 'sum_over_all_crit_pts'
        arguments = locals().copy()
        del arguments['self']
        del arguments['sum_over_all_crit_pts']
        del arguments['bin_count_betti']
        del arguments['moments']

        # Convert tuple directions to int
        d = arguments['n_directions'] 
        if isinstance(d, tuple):
            N_d = abs( d[1] - d[0] )
            arguments['n_directions'] = int(N_d) 
        # print(arguments)

        if sum_over_all_crit_pts:
            if bin_count_betti:
                X, i = self._summary_all_with_bin_count_betti(**arguments)
                print('bin_count_betti =', bin_count_betti)
            else:
                if moments:
                    X, i = self._summary_all_moments(**arguments)
                else:
                    X, i = self._summary_all(**arguments)
        else:
            X, i = self._summary_top(**arguments)

        summarised_x_tensor, integer_indices = X, i

        return summarised_x_tensor, integer_indices


    def _summary_top(self,
                     df=None, 
                     top=20, 
                     n_directions=12,
                     crit_categories=3,
                     integer_categories=[0,1,2],
                     projected_heights=True,
                     add_num_crit_pts=True,
                     add_euler_char=True,#,
                     pure_ect=False,
                     #sum_over_all_crit_pts=True
                    ):
        """
        Summarise the MAGIC feature vector by taking summary statistics of the
        feature vector across either all directions or all critical points.

        Parameters
        ----------
        df: a pandas dataframe

        top: int

        n_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally 
            reduced feature 

        num_crit_pts: bool 

        euler_char: bool 
        """
        # The length of the critical index
        if projected_heights:
            len_index = 1 + crit_categories
        else:
            len_index = crit_categories

        # Convert the input dataframe to a numpy 2-tensor
        x_2_tensor = df.copy().to_numpy()

        # The number of molecules
        N = x_2_tensor.shape[0]

        # Reshape the 2-tensor into a 4-tensor
        shape = N, n_directions, len_index, top
        x_4_tensor = np.reshape(x_2_tensor, shape)

        # Replace the 0-padded points with NaNs
        # [height, betti-1, r_betti0, betti1, q, logp] = [0,0,0,0,0,0]
        # broadcast the mask
        # padded_index = np.zeros((len_index, 1))
        # zero_mask = x_4_tensor == padded_index
        zero_mask = x_4_tensor == 0
        zero_mask = np.all(zero_mask, axis=2, keepdims=True)
        zero_mask = zero_mask * np.ones_like(x_4_tensor, dtype=bool) # broadcast
        x_4_tensor[zero_mask] = np.nan

        # Count the number of critical points
        # by counting the non-NaNs 
        if add_num_crit_pts:
            crit_mask = ~zero_mask # 4-tensor
            n_crit = np.sum(crit_mask, axis=3) # 3-tensor
            n_crit = np.sum(n_crit, axis=2, keepdims=True) # Keep the shape of a 3-tensor
            n_crit = n_crit / len_index

        # Calculate the 'partial' Euler characteristic
        # sum(Betti_0) - sum(Betti_1)
        if add_euler_char:
            n_0 = np.nansum(x_4_tensor[:, :, 2, :], 
                              axis=2,
                              keepdims=True)
            n_1 = np.nansum(x_4_tensor[:, :, 3, :], 
                              axis=2,
                              keepdims=True)
            euler_char = n_0 + 1 - n_1 # n.b. r_Betti_0 = Betti_0 - 1

        # Reshape the 4-tensor to a 3-tensor
        shape = N, n_directions, top * len_index
        x_3_tensor = np.reshape(x_4_tensor, shape)

        added_feat = 0
        # Concatenate the number of critical points
        if add_num_crit_pts:
            x_3_tensor = np.concatenate((x_3_tensor, n_crit), axis=2)
            added_feat += 1

        # Concatenate the partial Euler characteristic
        if add_euler_char:
            x_3_tensor = np.concatenate((x_3_tensor, euler_char), axis=2)
            added_feat += 1

        print('Started quantile calculation. . .')
        # For speed convert the tensor from a numpy array to a torch tensor
        # to take advantage of torch.nanquantile, which is much faster than
        # the very slow np.nanquantile 
        x_3_tensor = torch.from_numpy(x_3_tensor).to(torch.float32)

        # Calculate the median
        # over the directions axis
        # “ignores” NaN values, computing the quantiles q as if NaN values in 
        # input did not exist. If all values in a reduced row are NaN then the
        # quantiles for that reduction will be NaN
        # medians = np.nanquantile(x_3_tensor, 0.5, axis=1, method='nearest')
        medians = torch.nanquantile(x_3_tensor, 0.5, dim=1, interpolation='nearest')

        # Calculate the 75th quantile
        # over the directions axis
        # up = np.nanquantile(x_3_tensor, 0.75, axis=1, method='nearest')
        up = torch.nanquantile(x_3_tensor, 0.75, dim=1, interpolation='nearest')

        # Calculate the 25th quantile
        # over the directions axis
        # low = np.nanquantile(x_3_tensor, 0.25, axis=1, method='nearest')
        low = torch.nanquantile(x_3_tensor, 0.25, dim=1, interpolation='nearest')

        # Calculate the maximum
        # over the directions axis
        # maximum = np.nanquantile(x_3_tensor, 1, axis=1, method='nearest')
        maximum = torch.nanquantile(x_3_tensor, 1, dim=1, interpolation='nearest')

        # Calculate the minimum
        # over the directions axis
        # minimum = np.nanquantile(x_3_tensor, 0, axis=1, method='nearest')
        minimum = torch.nanquantile(x_3_tensor, 0, dim=1, interpolation='nearest')

        # Calculate the 90th quantile
        # over the directions axis
        # ninety = np.nanquantile(x_3_tensor, 0.90, axis=1, method='nearest')
        # ninety = torch.nanquantile(x_3_tensor, 0.90, dim=1, interpolation='nearest')

        # Calculate the 10th quantile
        # over the directions axis
        # ten = np.nanquantile(x_3_tensor, 0.10, axis=1, method='nearest')
        # ten = torch.nanquantile(x_3_tensor, 0.10, dim=1, interpolation='nearest')

        # Calculate the mean
        # over the directions axis
        # mean = np.nanmean(x_3_tensor, axis=1)

        # Calculate the sum over the direction axis
        # sums = np.nansum(x_3_tensor, axis=1)

        # Concatenate all summaries
        summaries = (maximum,
                     #ninety,
                     up,
                     medians, 
                     low,
                     #ten,
                     minimum
                     )
        # summarised_x_tensor = np.concatenate(summaries, axis=1)
        summarised_x_tensor = torch.cat(summaries, dim=1)

        # Replace NaNs with 0
        # Not necessary as LightGBM treats NaNs as missing values and can
        # handle them
        # nan_mask = torch.isnan(summarised_x_tensor)
        # summarised_x_tensor[nan_mask] = 0.0

        # Convert torch tensor back to numpy array
        summarised_x_tensor = summarised_x_tensor.detach().cpu().numpy()
        print('Finished quantile calculation!\n')

        # Record which columns contain integer features
        integer_indices = []
        num_summaries = len(summaries)

        # Integer columns
        for s in range(num_summaries):
            if projected_heights:
                start = s * (top * (1 + crit_categories) + added_feat)
                for k in integer_categories:
                    integer_indices.extend(list(np.arange(start + (1 + k) * top,
                                                          start + (1 + k) * top + top)))
            else:
                start = s * (top * crit_categories + added_feat)
                for k in integer_categories:                   
                    integer_indices.extend(list(np.arange(start + k * top,
                                                          start + k * top + top)))
        
        # Change dtype of element sin integer indices to int32 for 
        # compatability witht the LightGBM train function
        integer_indices = [int(i) for i in integer_indices]
        
        print(summarised_x_tensor.shape,
              num_summaries * (top * (1+crit_categories) + added_feat))
        print(integer_indices)

        return summarised_x_tensor, integer_indices

    def _summary_all(self,
                     df=None, 
                     top=20, 
                     n_directions=12,
                     crit_categories=3,
                     integer_categories=[0,1,2],
                     projected_heights=True,
                     add_num_crit_pts=True,
                     add_euler_char=False,
                     pure_ect=False,
                     #sum_over_all_crit_pts=True
                    ):
        """
        Summarise the MAGIC feature vector by taking summary statistics of the
        feature vector across either all directions or all critical points.

        Parameters
        ----------
        df: a pandas dataframe

        top: int

        n_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally 
            reduced feature 

        num_crit_pts: bool 

        euler_char: bool 
        """
        # The length of the critical index
        if projected_heights:
            len_index = 1 + crit_categories
        else:
            print('projected_heights =', projected_heights)
            len_index = crit_categories

        # Convert the input dataframe to a numpy 2-tensor
        # x_2_tensor = df.to_numpy()
        x_2_tensor = df.copy().to_numpy()

        # The number of molecules
        N = x_2_tensor.shape[0]

        # Reshape the 2-tensor into a 4-tensor
        shape = N, n_directions, len_index, top
        x_4_tensor = np.reshape(x_2_tensor, shape)

        # Replace the 0-padded points with NaNs
        # [height, betti-1, r_betti0, betti1, q, logp] = [0,0,0,0,0,0]
        # broadcast the mask
        # padded_index = np.zeros((len_index, 1))
        # zero_mask = x_4_tensor == padded_index
        x_4_tensor_zero_pad = x_4_tensor.copy()
        zero_mask = x_4_tensor == 0
        zero_mask = np.all(zero_mask, axis=2, keepdims=True)
        zero_mask = zero_mask * np.ones_like(x_4_tensor, dtype=bool) # broadcast
        x_4_tensor[zero_mask] = np.nan

        # Count the number of critical points
        # by counting the non-NaNs
        # pad the tensor for later concatenation 
        if add_num_crit_pts:
            crit_mask = ~zero_mask # 4-tensor
            n_crit = np.sum(crit_mask, axis=3) # 3-tensor
            n_crit = np.sum(n_crit, axis=2, keepdims=True) # Keep the shape of a 3-tensor
            n_crit = n_crit / len_index
            pad_size = n_directions * (top - 1)
            n_crit = np.pad(n_crit,
                            pad_width=((0,0),(0,pad_size),(0,0)),
                            mode='constant',
                            constant_values=np.nan) 

        # Calculate the 'partial' Euler characteristic
        # sum(Betti_0) - sum(Betti_1)
        # pad the tensor for later concatenation
        if add_euler_char:
            crit_mask = ~zero_mask
            # n_m1 = np.nansum(x_4_tensor[:, :, 1, :],
            #                   axis=2,
            #                   keepdims=True)
            # n_0 = np.nansum(x_4_tensor[:, :, 2, :], 
            #                   axis=2,
            #                   keepdims=True)
            # n_1 = np.nansum(x_4_tensor[:, :, 3, :], 
            #                   axis=2,
            #                   keepdims=True)
            # euler_char = n_m1 - n_0 + n_1 # 
            # pad_size = n_directions * (top - 1)
            # euler_char = np.pad(euler_char,
            #                     pad_width=((0,0),(0,pad_size),(0,0)),
            #                     mode='constant',
            #                     constant_values=np.nan)
            # The following tensors have shape
            # N, n_directions, top
            n_m1 = np.cumsum(x_4_tensor_zero_pad[:, :, 1, :],
                                axis=2)
            n_0 = np.cumsum(x_4_tensor_zero_pad[:, :, 2, :],
                               axis=2)
            n_1 = np.cumsum(x_4_tensor_zero_pad[:, :, 3, :],
                               axis=2)
            euler_char = n_m1 - n_0 + n_1

            # Replace the zero-padded entries with np.nans
            # this removes spurious rows of zeros from the ECT
            # The mask must be slized since the euler_char is
            # generated from slices of the x_4_tensor
            euler_char[zero_mask[:, :, 0, :]] = np.nan

            # Reshape tensor
            shape = N, n_directions * top
            euler_char = np.reshape(euler_char, shape)

            # Add a new axis at the end to make it
            # have the same number of dimensions as the
            # later x_3_tensor it will be concatenated with
            euler_char = euler_char[:, :, np.newaxis] 

        # Transpose the last two dimensions of the 4-tensor to give a shape of
        # intances x directions x top x index
        x_4_tensor_T = np.transpose(x_4_tensor, (0, 1, 3, 2)) 

        # Reshape the 4-tensor to a 3-tensor
        shape = N, n_directions * top, len_index
        x_3_tensor = np.reshape(x_4_tensor_T, shape)

        # Concatenate the number of critical points
        if add_num_crit_pts:
            x_3_tensor = np.concatenate((x_3_tensor, n_crit), axis=2)
            crit_categories += 1
            len_index += 1

        # Concatenate the partial Euler characteristic
        if add_euler_char:
            x_3_tensor = np.concatenate((x_3_tensor, euler_char), axis=2)
            crit_categories += 1
            len_index += 1
            if pure_ect:
                # Keep only projected heights and ECT to depth
                # Remove the columns associated with the reduced Betti
                # numbers
                x_3_tensor = x_3_tensor[:, :, [0, 4]]
                crit_categories -= 3
                len_index -= 3
                print(x_3_tensor.shape)

        print('Started quantile calculation. . .')
        # For speed convert the tensor from a numpy array to a torch tensor
        # to take advantage of torch.nanquantile, which is much faster than
        # the very slow np.nanquantile 
        x_3_tensor = torch.from_numpy(x_3_tensor).to(torch.float32)

        # Calculate the median
        # over the directions axis
        # “ignores” NaN values, computing the quantiles q as if NaN values in 
        # input did not exist. If all values in a reduced row are NaN then the
        # quantiles for that reduction will be NaN
        # medians = np.nanquantile(x_3_tensor, 0.5, axis=1, method='nearest')
        medians = torch.nanquantile(x_3_tensor, 0.5, dim=1, interpolation='nearest')

        # Calculate the 75th quantile
        # over the directions axis
        # up = np.nanquantile(x_3_tensor, 0.75, axis=1, method='nearest')
        up = torch.nanquantile(x_3_tensor, 0.75, dim=1, interpolation='nearest')

        # Calculate the 25th quantile
        # over the directions axis
        # low = np.nanquantile(x_3_tensor, 0.25, axis=1, method='nearest')
        low = torch.nanquantile(x_3_tensor, 0.25, dim=1, interpolation='nearest')

        # Calculate the maximum
        # over the directions axis
        # maximum = np.nanquantile(x_3_tensor, 1, axis=1, method='nearest')
        maximum = torch.nanquantile(x_3_tensor, 1, dim=1, interpolation='nearest')

        # Calculate the minimum
        # over the directions axis
        # minimum = np.nanquantile(x_3_tensor, 0, axis=1, method='nearest')
        minimum = torch.nanquantile(x_3_tensor, 0, dim=1, interpolation='nearest')

        # Calculate the 90th quantile
        # over the directions axis
        # ninety = np.nanquantile(x_3_tensor, 0.90, axis=1, method='nearest')
        ninety = torch.nanquantile(x_3_tensor, 0.90, dim=1, interpolation='nearest')

        # Calculate the 10th quantile
        # over the directions axis
        # ten = np.nanquantile(x_3_tensor, 0.10, axis=1, method='nearest')
        ten = torch.nanquantile(x_3_tensor, 0.10, dim=1, interpolation='nearest')

        # Calculate the 60th quantile
        # over the directions axis
        # sixty = np.nanquantile(x_3_tensor, 0.60, axis=1, method='nearest')
        sixty = torch.nanquantile(x_3_tensor, 0.60, dim=1, interpolation='nearest')

        # Calculate the 10th quantile
        # over the directions axis
        # forty = np.nanquantile(x_3_tensor, 0.40, axis=1, method='nearest')
        forty = torch.nanquantile(x_3_tensor, 0.40, dim=1, interpolation='nearest')

        # Concatenate all summaries
        summaries = (maximum,
                     ninety,
                     up,
                     sixty, 
                     medians,
                     forty, 
                     low,
                     ten,
                     minimum,
                     )
        
        # summarised_x_tensor = np.concatenate(summaries, axis=1)
        summarised_x_tensor = torch.cat(summaries, dim=1)

        # Replace NaNs with 0
        # Not necessary as LightGBM treats NaNs as missing values and can
        # handle them
        # nan_mask = torch.isnan(summarised_x_tensor)
        # summarised_x_tensor[nan_mask] = 0.0

        # Convert torch tensor back to numpy array
        summarised_x_tensor = summarised_x_tensor.detach().cpu().numpy()
        print('Finished quantile calculation!\n')

        # Record which columns contain integer features
        integer_indices = []
        num_summaries = len(summaries)

        # Integer columns
        for s in range(num_summaries):
            if projected_heights:
                start = s * (1 + crit_categories)
                for k in integer_categories:
                    integer_indices.extend(list(np.arange(start + (1 + k),
                                                          start + (1 + k) + 1)))
            else:
                start = s * crit_categories
                for k in integer_categories:                   
                    integer_indices.extend(list(np.arange(start + k,
                                                          start + k + 1)))
        
        # Change dtype of element sin integer indices to int32 for 
        # compatability witht the LightGBM train function
        integer_indices = [int(i) for i in integer_indices]
        
        # print(summarised_x_tensor.shape,
        #       num_summaries * (1+crit_categories))
        # print(integer_indices)
        # print([type(i) for i in integer_indices])

        return summarised_x_tensor, integer_indices

    def _summary_all_with_bin_count_betti(self,
                                          df=None, 
                                          top=20, 
                                          n_directions=12,
                                          crit_categories=3,
                                          integer_categories=[0,1,2],
                                          projected_heights=True,
                                          add_num_crit_pts=True,
                                          add_euler_char=False,
                                          #sum_over_all_crit_pts=True
                                          ):
        """
        Summarise the MAGIC feature vector by taking summary statistics of the
        feature vector across either all directions or all critical points.

        However, bin counts of the betti numbers are taken. This only works
        if there are projected heights and all three betti numbers!

        Parameters
        ----------
        df: a pandas dataframe

        top: int

        n_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally 
            reduced feature 

        num_crit_pts: bool 

        euler_char: bool 
        """
        # The length of the critical index
        if projected_heights:
            len_index = 1 + crit_categories
        else:
            print('projected_heights =', projected_heights)
            len_index = crit_categories

        # Convert the input dataframe to a numpy 2-tensor
        # x_2_tensor = df.to_numpy()
        x_2_tensor = df.copy().to_numpy()

        # The number of molecules
        N = x_2_tensor.shape[0]

        # Reshape the 2-tensor into a 4-tensor
        shape = N, n_directions, len_index, top
        x_4_tensor = np.reshape(x_2_tensor, shape)

        # Replace the 0-padded points with NaNs
        # [height, betti-1, r_betti0, betti1, q, logp] = [0,0,0,0,0,0]
        # broadcast the mask
        # padded_index = np.zeros((len_index, 1))
        # zero_mask = x_4_tensor == padded_index
        x_4_tensor_zero_pad = x_4_tensor.copy()
        zero_mask = x_4_tensor == 0
        zero_mask = np.all(zero_mask, axis=2, keepdims=True)
        zero_mask = zero_mask * np.ones_like(x_4_tensor, dtype=bool) # broadcast
        x_4_tensor[zero_mask] = np.nan

        # Count the number of critical points
        # by counting the non-NaNs
        # pad the tensor for later concatenation 
        if add_num_crit_pts:
            crit_mask = ~zero_mask # 4-tensor
            n_crit = np.sum(crit_mask, axis=3) # 3-tensor
            n_crit = np.sum(n_crit, axis=2, keepdims=True) # Keep the shape of a 3-tensor
            n_crit = n_crit / len_index
            pad_size = n_directions * (top - 1)
            n_crit = np.pad(n_crit,
                            pad_width=((0,0),(0,pad_size),(0,0)),
                            mode='constant',
                            constant_values=np.nan) 

        # Calculate the 'partial' Euler characteristic
        # sum(Betti_0) - sum(Betti_1)
        if add_euler_char:
            crit_mask = ~zero_mask
            n_m1 = np.cumsum(x_4_tensor_zero_pad[:, :, 1, :],
                                axis=2)
            n_0 = np.cumsum(x_4_tensor_zero_pad[:, :, 2, :],
                               axis=2)
            n_1 = np.cumsum(x_4_tensor_zero_pad[:, :, 3, :],
                               axis=2)
            euler_char = n_m1 - n_0 + n_1
            # old shape = N, n_directions, len_index, top
            # new shape = N, n_directions, top
            euler_char[zero_mask[:, :, 0, :]] = np.nan
            print('x_4:', x_4_tensor_zero_pad.shape, 'euler:', euler_char.shape)
            shape = N, n_directions * top
            euler_char = np.reshape(euler_char, shape)
            # Add a new axis at the end
            euler_char = euler_char[:, :, np.newaxis]

        # Transpose the last two dimensions of the 4-tensor to give a shape of
        # intances x directions x top x index
        x_4_tensor_T = np.transpose(x_4_tensor, (0, 1, 3, 2)) 

        # Reshape the 4-tensor to a 3-tensor
        shape = N, n_directions * top, len_index
        x_3_tensor = np.reshape(x_4_tensor_T, shape)

        # Concatenate the number of critical points
        if add_num_crit_pts:
            x_3_tensor = np.concatenate((x_3_tensor, n_crit), axis=2)
            crit_categories += 1
            len_index += 1

        # Concatenate the partial Euler characteristic
        if add_euler_char:
            x_3_tensor = np.concatenate((x_3_tensor, euler_char), axis=2)
            crit_categories += 1
            len_index += 1

        # Count of Betti numbers
        # Betti_-1: 0, 1
        # Betti_0: (0,) 1, 2, 3, 4, (5, 6, . . .) 
        # Betti_1: (0), 1, 2, 3, 4, (5, . . .)
        # 2 + 4 + 4 = 10 (15) features

        # Remove NaNs from Betti numbers and set to type int
        x_3_tensor_only_betti = x_3_tensor[:, :, 1:4] 
        mask = np.isnan(x_3_tensor_only_betti)
        x_3_tensor_only_betti[mask] = -999
        x_3_tensor_only_betti = x_3_tensor_only_betti.astype(int)

        # Count Betti numbers        
        betti_minus_1_count = ()
        for b in [0, 1]:
            # slicing by a list preserves dimensions of the array
            # shape N, n_directions * top, 1
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [0]]
            # mask betti values
            betti_value_mask = x_3_tensor_one_betti == b
            # count occurences of the betti values
            # shape N, 1
            count = np.sum(betti_value_mask ,
                           axis=1,
                           keepdims=False)#True
            betti_minus_1_count += (count, )

        # Normalise to change counts to a pmf
        n_crit = betti_minus_1_count[0] + betti_minus_1_count[1]
        betti_minus_1_count = n_crit, betti_minus_1_count[1] / n_crit # element-wise

        betti_0_count = ()
        for b in [1, 2, 3, 4]:#0-8
            # slicing by a list preserves dimensions of the array
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [1]]
            betti_value_mask = x_3_tensor_one_betti == b
            count = np.sum(betti_value_mask ,
                           axis=1,
                           keepdims=False)
            count = count / n_crit
            betti_0_count += (count, )

        betti_1_count = ()
        for b in [1, 2, 3, 4]:#0-6
            # slicing by a list preserves dimensions of the array
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [2]]
            betti_value_mask = x_3_tensor_one_betti == b
            count = np.sum(betti_value_mask,
                           axis=1,
                           keepdims=False)
            count = count / n_crit
            betti_1_count += (count, )

        # Concatenate the betti counts together
        # shape: N, 10 (15)
        counts = betti_minus_1_count + betti_0_count + betti_1_count
        betti_counts = np.concatenate(counts, axis=1)

        print('Started quantile calculation. . .')
        # Exclude Betti numbers from tensor
        mask = np.full(len_index, True, dtype=bool)
        if projected_heights:
            mask[1:1+3] = False
        else:
            mask[0:0+3] = False

        x_3_tensor = x_3_tensor[:, :, mask]

        # For speed convert the tensor from a numpy array to a torch tensor
        # to take advantage of torch.nanquantile, which is much faster than
        # the very slow np.nanquantile 
        x_3_tensor = torch.from_numpy(x_3_tensor).to(torch.float32)

        # Calculate the p-quantiles
        # over the directions (1) axis
        # “ignores” NaN values, computing the quantiles q as if NaN values in 
        # input did not exist. If all values in a reduced row are NaN then the
        # quantiles for that reduction will be NaN
        p = torch.tensor([1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0])
        summarised_x_3tensor = torch.nanquantile(x_3_tensor, p, dim=1, interpolation='nearest')

        # Reshape tensor from len(p), N, (len_index - 3)
        # to N, len(p), (len_index - 3)
        summarised_x_3tensor = torch.swapaxes(summarised_x_3tensor, 0, 1)

        # Reshape 3-tensor from N, len(p), (len_index - 3)
        # to a 2-tensor N, (len_index - 3) * len(p)
        shape = N, (len_index - 3) * len(p)
        summarised_x_tensor = torch.reshape(summarised_x_3tensor, shape)

        # Replace NaNs with 0
        # Not necessary as LightGBM treats NaNs as missing values and can
        # handle them
        # nan_mask = torch.isnan(summarised_x_tensor)
        # summarised_x_tensor[nan_mask] = 0.0

        # Convert torch tensor back to numpy array
        summarised_x_tensor = summarised_x_tensor.detach().cpu().numpy()
        print('Finished quantile calculation!\n')

        summarised_x_2tensor_with_counts = np.concatenate((summarised_x_tensor, 
                                                           betti_counts), 
                                                          axis=1)

        # Record which columns contain integer features
        integer_indices = []
        num_summaries = len(p)

        # Integer columns
        for s in range(num_summaries):
            if projected_heights:
                start = s * (1 + crit_categories)
                for k in integer_categories:
                    integer_indices.extend(list(np.arange(start + (1 + k),
                                                          start + (1 + k) + 1)))
            else:
                start = s * crit_categories
                for k in integer_categories:                   
                    integer_indices.extend(list(np.arange(start + k,
                                                          start + k + 1)))
        
        # Change dtype of element sin integer indices to int32 for 
        # compatability witht the LightGBM train function
        integer_indices = [int(i) for i in integer_indices]
        
        # print(summarised_x_tensor.shape,
        #       num_summaries * (1+crit_categories))
        # print(integer_indices)
        # print([type(i) for i in integer_indices])

        return summarised_x_2tensor_with_counts, integer_indices

    def _summary_all_moments(self,
                     df=None,
                     top=20,
                     n_directions=12,
                     crit_categories=3,
                     integer_categories=[0,1,2],
                     projected_heights=True,
                     add_num_crit_pts=True,
                     add_euler_char=False,
                     #sum_over_all_crit_pts=True
                    ):
        """
        Summarise the MAGIC feature vector by taking summary statistics of the
        feature vector across either all directions or all critical points.

        Parameters
        ----------
        df: a pandas dataframe

        top: int

        n_directions: int or 2-tuple of ints (start, end)

        crit_categories: int
            The total length of the index in the non-dimensionally
            reduced feature

        num_crit_pts: bool

        euler_char: bool
        """
        # The length of the critical index
        if projected_heights:
            len_index = 1 + crit_categories
        else:
            print('projected_heights =', projected_heights)
            len_index = crit_categories

        # Convert the input dataframe to a numpy 2-tensor
        # x_2_tensor = df.to_numpy()
        x_2_tensor = df.copy().to_numpy()

        # The number of molecules
        N = x_2_tensor.shape[0]

        # Reshape the 2-tensor into a 4-tensor
        shape = N, n_directions, len_index, top
        x_4_tensor = np.reshape(x_2_tensor, shape)

        # Replace the 0-padded points with NaNs
        # [height, betti-1, r_betti0, betti1, q, logp] = [0,0,0,0,0,0]
        # broadcast the mask
        # padded_index = np.zeros((len_index, 1))
        # zero_mask = x_4_tensor == padded_index
        x_4_tensor_zero_pad = x_4_tensor.copy()
        zero_mask = x_4_tensor == 0
        zero_mask = np.all(zero_mask, axis=2, keepdims=True)
        zero_mask = zero_mask * np.ones_like(x_4_tensor, dtype=bool) # broadcast
        x_4_tensor[zero_mask] = np.nan

        # Count the number of critical points
        # by counting the non-NaNs
        if add_num_crit_pts:
            crit_mask = ~zero_mask # 4-tensor
            n_crit = np.sum(crit_mask, axis=3) # 3-tensor
            n_crit = np.sum(n_crit, axis=2, keepdims=True) # Keep the shape of a 3-tensor
            n_crit = n_crit / len_index
            # pad the tensor for later concatenation
            pad_size = n_directions * (top - 1)
            n_crit = np.pad(n_crit,
                            pad_width=((0,0),(0,pad_size),(0,0)),
                            mode='constant',
                            constant_values=np.nan)

        # Calculate the 'partial' Euler characteristic
        # sum(Betti_-1) - sum(Betti_0) + sum(Betti_1)
        # pad the tensor for later concatenation
        if add_euler_char:
            crit_mask = ~zero_mask
            n_m1 = np.cumsum(x_4_tensor_zero_pad[:, :, 1, :],
                                axis=2)
            n_0 = np.cumsum(x_4_tensor_zero_pad[:, :, 2, :],
                               axis=2)
            n_1 = np.cumsum(x_4_tensor_zero_pad[:, :, 3, :],
                               axis=2)
            euler_char = n_m1 - n_0 + n_1
            # old shape = N, n_directions, len_index, top
            # new shape = N, n_directions, top
            euler_char[zero_mask[:, :, 0, :]] = np.nan
            print('x_4:', x_4_tensor_zero_pad.shape, 'euler:', euler_char.shape)
            shape = N, n_directions * top
            euler_char = np.reshape(euler_char, shape)
            # Add a new axis at the end
            euler_char = euler_char[:, :, np.newaxis]

        # Transpose the last two dimensions of the 4-tensor to give a shape of
        # intances x directions x top x index
        x_4_tensor_T = np.transpose(x_4_tensor, (0, 1, 3, 2))

        # Reshape the 4-tensor to a 3-tensor
        shape = N, n_directions * top, len_index
        x_3_tensor = np.reshape(x_4_tensor_T, shape)

        # Concatenate the number of critical points
        if add_num_crit_pts:
            x_3_tensor = np.concatenate((x_3_tensor, n_crit), axis=2)
            crit_categories += 1
            len_index += 1

        # Concatenate the partial Euler characteristic
        if add_euler_char:
            x_3_tensor = np.concatenate((x_3_tensor, euler_char), axis=2)
            crit_categories += 1
            len_index += 1

        print('Started summary calculation. . .')

        # Count of Betti numbers
        # Betti_-1: (0,) 1
        # Betti_0: (0,) 1, 2, 3, 4, (5, 6, ...)
        # Betti_1: (0,) 1, 2, 3, 4, (5, ...)
        # 1 + 4 + 4 = 9 (15) features
        # Do not need to count 0s due to this being implicit?
        # from the total number of critical points mean

        # Remove NaNs from Betti numbers and set to type int
        x_3_tensor_only_betti = x_3_tensor.copy()
        x_3_tensor_only_betti = x_3_tensor_only_betti[:, :, 1:4]
        mask = np.isnan(x_3_tensor_only_betti)
        x_3_tensor_only_betti[mask] = -999
        x_3_tensor_only_betti = x_3_tensor_only_betti.astype(int)

        # Count Betti numbers
        betti_minus_1_count = ()
        for b in [1]:
            # slicing by a list preserves dimensions of the array
            # shape N, n_directions * top, 1
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [0]]
            # mask betti values
            betti_value_mask = x_3_tensor_one_betti == b
            # count occurences of the betti values
            # shape N, 1
            count = np.sum(betti_value_mask ,
                           axis=1,
                           keepdims=False)#True
            betti_minus_1_count += (count, )

        betti_0_count = ()
        for b in [1, 2, 3, 4]:#5
            # slicing by a list preserves dimensions of the array
            # shape N, n_directions * top, 1
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [1]]
            betti_value_mask = x_3_tensor_one_betti == b
            # count occurences of the betti values
            # shape N, 1
            count = np.sum(betti_value_mask ,
                           axis=1,
                           keepdims=False)
            betti_0_count += (count, )

        betti_1_count = ()
        for b in [1, 2, 3, 4]:#5
            # slicing by a list preserves dimensions of the array
            # shape N, n_directions * top, 1
            x_3_tensor_one_betti = x_3_tensor_only_betti[:, :, [2]]
            betti_value_mask = x_3_tensor_one_betti == b
            # count occurences of the betti values
            # shape N, 1
            count = np.sum(betti_value_mask,
                           axis=1,
                           keepdims=False)
            betti_1_count += (count, )

        # Concatenate Betti counts into one array of 
        # shape N, 9
        counts = betti_minus_1_count + betti_0_count + betti_1_count
        betti_counts = np.concatenate(counts, axis=1)#2

        # Tensor excluding betti numbers
        mask = np.full(len_index, True, dtype=bool)
        if projected_heights:
            mask[1:1+3] = False
        else:
            mask[0:0+3] = False

        x_3_tensor_no_betti = x_3_tensor[:, :, mask]

        # Calculate means ignoring NaNs
        mean = np.nanmean(x_3_tensor_no_betti, axis=1)

        # Calculate standard deviations ignoring NaNs
        # unbiased correction 1/(N - 1) * 
        sd = np.nanstd(x_3_tensor_no_betti, ddof=1, axis=1)

        # Calculate skews ignoring NaNs
        skews = skew(x_3_tensor_no_betti, bias=False, nan_policy='omit', axis=1)

        # Calculate medians ignoring NaNs
        median = np.nanmedian(x_3_tensor_no_betti, axis=1)

        # Calculate medians ignoring NaNs
        maximum = np.nanmax(x_3_tensor_no_betti, axis=1)

        # Calculate medians ignoring NaNs
        minimum = np.nanmin(x_3_tensor_no_betti, axis=1)

        # For speed convert the tensor from a numpy array to a torch tensor
        # to take advantage of torch.nanquantile, which is much faster than
        # the very slow np.nanquantile
        # x_3_tensor = torch.from_numpy(x_3_tensor).to(torch.float32)

        # Calculate the quantiles
        # over the directions axis
        # “ignores” NaN values, computing the quantiles q as if NaN values in
        # input did not exist. If all values in a reduced row are NaN then the
        # quantiles for that reduction will be NaN
        # p = torch.tensor([1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0])
        # summarised_x_2tensor = torch.nanquantile(x_3_tensor, p, dim=1, interpolation='nearest')
        # print(summarised_x_2tensor.shape)
        # summarised_x_2tensor = torch.swapaxes(summarised_x_2tensor, 0, 1)
        # shape = N, len_index * len(p)
        # summarised_x_tensor = torch.reshape(summarised_x_2tensor, shape)

        # summarised_x_tensor =  torch.flatten(summarised_x_2tensor,
        #                                      start_dim=0, end_dim=0)

        # Replace NaNs with 0
        # Not necessary as LightGBM treats NaNs as missing values and can
        # handle them
        # nan_mask = torch.isnan(summarised_x_tensor)
        # summarised_x_tensor[nan_mask] = 0.0

        # Convert torch tensor back to numpy array
        # summarised_x_tensor = summarised_x_tensor.detach().cpu().numpy()

        # Concatenate quantiles
        summaries = (
                    mean, 
                    sd, 
                    # skews,
                    # median,
                    # maximum,
                    # minimum, 
                    betti_counts)
        summarised_x_tensor = np.concatenate(summaries, axis=1)
        print('Finished summary calculation!\n')

        return summarised_x_tensor, None

    def un_aug_list(self,
                    aug_list,
                    n_active,
                    n_inactive,
                    active_aug_multi=1,
                    inactive_aug_multi=1):
        """
        Unaugment a list of indices.
        """

        un_aug_list = []

        for i in aug_list:
            if i <= n_active:
                if i%active_aug_multi == 0:
                    un_aug_list.append(i)
            elif n_active < i:
                if i%inactive_aug_multi == 0:
                    un_aug_list.append(i)

        return un_aug_list
    
    def un_aug_test_splits(self,
                           cv, 
                           x, y, 
                           groups, 
                           target_details):
        """
        Unaugment the test indices but not the train indices.
        """

        td = target_details
        train_idx_list, un_aug_test_idx_list = [], []

        for (train, test) in cv.split(x, y, groups):
            train_idx_list.append(train)
            un_aug_test = self.un_aug_list(test, td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_test_idx_list.append(un_aug_test)

        un_aug_iter = zip(train_idx_list, un_aug_test_idx_list)

        return un_aug_iter
    
    def un_aug_train_and_test_splits(self,
                           cv, 
                           x, y, 
                           groups, 
                           target_details):
        """
        Unaugment the train and test indices.
        """

        td = target_details
        un_aug_train_idx_list, un_aug_test_idx_list = [], []

        for (train, test) in cv.split(x, y, groups):
            un_aug_train = self.un_aug_list(train, td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_train_idx_list.append(un_aug_train)

            un_aug_test = self.un_aug_list(test, td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_test_idx_list.append(un_aug_test)

        un_aug_iter = zip(un_aug_train_idx_list, un_aug_test_idx_list)

        return un_aug_iter

    def un_aug_inner_val_splits(self,
                                cv, 
                                x_train, y_train, 
                                groups_train, 
                                target_details,
                                train_idx):
        """
        Unaugment the inner validation indices but not the inner train indices.
        """

        td = target_details
        inner_train_idx_list, un_aug_val_idx_list = [], []

        for (inner_train, val) in cv.split(x_train, y_train, groups_train):
            # Train indices in terms of outer indices
            # Defined such that
            # x_inner_train = x[inner_train_idx]
            # NOT x_inner_train = x[train_idx][inner_train_idx]
            inner_train_idx_list.append(train_idx[inner_train])

            # Validation indices in terms of outer indices
            un_aug_val = self.un_aug_list(train_idx[val], td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_val_idx_list.append(un_aug_val)

        un_aug_iter = zip(inner_train_idx_list, un_aug_val_idx_list)

        return un_aug_iter
    
    def un_aug_train_and_test_splits(self,
                                     cv, 
                                     x, y, 
                                     groups, 
                                     target_details):
        """
        Unaugment the test indices and the train indices.
        """

        td = target_details
        un_aug_train_idx_list, un_aug_test_idx_list = [], []

        for (train, test) in cv.split(x, y, groups):
            un_aug_train = self.un_aug_list(train, td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_train_idx_list.append(un_aug_train)

            un_aug_test = self.un_aug_list(test, td[0] * td[2], td[1], active_aug_multi=td[2])
            un_aug_test_idx_list.append(un_aug_test)

        un_aug_iter = zip(un_aug_train_idx_list, un_aug_test_idx_list)

        return un_aug_iter
    
    def means_of_slices(self,
                        iterable, 
                        slice_size):
        """
        Calculate the means of slices.
        """

        iterator = iter(iterable)

        while True:
            slice = list(islice(iterator, slice_size))
            if slice:
                yield sum(slice)/len(slice)
            else:
                return

    def medians_of_slices(self,
                          iterable, 
                          slice_size):
        """
        Calculate the medians of slices.
        """

        iterator = iter(iterable)

        while True:
            slice = list(islice(iterator, slice_size))
            if slice:
                yield np.median(slice)
            else:
                return
            
    def TTA(self,
            scores,
            n_active,
            n_inactive,
            active_aug_multi=1,
            inactive_aug_multi=1,
            method='mean'):
        """
        Carry out test-time-augmentation on a list
        of scores.
        """

        scores = scores.tolist()

        active_probs = scores[0:n_active]
        inactive_probs = scores[n_active:n_active + n_inactive]

        if method == 'mean':
            tta_probs = list(self.means_of_slices(active_probs, active_aug_multi))
            tta_probs.extend(list(self.means_of_slices(inactive_probs, inactive_aug_multi)))
        elif method == 'median':
            tta_probs = list(self.medians_of_slices(active_probs, active_aug_multi))
            tta_probs.extend(list(self.medians_of_slices(inactive_probs, inactive_aug_multi)))
        else:
            raise ValueError("method must be one of 'mean' or 'median'")

        return tta_probs