from sklearn.decomposition import PCA
from scipy.stats import skew
import numpy as np


class DataAlignment:
    """
    Aligns point cloud data with the principal component axes 
    in such a way that there is positive skew along the first
    two principal component axes.

    Examples
    ========
        >>> a = DataAlignment()
        >>> a.align(list_df)
        >>> a.aligned_list_df
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """
        # self.list_df = None
        # self.aligned_list_df = None
        # self.groups = None

    def align(self, list_df):
        """
        Align the list of point clouds using pca.

        Parameters
        ===========
        list_df: A list
            A list of panda data frames of point clouds and VdW radii

        Returns
        ========
        aligned_df: a pandas data frame
            A data frame augmented by the rotated copies of the point clouds
        """

        self.list_df = list_df

        # Generate the list of aligned dataframes
        aligned_list_df = [self._alignSingle(df) for df in list_df]
        groups = [idx for idx, df in enumerate(list_df)]

        self.aligned_list_df = aligned_list_df
        self.groups = groups

        return aligned_list_df

    def _alignSingle(self, df):
        """
        Align the a single point cloud using pca.

        Parameters
        ===========
        df: A pandas dataframe
            A panda data frame containing a point cloud and features

        Returns
        ========
        aligned_df: a pandas data frame
            A data frame augmented by the rotated copies of the point clouds
        """

        # Extract the point cloud
        X = df[ ["x", "y", "z"] ].to_numpy()

        # Set the random state and perform the PCA
        pca = PCA(random_state=2024)
        pca.fit(X)

        # Find the determinant of the matrix of eigenvectors 
        # (also called the loadings matrix or principal axes matrix)
        determinant = np.linalg.det(pca.components_)

        # Enforce that the matrix is a proper rotation.
        # Note that for scikit-learn PCA, each row of the matrix 
        # is an eigenvector (not each column)
        if determinant < 0:
            # Flip the sign of the eigenvector with the
            # lowest eigenvalue
            pca.components_[-1, :] *= -1
        else:
            pass

        # Align the point cloud with the PCA axes
        # via a proper rotation
        pca_X = pca.transform(X)

        # Calculate skew of the point cloud along the PCA axes.
        paxis_skews = skew(pca_X, axis=0)

        # Find the rotation matrix that transforms the rotated point cloud 
        # such that there is positive skew along the first two principal axes.
        if (paxis_skews[0] < 0) and (paxis_skews[1] < 0):
            R = np.diag([-1, -1, 1]) # z-axis pi-rotation
        elif (paxis_skews[0] < 0) and (paxis_skews[1] >= 0):
            R = np.diag([-1, 1, -1]) # y-axis pi-rotation
        elif (paxis_skews[0] >= 0) and (paxis_skews[1] < 0):
            R = np.diag([1, -1, -1]) # x-axis pi-rotation
        else:
            R = np.diag([1, 1, 1]) # identity

        # Orient the rotated point cloud such that there is positive skew along 
        # the first two principal axes.
        aligned_X = pca_X @ R.T
        
        # Create a copy of the original data frame with an aligned point cloud
        aligned_df = df.copy()
        aligned_df.loc[:, ["x", "y", "z"]] = aligned_X

        return aligned_df