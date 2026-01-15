# Data frames
import pandas as pd

# File paths
import os

# Numerical calculations
import numpy as np

# Parallel processing
import multiprocessing as mp
import joblib
from joblib import Parallel, delayed

# Progress bar
import contextlib
from tqdm import tqdm

# Itertools
import itertools

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Weighted Delaunay triangulation
from gudhi import AlphaComplex, SimplexTree


class MoleculeComplexRand:
    """
    Construct an alpha complex from the weighted Delaunay triangulation with
    weights given by the van der Waals radii of the atoms.

    Uses the top random vertices.

    Examples
    ========
        >>> m = MoleculeComplex()
        >>> simplices = m.build(df)
        >>> print(m.dim_complex)
        >>> r.plot_skeleton(n=1)
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """

        self.df = None
        self.critical_points_df = None

        self.simplex_tree = None
        self.simplices = None
        self.zero_simplices = None
        self.one_simplices = None
        self.two_simplices = None
        self.three_simplices = None

        self.feature_vector = None
        self.feature_list = None

        # The dimension of a simplicial complex is defined to be the dimension
        # of the highest-dimensional simplex that is in the simplicial complex
        self.dim_complex = None

    def build(self, df):
        """
        Compute the weighted delaunay triangulation of a Euclidean point set.
        Then keep the simplices whose edge lengths are less than the sum of van
        der Waals radii of the atoms which comprise the edge.

        Parameters
        ===========
        df: A pandas data frame
            A data frame of
        Returns
        ========
        simplices: list of tuples
            List of simplices.
        """

        self.df = df

        # Create an array of points from the atom positions and an array of
        # weights are the squares of the van der Waals radii of the atoms.
        points = self._getPoints(df)
        weights = self._getRadii(df) ** 2
        N = np.shape(points)[0]

        # Compute the thresholds and pairwise distance matrices.
        T = self._getThresh(df)
        D = self._getPDM(points)

        # Compute the weighted delaunay triangulation given the points and
        # weights using Gudhi.
        wdt = AlphaComplex(points=points, weights=weights)
        wdt_simplex_tree = wdt.create_simplex_tree()
        delaunay_simplices = wdt_simplex_tree.get_simplices()

        # Make a list of sets of sets for the simplices of different lengths
        # in the delaunay simplicial complex
        simplices_sets = [set(), set(), set(), set()] #4 * [set()]

        # Use a Rips-like criterion to build a subcomplex of the
        # weighted delaunay triangulation
        for simplex, filtration_value in delaunay_simplices:
            for L in range(1, 5):
                # find all combinations of length L in simplex
                for subset in itertools.combinations(simplex, L):
                    # make the combination into a set and add it to the set of
                    # sets if it is not already present (to avoid duplicate
                    # elements) and all edges are less than the sum of
                    # van der Waals radii
                    count_greater_than_thresh = 0
                    for edge in itertools.combinations(subset, 2):
                        if D[edge[0], edge[1]] > T[edge[0], edge[1]]:
                            count_greater_than_thresh += 1
                    if count_greater_than_thresh == 0:
                        # Sets in Python can only have frozensets as elements
                        # not sets themselves.
                        simplices_sets[L - 1].add(frozenset(subset))

        # Convert the list of sets of sets to a list of lists of lists
        simplices_list = [[], [], [], []]

        for idx, set_of_sets in enumerate(simplices_sets):
            simplices_list[idx] = [list(set_) for set_ in set_of_sets]

        self.simplices = simplices_list
        self.zero_simplices = sorted(simplices_list[0])
        self.one_simplices = simplices_list[1]
        self.two_simplices = simplices_list[2]
        self.three_simplices = simplices_list[3]

        # external_one_simplices = self._getExternalOneSimplices()
        # interior = [s for s in one_simplices if s not in external_one_simplices]

        # Create a Gudhi SimplexTree data structure for representing the
        # simplicial complex
        simplex_tree = SimplexTree()

        # Insert the simplices into the SimplexTree
        for simplices in simplices_list:
            for simplex in simplices:
                simplex_tree.insert(simplex)

        self.simplex_tree = simplex_tree

        # The dimension of the simplicial complex is the dimension of the
        # highest dimensional simplex in the complex
        self.dim_complex = simplex_tree.dimension()

        return simplices_list

    def plot_skeleton(self, n=2, renderer='colab'):
        '''
        Given a dataframe plot the all the points in the point cloud and the
        n-skeleton of the simplicial complex. n must be less than or equal to 2.

        Parameters
        ===========
        n: int <= 2
           An integer specifying the n-skeleton to be plotted.
        renderer: str
            The renderer to use for the plotly figure. Must be one of
            'plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
            'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
            'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
            'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
            'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png'. Default is
            'colab'.
        Returns
        ========
        :
          A plotly figure.
        '''
        df = self.df
        simplices = self.simplices

        if (n > 2) or (n < 1):
            raise TypeError('n must be 1 or 2.')

        # plot 0-simplices
        fig = px.scatter_3d(df,
                            x='x', y='y', z='z',
                            color='symbol',
                            opacity=0.5,
                            size='vdw_radius')

        one_simplices = self.one_simplices
        two_simplices = np.array(self.two_simplices)
        
        plot_points = [ [], [], [] ]
        for s in one_simplices:
            df_one_simplex = df.iloc[np.r_[ np.array(s) ], :]
            plot_points[0].extend(df_one_simplex["x"].to_list())
            plot_points[1].extend(df_one_simplex["y"].to_list())
            plot_points[2].extend(df_one_simplex["z"].to_list())
            plot_points[0].append(None) # to ensure no lines plotted
            plot_points[1].append(None) # between non-connected points
            plot_points[2].append(None)

        fig.add_trace(go.Scatter3d(x=plot_points[0],
                                    y=plot_points[1],
                                    z=plot_points[2],
                                    mode='lines',
                                    name='1-simplex',
                                    showlegend=False,
                                    opacity=0.3,
                                    line=dict(color='black', width=3)
                      )
        )

        if n == 2:
            fig.add_trace(go.Mesh3d(x=df["x"],
                                    y=df["y"],
                                    z=df["z"],
                                    i=two_simplices[:, 0],
                                    j=two_simplices[:, 1],
                                    k=two_simplices[:, 2],
                                    alphahull=0,# convex hull
                                    opacity=0.25,
                                    color='orange',
                                    name='2-simplex')
                              )

        fig.update_layout(width=600, height=600)
        fig.show(renderer=renderer)

    def getDirectionVectors(self, N=20, method='inverseTransform',
                            random_seed=None):
        '''
        Uniformly sample the unit 2-sphere uniformly N times to generate
        N unit vectors. The choice of methods is:

        'inverseTransform':
        Uses inverse transform sampling to randomly generate samples.

        'gaussian':
        Samples each ordinate randomly from a Gaussian and then
        normalises to one.

        'cubeRandom':
        Samples the cube [-1, 1]^3 uniformly randomly, discarding points with a
        norm greater than one. Then normalises the remaining samples.

        'regularPolyhedron':
        Generates directions corresponding to vertices on regular convex
        polyhedra. The desired polyhedron is specified by the number of
        vertices 'N':

        N = 4 is a tetrahedron,
        N = 6 is an octahedron,
        N = 8 is a cube,
        N = 12 is an icosahedron,
        N = 20 is a dodecahedron.

        Any other value of 'N' is not allowed.

        'squareAntiprism':
        Generates directions corresponding to vertices of a square antiprism.

        Parameters
        ===========
        N: An int
            An integer specifying the number of directions.
        method: 'inverseTransform' or 'gaussian' or 'cubeRandom'
                or 'regularPolyhedron'
            Specifies the method of generating the uniform samples.
        random_seed: None or a float
            If a float then a random_seed is used.
        Returns
        ========
        t_mat: An Nx3 array
            An Nx3 array of N direction vectors in 3 dimensions.
        '''

        t_mat = np.zeros( (N, 3) )

        # Set the random seed generator
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
        else:
            rng = np.random

        if method == 'inverseTransform':
            u = rng.uniform(low=-1.0, high=1.0, size=(N))
            theta = 2 * np.pi * rng.uniform(low=0.0, high=1.0, size=(N))
            t_mat[:, 0] = (1 - u ** 2) ** (0.5) * np.cos(theta)
            t_mat[:, 1] = (1 - u ** 2) ** (0.5) * np.sin(theta)
            t_mat[:, 2] = u
        elif method == 'gaussian':
            for i in range(N):
                r = np.array([0, 0, 0])
                while np.linalg.norm(r) < 10 ** (-5):
                    r = rng.normal(loc=0.0, scale=1.0, size=(1, 3))
                norm = np.linalg.norm(r)
                r = r / norm
                t_mat[i, :] = r
        elif method == 'cubeRandom':
            for i in range(N):
                r = np.array([10, 10, 10])
                while np.linalg.norm(r) > 1:
                    r = rng.uniform(low=-1.0, high=1.0, size=(1, 3))
                norm = np.linalg.norm(r)
                r = r / norm
                t_mat[i, :] = r
        elif method == 'regularPolyhedron':
            if N == 4:
                # Tetrahedron
                t_mat = np.array( [[1, 1, 1],
                                   [1, -1, -1],
                                   [-1, 1, -1],
                                   [-1, -1, 1]] )# / np.sqrt(3)
            elif N == 6:
                # Octahedron
                t_mat = np.array( [[1, 0, 0],
                                   [-1, 0, 0],
                                   [0, 1, 0],
                                   [0, -1, 0],
                                   [0, 0, 1],
                                   [0, 0, -1]] )
            elif N == 8:
                # Cube
                co_iter = itertools.product([1, -1], repeat=3)
                t_mat = np.array( [list(i) for i in co_iter] )# / np.sqrt(3)
            elif N == 12:
                # Icosahedron
                phi = (1 + 5 ** 0.5) / 2 #2
                co_iter = itertools.product([phi,-phi], [1,-1], [0,0])
                b = [list(i) for idx, i in enumerate(co_iter) if (idx % 2 == 0)]
                # All cyclic permutations
                a = [x[i:]+x[:i] for x in b for i in range(len(x))]
                t_mat = np.array(a) #/ np.sqrt(1 + phi **2)
            elif N == 20:
                # Dodecahedron
                phi = (1 + 5 ** 0.5) / 2
                co_iter = itertools.product([1/phi,-1/phi], [phi,-phi], [0,0])
                b = [list(i) for idx, i in enumerate(co_iter) if (idx % 2 == 0)]
                # All cyclic permutations
                a = [x[i:]+x[:i] for x in b for i in range(len(x))]

                co_cube = itertools.product([1, -1], repeat=3)
                co_list = [list(i) for i in co_cube]

                co_list.extend(a)

                t_mat = np.array(co_list)
            else:
                raise ValueError("When generating directions corresponding to"
                " vertices on regular convex polyhedra N can only be 4, 6, 8,"
                " 12, or 20.")
        elif method == 'squareAntiprism':
            t_mat = np.array( [[0, 2 ** (1/2), 2 ** (-1/4)],
                               [0, -2 ** (1/2), 2 ** (-1/4)],
                               [2 ** (1/2), 0, 2 ** (-1/4)],
                               [-2 ** (1/2), 0, 2 ** (-1/4)],
                               [1, 1, -2 ** (-1/4)],
                               [1, -1, -2 ** (-1/4)],
                               [-1, 1, -2 ** (-1/4)],
                               [-1, -1, -2 ** (-1/4)]] )

        return t_mat

    def getCriticalPoints(self,
                          direction,
                          pad_to_len=False,
                          top=False,
                          feature=False,
                          random_seed=None):
        '''
        Given an array of points calculate the critical points with respect to
        the t-axis (the first column). Select randomly the critical and regular
        points.

        Parameters
        ===========
        direction: A 1x3 array
            An array representing the direction vector.
        pad_to_len: False or an int
            If False then the feature vector is not zero padded. If an integer
            greater than the number of points in the simplicial complex is
            provided, then the number of points in the simplicial complex is
            zero padded.
        top: False or int
            If False then the feature vector contains the critical point
            analysis for every point. If an int is provided then the feature
            vector only contains the top 'top' critical points for a given
            direction, excluding non-critical points. If there are fewer than
            'top' critical points in a given direction then the feature vector
            is zero paddded. If both 'top' and 'pad_to_len' are specified then
            'top' has priority.
        feature: A boolean
            If True then just the feauture vector is outputted. If False then
            the projected heights, the labels of the critical points, and the
            feature vector are outputted.
        Returns
        ========
        critical_points: A list of lists and arrays
            A list of the form
            [array of t-ordinates,
            list of critical point labels,
            feature vector array]
            If 'feature' is True then just the feature vector array is outputted.
            If 'top' or 'pad_to_len' are specified then the dimensions of the
            feature vector are altered.

        '''
        # Get the new Cartesian coordinates of the projected points
        # where the new z-axis, t, is aligned with the direction vector
        X_tuv = self._getProjection(direction)
        
        # Set the random seed generator
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
        else:
            rng = np.random

        # Get the indices that sort the projected points randomly
        # (This is not efficient)
        X_sorted_indices = X_tuv[:, 0].argsort()[::-1] 
        rng.shuffle(X_sorted_indices) 

        # Preallocate the critical_points output list   
        critical_points = self._preallocateCriticalPointsOutput(X_tuv, 
                                                                pad_to_len,
                                                                indices=X_sorted_indices)
        
        for idx, vertex in enumerate(X_sorted_indices):
            # Calculate the upper link of the 0-simplex [vertex]
            # given the projected heights X_t = X_tuv[:, 0] of the 0-simplices
            upper_link_tree = self._getUpperLink([vertex], X_tuv[:, 0])

            # Calculate the reduced Betti numbers of the upper link
            # [rBetti_-1, rBetti_0, rBetti_1]
            reduced_betti_numbers = self._getIndex(upper_link_tree)

            # Insert the index into the features
            critical_points[2][idx, 1:4] = reduced_betti_numbers

            # Insert the index into the plotting labels
            # Keep the original 0-simplex ordering for consisitency
            # with self.df for later plotting
            critical_points[1][vertex] = '%d, %d, %d' %tuple(reduced_betti_numbers)

        # Keep only the top randomly selected points
        if ( (top is not False) and isinstance(top, (int, float)) ):
            N_top = int(top)

            # Create a randomly shuffled array of the points and their indices
            shuffled_points = [row for row in critical_points[2]]
            shuffled_points = np.array(shuffled_points)

            # Keep only the first 'N_top'. If there are fewer than 
            # 'N_top' points, then zero pad the matrix.
            # L = len(only_critical_points)
            L = len(shuffled_points)
            feature_matrix = np.zeros( (N_top, 14) )

            if L == 0:
                pass
            elif L >= N_top:
                feature_matrix = shuffled_points[:N_top]
            elif L < N_top:
                feature_matrix[0:L, :] = shuffled_points

            critical_points[2] = feature_matrix

        # Flatten the feature matrix into a vector
        critical_points[2] = critical_points[2].flatten(order='F')

        # Make the critical points data frame for later plotting
        df = self.df.copy()
        df.loc[:, 'symbol'] = critical_points[1]
        df.rename(columns={'symbol': 'critical point'}, inplace=True)
        self.critical_points_df = df

        if feature:
            critical_points = critical_points[2]

        return critical_points

    def plot_crit_pt(self, show_internal=False, two_skeleton=True, renderer='colab'):
        '''
        Plot the 1- (or 2-) skeleton of the complex with the points labelled according to
        the type of critical point they are. The critical points of the complex
        with respect to a particular direction must be calculated earlier.

        Parameters
        ===========
        show_internal: Boolean
            If true then colour the internal parts of the 1-skeleton blue.
        renderer: str
            The renderer to use for the plotly figure. Must be one of
            'plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
            'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
            'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
            'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
            'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png'. 
            Default is 'colab'.

        Returns
        ========
        :
          A plotly figure.
        '''

        df = self.critical_points_df
        one_simplices = self.one_simplices
        two_simplices = np.array(self.two_simplices)
        external_one_simplices = self._getExternalOneSimplices()
        interior = [s for s in one_simplices if s not in external_one_simplices]

        # plot 0-simplices
        fig = px.scatter_3d(df,
                            x='x', y='y', z='z',
                            color='critical point',
                            opacity=0.5,
                            size='vdw_radius')


        # Plot 2-simplices
        if two_skeleton:
            for tw in two_simplices:
                # Not efficient by adding the trace during each iteration
                # unlike below where it is all done at the end
                # but has the benefit of naming each simplex.
                # x, y, z are the co-ordinates of the 0-simplices
                # i, j, k specify the vertices for each 2-simplex
                fig.add_trace(go.Mesh3d(x=df["x"],
                                        y=df["y"],
                                        z=df["z"],
                                        i=[tw[0]],
                                        j=[tw[1]],
                                        k=[tw[2]],
                                        alphahull=0,
                                        opacity=0.25,
                                        color='orange',
                                        name=str(tw))
                              )

            # fig.add_trace(go.Mesh3d(x=df["x"],
            #                         y=df["y"],
            #                         z=df["z"],
            #                         i=two_simplices[:, 0],
            #                         j=two_simplices[:, 1],
            #                         k=two_simplices[:, 2],
            #                         alphahull=0,
            #                         opacity=0.3,
            #                         color='orange',
            #                         name='2-simplex')
            #                   )

        # plot 1-simplices
        plot_points = [ [], [], [] ]
        if show_internal == False:
            plot_simplices = one_simplices
        elif show_internal == True:
            plot_simplices = external_one_simplices
            plot_interior = [ [], [], [] ]
            for s in interior:
                df_one_simplex = df.iloc[np.r_[ np.array(s) ], :]
                plot_interior[0].extend(df_one_simplex["x"].to_list())
                plot_interior[1].extend(df_one_simplex["y"].to_list())
                plot_interior[2].extend(df_one_simplex["z"].to_list())
                plot_interior[0].append(None) # to ensure no lines plotted
                plot_interior[1].append(None) # between non-connected points
                plot_interior[2].append(None)
            fig.add_trace(go.Scatter3d(x=plot_interior[0],
                                        y=plot_interior[1],
                                        z=plot_interior[2],
                                        mode='lines',
                                        name='internal 1-simplex',
                                        showlegend=False,
                                        opacity=0.3,
                                        line=dict(color='blue', width=3)
                          )
            )

        for s in plot_simplices:
            df_one_simplex = df.iloc[np.r_[ np.array(s) ], :]
            plot_points[0].extend(df_one_simplex["x"].to_list())
            plot_points[1].extend(df_one_simplex["y"].to_list())
            plot_points[2].extend(df_one_simplex["z"].to_list())
            plot_points[0].append(None) # to ensure no lines plotted
            plot_points[1].append(None) # between non-connected points
            plot_points[2].append(None)

        fig.add_trace(go.Scatter3d(x=plot_points[0],
                                    y=plot_points[1],
                                    z=plot_points[2],
                                    mode='lines',
                                    name='1-simplex',
                                    showlegend=False,
                                    opacity=0.3,
                                    line=dict(color='black', width=3)
                      )
        )

        fig.update_layout(width=700, height=600)
        fig.show(renderer=renderer)

    def getFeature(self,
                   directions,
                   pad_to_len=False,
                   top=False,
                   parallel_calc=True,
                   backend='loky'):
        '''
        Given an array of directions, calculate the feature vector given by the
        critical points with respect to each direction.

        Parameters
        ===========
        direction: A Nx3 array
            An array of all the direction vectors.
        pad_to_len: False or an int
            If False then the feature vector is not zero padded. If an integer
            greater than the number of points in the simplicial complex is
            provided, then the number of points in the simplicial complex is
            zero padded.
        top: False or int
            If False then the feature vector contains the critical point
            analysis for every point. If an int is provided then the feature
            vector only contains the top 'top' critical points for a given
            direction, excluding non-critical points. If there are fewer than
            'top' critical points in a given direction then the feature vector
            is zero paddded. If both 'top' and 'pad_to_len' are specified then
            'top' has priority.
        parallel_calc: A boolean
            If True then the feature vector is calculated using a parallel for
            loops using multiprocessing from the joblib package
        backend: 'locky' or 'threading'
            If parallel_calc is True then 'locky' is for process-based
            multiprocessing and 'threading' is for thread-based multiprocessing.
        Returns
        ========
        feature_vector: An array
            the feature vector given by the critical points with respect to
            each direction
        '''
        feature_vector = []
        num_cores = mp.cpu_count()

        if parallel_calc == False:
            for v in directions:
                critical = self.getCriticalPoints(v,
                                                  pad_to_len=pad_to_len,
                                                  top=top,
                                                  feature=True)
                feature_vector.extend(critical)
        elif parallel_calc == True:
            feature_list = Parallel(n_jobs=num_cores,
                                      backend=backend)(
                                          delayed(
                                              self.getCriticalPoints)(
                                                  v,
                                                  pad_to_len=pad_to_len,
                                                  top=top,
                                                  feature=True)
                                              for v in directions)
            feature_vector = [item for sublist in feature_list
                              for item in sublist]

        feature_vector = np.array(feature_vector)

        self.feature_vector = feature_vector

        return feature_vector

    def build_getFeature(self,
                         df,
                         directions,
                         pad_to_len=False,
                         top=False,
                         parallel_calc=True,
                         backend='multiprocessing'):
        '''

        First compute the weighted delaunay triangulation of a Euclidean point
        set. Then keep the simplices whose edge lengths are less than the sum
        of van der Waals radii of the atoms which comprise the edge. Finally,
        given an array of directions, calculate the feature vector given by the
        critical points with respect to each direction.

        Parameters
        ===========
        df: A pandas data frame
            A data frame of
        direction: A Nx3 array
            An array of all the direction vectors.
        pad_to_len: False or an int
            If False then the feature vector is not zero padded. If an integer
            greater than the number of points in the simplicial complex is
            provided, then the number of points in the simplicial complex is
            zero padded.
        top: False or int
            If False then the feature vector contains the critical point
            analysis for every point. If an int is provided then the feature
            vector only contains the top 'top' critical points for a given
            direction, excluding non-critical points. If there are fewer than
            'top' critical points in a given direction then the feature vector
            is zero paddded. If both 'top' and 'pad_to_len' are specified then
            'top' has priority.
        parallel_calc: A boolean
            If True then the feature vector is calculated using a parallel for
            loops using multiprocessing from the joblib package
        backend: 'multiprocessing' or 'locky' or 'threading'
            If parallel_calc is True then 'multiprocessing' is the previous
            process-based backend based on multiprocessing.Pool, 'locky' is for
            process-based multiprocessing, and'threading' is for thread-based
            multiprocessing.
        Returns
        ========
        feature_vector: An array
            the feature vector given by the critical points with respect to
            each direction
        '''

        # Build the complex.
        self.build(df)

        feature_vector = []
        num_cores = mp.cpu_count()

        if parallel_calc == False:
            for v in directions:
                critical = self.getCriticalPoints(v,
                                                  pad_to_len=pad_to_len,
                                                  top=top,
                                                  feature=True)
                feature_vector.extend(critical)
        elif parallel_calc == True:
            feature_list = Parallel(n_jobs=num_cores,
                                      backend=backend)(
                                          delayed(
                                              self.getCriticalPoints)(
                                                  v,
                                                  pad_to_len=pad_to_len,
                                                  top=top,
                                                  feature=True)
                                              for v in directions)
            feature_vector = [item for sublist in feature_list
                              for item in sublist]

        feature_vector = np.array(feature_vector)

        self.feature_vector = feature_vector

        return feature_vector

    def build_getFeatures(self,
                          list_df,
                          directions,
                          pad_to_len=False,
                          top=False,
                          parallel_calc=True,
                          backend='multiprocessing',
                          progress_bar=False):
        '''

        For each data frame in the list, first compute the weighted delaunay
        triangulation of a Euclidean point set. Then keep the simplices whose
        edge lengths are less than the sum of van der Waals radii of the atoms
        which comprise the edge. Finally, given an array of directions,
        calculate the feature vector given by the critical points with respect
        to each direction.

        Parameters
        ===========
        list_df: A list of pandas data frames
            A list of data frames of
        direction: A Nx3 array
            An array of all the direction vectors.
        pad_to_len: False or an int
            If False then the feature vector is not zero padded. If an integer
            greater than the number of points in the simplicial complex is
            provided, then the number of points in the simplicial complex is
            zero padded.
        top: False or int
            If False then the feature vector contains the critical point
            analysis for every point. If an int is provided then the feature
            vector only contains the top 'top' critical points for a given
            direction, excluding non-critical points. If there are fewer than
            'top' critical points in a given direction then the feature vector
            is zero paddded. If both 'top' and 'pad_to_len' are specified then
            'top' has priority.
        parallel_calc: A boolean
            If True then the list of feature vectors is calculated using a
            parallel for loops using multiprocessing from the joblib package
        backend: 'multiprocessing' or 'locky' or 'threading'
            If parallel_calc is True then 'multiprocessing' is the previous
            process-based backend based on multiprocessing.Pool, 'locky' is for
            process-based multiprocessing, and'threading' is for thread-based
            multiprocessing.
        progress_bar: A boolean
            If True then a progress bar is shown during the calculation of
            feature vectors.
        Returns
        ========
        feature_list: A list of feature vectors
            A list of feature vectors for each data frame given by the critical
            points with respect to each direction
        '''

        feature_list = []
        num_cores = mp.cpu_count()

        if parallel_calc == False:
            if progress_bar == False:
                for idx, df in enumerate(list_df):
                    #print(idx) # Can uncomment to identify problematic molecules
                    feature_vector = self.build_getFeature(df,
                                                          directions,
                                                          pad_to_len=pad_to_len,
                                                          top=top,
                                                          parallel_calc=False)
                    feature_list.append(feature_vector)

            elif progress_bar == True:
                  for idx, df in enumerate(tqdm(list_df)):
                      #print(idx) # Can uncomment to identify problematic molecules
                      feature_vector = self.build_getFeature(df,
                                                            directions,
                                                            pad_to_len=pad_to_len,
                                                            top=top,
                                                            parallel_calc=False)
                      feature_list.append(feature_vector)
        elif parallel_calc == True:
            if progress_bar == False:
                feature_list = Parallel(n_jobs=num_cores,
                                        backend=backend)(
                                            delayed(
                                                self.build_getFeature)(
                                                    df,
                                                    directions,
                                                    pad_to_len=pad_to_len,
                                                    top=top,
                                                    parallel_calc=False)
                                                for df in list_df)

            elif progress_bar == True:
                L = len(list_df)
                tqdm_obj = tqdm(desc="Feature vector calculation", total=L)
                with self._tqdmJoblib(tqdm_obj) as progressBar:
                    feature_list = Parallel(n_jobs=num_cores,
                                            backend=backend)(
                                                delayed(
                                                    self.build_getFeature)(
                                                        df,
                                                        directions,
                                                        pad_to_len=pad_to_len,
                                                        top=top,
                                                        parallel_calc=False)
                                                    for df in list_df)

        self.feature_list = feature_list

        return feature_list

    def _getPoints(self, df):
        '''
        Given a dataframe return an array of 3-vectors, which are the positions
        of the atoms.

        Parameters
        ===========
        df: A pandas data frame
            A data frame of

        Returns
        ========
        X: An Nx3 array
            An Nx3 array of N Euclidean vectors in 3 dimensions.
        '''
        X = df[ ['x', 'y', 'z'] ].to_numpy()

        return X

    def _getRadii(self, df):
        '''
        Given a dataframe return an array, which are the van der Waals radii of
        the atoms

        Parameters
        ===========
        df: A pandas data frame
            A data frame of

        Returns
        ========
        r: A 1D array
          An 1D array of van der Waals radii.
        '''
        r = df[ 'vdw_radius' ].to_numpy()

        return r

    def _getChemistry(self):
        '''
        Given a dataframe return an array, which contains the chemical
        information [partial_charge, lipophilicity, vdw_radius, atomic_number]

        Parameters
        ===========
        df: A pandas data frame
            A data frame of
        Returns
        ========
        C: An Nx10 array
          An Nx10 array of chemical information.
        '''
        df = self.df

        C = df[ ['partial_charge', 'lipophilicity',
                 'molar_refractivity', 'asa', 'tpsa',
                 'gasteiger_q', 'eem_q', 'yaehmop_q',
                 'vdw_radius', 'atomic_number'] ].to_numpy()

        return C

    def _getPDM(self, X):
        """
        Given a set of Euclidean vectors, return a pairwise distance matrix.

        Parameters
        ===========
        X: An Nxd array
          An Nxd array of N Euclidean vectors in d dimensions
        Returns
        ========
        D: An NxN array
          An NxN array of all pairwise distances
        """
        XSqr = np.sum(X ** 2, 1)
        D = XSqr[:, None] + XSqr[None, :] - 2 * X.dot(X.T)
        D[D < 0] = 0  # Numerical precision
        np.fill_diagonal(D, 0)
        D = np.sqrt(D)

        return D

    def _getThresh(self, df):
        """
        Given a dataframe return a distance threshold matrix based upon the sum
        of Van der Waals radii.

        Parameters
        ===========
        X: An Nxd array of N Euclidean vectors in d dimensions
        Returns
        ========
        Thresh_mat: An NxN array of thresholds. 
        """
        N = df.shape[0]
        vdw_radii = df['vdw_radius'].to_numpy()
        one_vector = np.ones((N,1))
        
        # Convert vdw_radii into a matrix where
        # V_{ij} = (vdw_radii)_j 
        V = np.kron(vdw_radii, one_vector)

        # (Thresh_mat)_{ij} = (vdw_radi)_i + (vdw_radi)_j
        Thresh_mat = V + V.T

        return Thresh_mat

    def _getProjection(self, direction):
        '''
        Given a dataframe return an Nxd array of N Euclidean vectors in 3
        dimensions in a new orthonormal basis where one basis vector is
        parellel to the given direction vector.

        Parameters
        ===========
        df: A pandas data frame
            A data frame of
        direction: A 3x1 array
            An array representing the direction vector.
        Returns
        ========
        X_tuv: An Nx3 array
          An Nx3 array of N projected Euclidean vectors in 3 dimensions. The
          columns are the t, u, and v ordinates respectively.
        '''

        df = self.df
        X = self._getPoints(df)

        # Create an orthonormal basis t, u, v
        t = direction / np.linalg.norm(direction)

        if (t[0] != 0) or (t[1] != 0):
            idx = np.array( [0, 1, 2] )

            c = (t[idx[0]] ** 2 + t[idx[1]] ** 2) ** (-0.5)
            u = c * np.array( [-t[idx[1]], t[idx[0]], 0] )

            v = np.cross(t, u)
        elif (t[0] == 0) and (t[1] == 0):
            u = np.array( [1, 0, 0] )
            v = np.array( [0, 1, 0] )

        # Transform points into this basis
        transition_mat = np.array( [t, u, v] )
        transition_mat_inv = np.linalg.inv(transition_mat)

        X_tuv = X @ transition_mat_inv

        return X_tuv

    def _preallocateCriticalPointsOutput(self,
                                         X_tuv,
                                         pad_to_len,
                                         indices=False
                                         ):
        '''
        Given a 

        Parameters
        ===========
        X_tuv: An array
            An array of the Cartesian co-ordinates of the points in the frame
            with the z-axis aligned with the direction vector.
        pad_to_len: False or an int
            If False then the feature vector is not zero padded. If an integer
            greater than the number of points in the simplicial complex is
            provided, then the number of points in the simplicial complex is
            zero padded.
        indices: False or an array of integers
            If an array of integers, then it is used to sort the t-ordinates and
            feature vector array.
        Returns
        ========
        critical_points: A list of lists and arrays
            A list of the form
            [array of t-ordinates,
            list of critical point labels,
            feature vector array]
            If 'feature' is True then just the feature vector array is outputted.
            If 'top' or 'pad_to_len' are specified then the dimensions of the
            feature vector are altered.
        '''
        # Number of points
        N = np.shape(X_tuv)[0]

        # Get the chemical information of the projected points
        C = self._getChemistry()

        # Sort the co-ordinates and chemical features
        if indices is not False:
            X_tuv = X_tuv[indices]
            C = C[indices]

        # Make the output
        if (pad_to_len is False) or (pad_to_len <= N):
            critical_points = [ X_tuv[:, 0], # t-ordinates
                              N * ['Unknown'], # labels for plotting
                              np.zeros( (N, 14) ) # critical point features 1+3+10 = 14
                              ]
            critical_points[2][:, 0] = X_tuv[:, 0]

            # Add chemistry
            critical_points[2][:, 4:] = C

            # Label which points are padded or not
            #critical_points[2][0:N, 5] = np.ones(N)
        elif pad_to_len > N:
            N_pad = int(pad_to_len)
            critical_points = [ X_tuv[:, 0], # t-ordinates
                              N * ['Unknown'], # labels for plotting
                              np.zeros( (N_pad, 14) ) # critical point features
                              ]
            critical_points[2][0:N, 0] = X_tuv[:, 0] # padded t-ordinates

            # Add chemistry
            critical_points[2][0:N, 4:] = C

            # Label which points are padded or not
            #critical_points[2][0:N, 5] = np.ones(N)

        return critical_points

    def _getIndex(self, upper_link_tree):
        '''
        Given the upper link of a 0-simplex, return the index of the 0-simplex 
        equal to the reduced Betti numbers of the upper link

        Parameters
        ===========
        upper_link_tree: A GUDHI SimplexTree
            A SimplexTree of the upper link

        Returns
        ========
        index: An array
          A 1x3 array of reduced Betti numbers
        '''

        # Calculate the Betti numbers of the upper link
        # [Betti_{0}, Betti_{1}, ...]
        # Gudhi betti_numbers function requires compute_persistence() 
        # function to be launched first.
        # persistence_dim_max=True required to avoid betti_numbers() crashing
        # the kernel if the SimplexTree contains isolated 0-simplices
        upper_link_tree.compute_persistence(persistence_dim_max=True,  
                                            homology_coeff_field=2,# 11 default 
                                            min_persistence=-1)# 0 default
        betti_numbers = upper_link_tree.betti_numbers() 
        # betti_numbers = upper_link_tree.persistent_betti_numbers(from_value=0,
        #                                                           to_value=0)

        # Calculate the reduced Betti numbers of the upper link
        # for a simplicial 3-complex
        # [rBetti_{-1}, rBetti_{0}, rBetti_{1}]
        reduced_betti_numbers = np.zeros(3)

        # rBetti_{-1}
        if np.sum(betti_numbers) == 0:
            reduced_betti_numbers[0] = 1

        # rBetti_{0} = Betti_{0} - 1
        try:
            reduced_betti_numbers[1] = betti_numbers[0] - 1
        except:
            pass
        
        # rBetti_{1} = Betti_{1}
        try:
            reduced_betti_numbers[2] = betti_numbers[1]
        except:
            pass

        index = reduced_betti_numbers

        return index

    def _getUpperLink(self, zero_simplex, images):
        '''
        Given a 0-simplex return the upper link of the simplex in the simplicial
        complex calculated earlier given a height function

        Parameters
        ===========
        0-simplex: A list
            A list of 1 integer
        images: An array
            An array of the images of the vertices under the height function

        Returns
        ========
        upper_link_tree: A SimplexTree
          A SimplexTree of the upper link
        '''
        # The simplex tree of the whole simplicial complex
        complex_tree = self.simplex_tree

        # Calculate the star of the 0-simplex
        # (and its filtration values)
        star = complex_tree.get_star(zero_simplex)

        # Calculate a list of the boundaries of the simplices in the star
        d_star = [pair
                  for (simplex, _) in star
                  for pair in complex_tree.get_boundaries(simplex)]

        # Calculate the upper link
        # CHECK orderings
        f_o = images[zero_simplex[0]]
        upper_link = [pair
                      for pair in d_star
                      if ((pair not in star)
                            and
                            (self._getFV(pair[0], images) > f_o)
                          ) ]

        # Construct the upper link simplex tree
        upper_link_tree = SimplexTree()

        for simplex, filtration in upper_link:
            upper_link_tree.insert(simplex, filtration=0)

        # The following is not required if using persistence_dim_max=True
        # and it should be patched in a later GUDHI release.
        # Insert an N-simplex (and its subfaces) 
        # containing all N 0-simplices and 1 new 0-simplex 
        # (in case there is only 1 0-simplex in the upper link) 
        # with a higher filtration than the rest of the upper link.
        # This avoid the Kernel crashing when Betti numbers are calculated
        # for complexes containing isolated 0-simplices
        # n_simplex = [simplex[0] 
        #              for (simplex,_) in upper_link
        #              if len(simplex) == 1]
        # n_simplex.append(999999) # Cannot be too large to exceed maximum integer

        # If already present, assigns the minimal filtration value between input
        # filtration_value and the value already present
        # upper_link_tree.insert(n_simplex, filtration=1)

        # print('augmented upper link')
        # print([s for s in upper_link_tree.get_simplices()])

        return upper_link_tree

    def _getFV(self, simplex, images):
        '''
        Given the images of the vertices under a height function, return the
        filtration value of a simplex

        Parameters
        ===========
        1-simplex: A list
            A list of integers
        images: An array
            An array of the images of the vertices under the height function

        Returns
        ========
        filtration_value: A Float
          A float of the filtration value of the simplex
        '''
        simplex_array = np.array(simplex)
        vertex_heights = images[simplex_array]

        # The filtration value of a simplex is the minimum height of
        # any of its vertices for superlevel set filtration
        filtration_value = np.min(vertex_heights)

        return filtration_value

    def _getExternalOneSimplices(self):
        """
        Find the one simplices which are in the external 2-surface of the
        simplicial 3-complex.

        Parameters
        ===========
        Returns
        ========
        external_one_simplices: A list
            A list of the external one simplices.
        """
        one_simplices = self.one_simplices
        two_simplices = self.two_simplices
        three_simplices = self.three_simplices
        external_one_simplices = []
        external_two_simplices = []

        for tw in two_simplices:
            count_in_three_simplices = [set(tw).issubset(th) for th in three_simplices].count(True)
            if count_in_three_simplices <= 1:
                external_two_simplices.append(tw)

        for o in one_simplices:
            count_in_two_simplices = any(set(o).issubset(tw) for tw in two_simplices)
            #count_in_three_simplices = any(set(o).issubset(th) for th in three_simplices)
            count_in_external_two_simplices = any(set(o).issubset(etw) for etw in external_two_simplices)
            if count_in_external_two_simplices != 0:
                external_one_simplices.append(o)
            elif ( #(count_in_external_two_simplices == 0) and
                 (count_in_two_simplices == 0)
                 ):# CHECK
                external_one_simplices.append(o)

        external_one_simplices = [list(map(int, i)) for i in
                                  external_one_simplices]

        return external_one_simplices

    @contextlib.contextmanager
    def _tqdmJoblib(self, tqdm_object):
        """
        Context manager to patch joblib to report into tqdm progress bar
        given as argument.

        Parameters
        ===========
        tqdm_object: A tdqm object
            A tqdm object decorates an iterable object, returning an
            iterator which acts exactly like the original iterable, but
            prints a dynamically updating progressbar every time a value is
            requested.
        """
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()