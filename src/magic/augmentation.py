import numpy as np
import pandas as pd

class DataAugmentation:
    """
    Augments the training data with rotated copies of the point clouds. Point
    clouds may be rotated randomly or according to the symmetries of regular
    convex polyhedra.
    
    Examples
    ========
        >>> a = DataAugmentation()
        >>> a.augment(list_df, rotation='octahedron')
        >>> a.augmented_list_df
    """

    def __init__(self):
        """Default constructor

        Parameters
        ----------
        """
        self.list_df = None
        self.augmented_list_df = None
        self.groups = None
        self.groups_1 = None
        self.groups_2 = None
        self.rot_mat_list = None

    def augment_both(self, list_df_1, list_df_2,
                     N_aug=[1, 1], random_seed=None):
        """
        Augment two lists of point cloud data frames with rotated copies.
        Rotations are random rotations.

        Parameters
        ===========
        list_df_1: A list
            A list of panda data frames of point clouds and VdW radii
        list_df_2: A list
            A list of panda data frames of point clouds and VdW radii
        N_aug: A list
            A 1 x 2 list [N_1, N_2] which are the number of rotated copies to
            augment each dataframe in each list of dataframes. The list must
            contain positive integers.
        random_seed: None or a float
            A random seed to initalise the random number generation
        Returns
        ========
        augmented_df: a pandas data frame
            A data frame augmented by the rotated copies of the point clouds
        """
        # Ensure that the augmentation multiplicities are integers
        N_1, N_2 = int(N_aug[0]), int(N_aug[1])

        # Create the random number generator
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
        else:
            rng = np.random

        # Create the augmented lists
        augmented_list_df_1 = self.augment_single(list_df_1,
                                                  N_aug=N_1, rng=rng)

        # rng = np.random.default_rng(seed=random_seed)    
        augmented_list_df_2 = self.augment_single(list_df_2,
                                                  N_aug=N_2, rng=rng)

        return augmented_list_df_1, augmented_list_df_2

    def augment_single(self, list_df,
                       N_aug=60, rng=None):
        """
        Augment a list of point cloud data frames with rotated copies.
        Rotations are random rotations.

        Parameters
        ===========
        list_df: A list
            A list of panda data frames of point clouds and VdW radii
        N_aug: An int
            A positive integer which is the number of rotated copies to
            augment each dataframe in the list of dataframes
        rng: None or a rng
            A random number generator to initalise the random number generation
        Returns
        ========
        augmented_df: a pandas data frame
            A data frame augmented by the rotated copies of the point clouds
        """

        N_aug = int(N_aug)
        N_df = len(list_df)
        augmented_list_df = []
        groups = []

        # Create the random number generator
        if rng is None:
            rng = np.random

        # Generate the N_df x N_rot x 3 x 3 4-tensor of rotations.
        # (A N_df x N_rot 2-tensor of 3 x 3 rotation 2-tensors)
        rot_mat_tensor = self._random_rotation_tensor(N_df, N_aug,
                                                      generator=rng)

        # Augment the list of dataframes with rotated copies of the 
        # point clouds
        augmented_list_df = [self._rotate_df(df, R)
                            for (idx, df) in enumerate(list_df)
                            for R in rot_mat_tensor[idx, :, :, :]]

        return augmented_list_df

    def augment(self, list_df, rotation='random', random_seed=None, N=24,
                add_rand=False, include_identity=True):
        """
        Augment the list of point cloud data frames with rotated copies.
        Rotations are random rotations or polyhedral rotations or a mixture.

        Parameters
        ===========
        list_df: A list
            A list of panda data frames of point clouds and VdW radii
        rotation: A string
            A
        random_seed: A float
            A
        N: An int
            The number of randomly rotated copies for each dataframe in the list
            of dataframes
        add_rand: A bool
            Whether or not to add random rotations in addition to polyhedral
            rotations
        include_identity: A bool
            Whether or not to include the identity in the rotations. If true,
            then an unrotated copy of each dataframe in the list of dataframes
            is included
        Returns
        ========
        augmented_df: a pandas data frame
            A data frame augmented by the rotated copies of the point clouds
        """

        N = int(N)
        N_df = len(list_df)
        self.list_df = list_df
        polyhedra = ['tetrahedron', 'octahedron', 'cube', 'icosahedron',
                     'dodecahedron']
        augmented_list_df = []
        groups = []

        if add_rand==False:
            if rotation=='random':
                # Create the random number generator
                rng = np.random.default_rng(seed=random_seed)
                for idx,df in enumerate(list_df):
                    rot_mat_list = self._random_rotation(N,
                                                         generator=rng,
                                                         include_identity=include_identity)
                    #print('Random seed:', seed)
                    #print('In loop list:', rot_mat_list)
                    for R in rot_mat_list:
                        rotated_df = df.copy()
                        X = rotated_df[["x", "y", "z"]].to_numpy()

                        rotated_X = X @ R.T
                        rotated_df.loc[:, ["x", "y", "z"]] = rotated_X

                        augmented_list_df.append(rotated_df)
                        groups.append(idx)

            elif rotation in polyhedra:
                rot_mat_list = self._polyhedron_rotation(polyhedron=rotation)

                for idx,df in enumerate(list_df):
                    for R in rot_mat_list:
                        rotated_df = df.copy()
                        X = rotated_df[["x", "y", "z"]].to_numpy()

                        rotated_X = X @ R.T
                        rotated_df.loc[:, ["x", "y", "z"]] = rotated_X

                        augmented_list_df.append(rotated_df)
                        groups.append(idx)

            else:
                raise ValueError("rotation must be 'random' or a regular convex "
                                "polyhedra: "
                                "'tetrahedron' or 'octahedron' or 'cube' or "
                                "'icosahedron' or 'dodecahedron'")
        elif add_rand==True:
            rng = np.random.default_rng(seed=random_seed)
            if rotation in polyhedra:
                rot_mat_list = self._polyhedron_rotation(polyhedron=rotation)
                self.rot_mat_list = rot_mat_list

                for idx,df in enumerate(list_df):
                    for R in rot_mat_list:
                        rotated_df = df.copy()
                        X = rotated_df[["x", "y", "z"]].to_numpy()

                        rotated_X = X @ R.T
                        rotated_df.loc[:, ["x", "y", "z"]] = rotated_X

                        augmented_list_df.append(rotated_df)
                        groups.append(idx)

                    rand_rot_mat_list = self._random_rotation(N,
                                                              generator=rng,
                                                              include_identity=False)

                    for R in rand_rot_mat_list:
                        rotated_df = df.copy()
                        X = rotated_df[["x", "y", "z"]].to_numpy()

                        rotated_X = X @ R.T
                        rotated_df.loc[:, ["x", "y", "z"]] = rotated_X

                        augmented_list_df.append(rotated_df)
                        groups.append(idx)

            else:
                raise ValueError("rotation must be a regular convex polyhedra: "
                                "'tetrahedron' or 'octahedron' or 'cube' or "
                                "'icosahedron' or 'dodecahedron'. It cannot be "
                                "'random' when 'add_rand' is 'True'")
        else:
            raise ValueError("'add_rand' must be 'True' or 'False'")

        self.augmented_list_df = augmented_list_df
        self.groups = groups

        return augmented_list_df

    def _polyhedron_rotation(self, polyhedron='tetrahedron'):
        """
        Compute N 3D rotation matrices corresponding to rotational symmetries of
        regular convex polyhedra.

        Parameters
        ===========
        polyhedron: a string
            The polyhedron
        Returns
        ========
        rot_mat_list: A list
            A list of 3x3 rotation matrices
        """

        rot_mat_list = []

        if polyhedron == 'tetrahedron':
            vertex = [np.array([1, 1, 1]), np.array([1, -1, -1]),
                      np.array([-1, 1, -1]), np.array([-1, -1, 1])]
            edge_midpoint = [np.array([1, 0, 0]), np.array([0, 1, 0]),
                             np.array([0, 0, 1])]

            rot_mat_list = [np.identity(3)]

            for v in vertex:
                rot_mat_list.append(self._rotation_axis_angle(v, 2*np.pi/3))
                rot_mat_list.append(self._rotation_axis_angle(v, -2*np.pi/3))

            for m in edge_midpoint:
                rot_mat_list.append(self._rotation_axis_angle(m, np.pi))

        elif polyhedron == 'octahedron' or polyhedron == 'cube':
            vertex = [np.array([1, 1, 1]), np.array([1, -1, -1]),
                      np.array([-1, 1, -1]), np.array([-1, -1, 1])]
            edge_midpoint = [np.array([1, 1, 0]), np.array([-1, 1, 0]),
                             np.array([1, 0, 1]), np.array([-1, 0, 1]),
                             np.array([0, 1, 1]), np.array([0, -1, 1])]
            face_centre = [np.array([1, 0, 0]), np.array([0, 1, 0]),
                           np.array([0, 0, 1])]

            rot_mat_list = [np.identity(3)]

            for v in vertex:
                rot_mat_list.append(self._rotation_axis_angle(v, 2*np.pi/3))
                rot_mat_list.append(self._rotation_axis_angle(v, -2*np.pi/3))

            for m in edge_midpoint:
                rot_mat_list.append(self._rotation_axis_angle(m, np.pi))

            for c in face_centre:
                rot_mat_list.append(self._rotation_axis_angle(c, np.pi/2))
                rot_mat_list.append(self._rotation_axis_angle(c, -np.pi/2))
                rot_mat_list.append(self._rotation_axis_angle(c, np.pi))

        elif polyhedron == 'icosahedron' or polyhedron == 'dodecahedron':
            rot_mat_list = [np.identity(3)]
        else:
            raise ValueError("polyhedron must be a regular convex polyhedra: "
                            "'tetrahedron' or 'octahedron' or 'cube' or "
                            "'icosahedron' or 'dodecahedron'")

        self.rot_mat_list = rot_mat_list

        return rot_mat_list

    def _random_rotation(self, N,
                         include_identity=True,
                         generator=None):
        """
        Compute N random 3D rotation matrices

        Parameters
        ===========
        N: an int
            The number of rotation matrices to be generated for each dataframe
        generator: generator object
            Initialised generator object
        Returns
        ========
        rot_mat_list: A list
            A list of 3x3 rotation matrices
        """

        N = int(N)
        rot_mat_list = [np.identity(3)]

        if include_identity == False:
            rot_mat_list = []

        rng = np.random

        if rng is not None:
            rng = generator

        # Generate uniformly random unit quaternions
        q_mat = rng.normal(loc=0.0, scale=1.0, size=(N, 4))
        q_mat = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)

        for q in q_mat:
            w, x, y, z = q[0], q[1], q[2], q[3]
            # Generate a uniform rotation matrix from a quaternion
            rot_mat = (np.identity(3) +
                      2 * np.array([[-y ** 2 - z ** 2, x * y - z * w, x * z + y * w],
                                    [x * y + z * w, -x ** 2 - z ** 2, y * z - x * w],
                                    [x * z - y * w, y * z + x * w, -x ** 2 - y ** 2]]))
            rot_mat_list.append(rot_mat)

        self.rot_mat_list = rot_mat_list

        return rot_mat_list

    def _rotate_df(self, df, R):
        """
        Rotate the co-ordinates of a daframe

        Parameters
        ===========
        df: A pandas dataframe
            A panda data frames of a point cloud and features
        R: A 3x3 array
            A rotation matrix
        Returns
        ========
        augmented_df: a pandas dataframe
            A dataframe with a rotated point cloud
        """
        # Copy the original dataframe
        rotated_df = df.copy()

        # Extract the point cloud of the dataframe
        X = rotated_df[["x", "y", "z"]].to_numpy()

        # Rotate the point cloud
        rotated_X = X @ R.T

        # Store the rotated point cloud
        rotated_df.loc[:, ["x", "y", "z"]] = rotated_X

        return rotated_df

    def _random_rotation_tensor(self, N_df, N_rot,
                                generator=None):
        """
        Compute N_df x N_rot random 3D rotation matrices

        Parameters
        ===========
        N_df: an int
            The number of dataframes
        N_rot: an int
            The number of rotation matrices to be generated for each dataframe
        generator: None or a rng
            If rng then a random generator is used
        Returns
        ========
        rot_mat_tensor: A tensor
            A N_df x N_rot x 3 x 3 4-tensor. N_df x N_rot 2-tensor of
            3 x 3 rotation matrices
        """

        # Set the random seed and random number generator
        rng = generator

        # Generate a tensor of uniformly random unit quaternions
        # A N_df x N_rot x 4 3-tensor
        q_tensor = rng.normal(loc=0.0, scale=1.0, size=(N_rot, N_df, 4))
        q_tensor = q_tensor / np.linalg.norm(q_tensor, axis=2, keepdims=True)

        rot_mat_tensor = self._rot_tensor(q_tensor)

        return rot_mat_tensor

    def _rot_tensor(self, q_tensor):
        """
        Compute a 3 x 3 rotation matrix from a matrix of unit quaternions.

        Parameters
        ===========
        q_tensor: an array
            A N x M x 4 3-tensor. Each row is a unit quaternion.
        Returns
        ========
        rot_tensor: an array
            A M x N x 3 x 3 4-tensor of rotation matrices.
        """
        # N x M 2-tensor
        w, x = q_tensor[:, :, 0], q_tensor[:, :, 1]
        y, z = q_tensor[:, :, 2], q_tensor[:, :, 3]

        # Identity 4-tensor ready for broadcasting
        # shape 3 x 3 x 1 x 1
        I = np.eye(3)
        I = I[..., np.newaxis, np.newaxis]

        # A 3 x 3 x N x M 4-tensor
        tensor = np.array(
                          [[-y ** 2 - z ** 2, x * y - z * w, x * z + y * w],
                          [x * y + z * w, -x ** 2 - z ** 2, y * z - x * w],
                          [x * z - y * w, y * z + x * w, -x ** 2 - y ** 2]]
                          )

        # Scale tensor and add the identity with broadcasting
        # A 3 x 3 x N x M 4-tensor
        rot_tensor = I + 2 * tensor

        # Swap the first 2 axes
        # A 3 x 3 x N x M 4-tensor
        rot_tensor = np.swapaxes(rot_tensor, 0, 1)

        # Reverse the order of axes
        # A M x N x 3 x 3 4-tensor
        rot_tensor = rot_tensor.T

        return rot_tensor

    def _rot_mat_single(self, quaternion):
        """
        Compute a 3 x 3 rotation matrix from unit quaternion.

        Parameters
        ===========
        quaternion: an array
            A 1 x 4 array that is a unit quaternion
        Returns
        ========
        rot_mat: an array
            A 3 x 3 array that is a rotation matrix.
        """

        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

        rot_mat = (np.identity(3) +
                   2 * np.array(
                       [[-y ** 2 - z ** 2, x * y - z * w, x * z + y * w],
                       [x * y + z * w, -x ** 2 - z ** 2, y * z - x * w],
                       [x * z - y * w, y * z + x * w, -x ** 2 - y ** 2]]
                               )
                  )

        return rot_mat

    def _rotation_axis_angle(self, axis, angle):
        """
        Compute the 3D rotation matrix given an axis vector and an angle.

        Parameters
        ===========
        axis: a np.array
            The rotation axis vector
        angle: a float
            The rotation angle in radians
        Returns
        ========
        rotation_mat: A 3x3 np.array
            A 3x3 rotation matrix
        """

        u = axis / np.linalg.norm(axis)

        axis_asym_mat = np.column_stack( (np.cross(u, np.array([1, 0, 0])),
                                          np.cross(u, np.array([0, 1, 0])),
                                          np.cross(u, np.array([0, 0, 1]))) )

        rotation_mat = (np.cos(angle) * np.identity(3) +
                        np.sin(angle) * axis_asym_mat +
                        (1 - np.cos(angle)) * np.outer(u, u) )

        return rotation_mat