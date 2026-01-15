import numpy as np
import pandas as pd
from mendeleev.fetch import fetch_table
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, rdEHTTools, Descriptors
from sklearn.decomposition import PCA
from scipy.stats import skew
from collections import Counter
import multiprocessing as mp
from multiprocessing import active_children
from joblib.externals.loky import get_reusable_executor
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects

class Baseline:
    """
    A class for creating baseline quantile features from sdf files
    """

    def __init__(self, file_path_sdf, target_name):
        #file_path_sdf = '/data/math-morse-features/mert4443/data/sdf/dude/'
        self.sdf_folder = file_path_sdf + target_name + '/' 
        self.target_name = target_name
        self.make_chem_table()

    def make_chem_table(self):
        # Use mendeleev to create a data frame of elements and their chemical information
        ptable = fetch_table('elements') # a data frame of elements and properties
        cols = ['symbol', 'vdw_radius', 'atomic_number'] # specific columns we need
        chem_table = ptable[cols].copy() # Van der Waals radius in pm from the mendeleev table
        chem_table.loc[:, 'vdw_radius'] = chem_table.loc[:, 'vdw_radius'] / 100 # convert from pico metres to angstroms
        self.chem_table = chem_table

    def quantiles(self,
                  active_filename='actives_final.lowE.prepped.sdf',
                  inactive_filename='decoys_final.lowE.prepped.sdf'):
        """
        A function that reads point cloud data and chemical features from sdf files for actives
        and inactives and records property quantiles
        """
        target = self.target_name
        print(target)

        target_dict = {}
        target_dict[target] = {}
        
        # active 
        target_dict[target]['active'] = self.reading_one_class_parallel(self.sdf_folder + active_filename)

        # inactive 
        target_dict[target]['inactive'] = self.reading_one_class_parallel(self.sdf_folder + inactive_filename)

        return target_dict


    def reading_one_class_parallel(self, file_path):
        """
        A function that reads point cloud data and chemical features from one set of sdf files
        and records quantiles
        """
        # Number of available cores
        num_cores = mp.cpu_count()
        print('cores =', num_cores)

        feature_list = []
        
        con_file = Chem.SDMolSupplier(file_path, removeHs=False)
        N_mols = len(con_file)
        print('Number of molecules:', N_mols)

        feature_list = Parallel(n_jobs=num_cores,
                                backend='loky')(
                                    delayed(self._reading_one_sdf_file)(file_path, i)
                                            for i in range(N_mols))
        
        # Filter out features that are None
        feature_list = [feature for feature in feature_list if feature is not None]

        print('number of active processes:', len(active_children()))

        # Shutdown all active processes
        get_reusable_executor().shutdown(wait=True)
        print('number of active processes:', len(active_children()))

        return feature_list

    #@delayed
    #@wrap_non_picklable_object

    def _reading_one_sdf_file(self, 
                              file_path,
                              idx
                              ):
        """
        A function that reads point cloud data and chemical features from one sdf file
        and records quantiles
        """
        chem_table = self.chem_table
        con_file = Chem.SDMolSupplier(file_path, removeHs=False)
        molecule = con_file[idx]

        # names = molecule.GetPropNames()
        # print('properties:')
        # print( [(name, molecule.GetProp(name)) for name in names] )

        if molecule.GetProp('IX_CONF_ID')=='1':
            # Get atom cordinates
            XYZ = molecule.GetConformer(0).GetPositions()
            XYZ = np.array(XYZ)

            # Centre them using numpy
            XYZ = XYZ - np.mean(XYZ, axis=0)

            # Random quaternion
            q_mat = np.random.normal(loc=0.0, scale=1.0, size=(4))
            q_mat = q_mat / np.linalg.norm(q_mat, 
                                            #  axis=1, 
                                            keepdims=True)
            w, x, y, z = q_mat[0], q_mat[1], q_mat[2], q_mat[3]

            # Generate a uniform rotation matrix from a quaternion
            R = (np.identity(3) +
                    2 * np.array([[-y ** 2 - z ** 2, x * y - z * w, x * z + y * w],
                                  [x * y + z * w, -x ** 2 - z ** 2, y * z - x * w],
                                  [x * z - y * w, y * z + x * w, -x ** 2 - y ** 2]]))

            # Randomly rotated point cloud
            XYZ_r = XYZ @ R.T

            # PCA aligned point cloud
            # Rotate the point cloud axes to PCA axes.
            pca = PCA()
            pca_X = pca.fit_transform(XYZ)

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
            XYZ_a = pca_X @ R.T
    
            # Get the atomic symbols
            symbols = [molecule.GetAtomWithIdx(idx).GetSymbol()
                            for idx in range(len(XYZ))]
            
            # Get the partial charges
            # partial_charge = np.array(molecule.GetProp('IX_Q').split('\n')).astype(float) 
            partial_charge = molecule.GetProp('IX_Q').split('\n')

            # Get the lipophilicities
            # Crippen alogp?
            crippen = Chem.rdMolDescriptors._CalcCrippenContribs(molecule)
            lipophilicity = [alogp for alogp, apol in crippen]
            # lipophilicity = np.array(molecule.GetProp('IX_ALOGP').split('\n')).astype(float) 
            # lipophilicity = molecule.GetProp('IX_ALOGP').split('\n')

            # Get the atomic contributions to the Crippen molar refractivity
            # similar to the polarisability
            # mh = Chem.AddHs(molecule) # for implicit H contribution
            # crippen = Chem.rdMolDescriptors._CalcCrippenContribs(molecule)
            apols = [apol for alogp, apol in crippen]

            # Get the Gasteiger partial charges
            # Class I (non-QM)
            Chem.AllChem.ComputeGasteigerCharges(molecule)
            gasteiger = [molecule.GetAtomWithIdx(idx).GetDoubleProp('_GasteigerCharge')
                        for idx in range(len(XYZ))]

            # Get the EEM partial charges
            # Class I (non-QM)
            eem = Chem.AllChem.CalcEEMcharges(molecule)

            # Get YAeHMOP extended Huckel partial charges  
            # Require added hydrogens
            # Class II (MO)?
            # mh = Chem.AddHs(molecule)
            # Causes errors: suspicious distance 
            # passed, result = Chem.rdEHTTools.RunMol(molecule)
            # huckel = result.GetAtomicCharges()

            # Get Mulliken partial charges  
            # Class II?
            # mulliken = 

            # Get Lowdin partial charges  
            # Class II?
            # lowdin = 

            # Get RESP partial charges  
            # Class III?
            # resp = 

            # Get atomic contributions to Labute ASA
            # Aproximate surface area 
            # mh = Chem.AddHs(molecule) 
            asa, _ = Chem.rdMolDescriptors._CalcLabuteASAContribs(molecule)

            # Get atomic contributions to TPSA 
            # Topological Surface Area
            # mh = Chem.AddHs(molecule) 
            tpsa = Chem.rdMolDescriptors._CalcTPSAContribs(molecule)
      
            # Preallocate numpy array of atomic properties
            # There are 17 properties
            N = len(XYZ)
            property_array = np.zeros((N, 17))
            
            # Fill columns
            property_array[:, 0:3] = XYZ
            property_array[:, 3] = partial_charge
            property_array[:, 4] = lipophilicity
            property_array[:, 5] = apols
            property_array[:, 6] = asa
            property_array[:, 7] = tpsa
            property_array[:, 8] = gasteiger
            property_array[:, 9] = eem
            # property_array[:, 10] = huckel # problematic?
            property_array[:, 11:14] = XYZ_r
            property_array[:, 14:17] = XYZ_a

            # Fill quantile array with p-quantiles
            # Each row is a different p-quantile (?)
            p = [1, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0]
            quantile_2array = np.quantile(property_array, p, axis=0)

            # Flatten 2 array into a vector
            # 'C' denotes row major flattening
            quantile_array = quantile_2array.flatten(order='C') 
      
            # Preallocate numpy array of molecular properties
            # There are 28 molecular properties
            # There are 72 USR + USRCAT properties
            # There are 114 WHIM properties
            # Total of 72 + 28 + 210 + 114 = 424 properties
            # Total of 72 + 28 + 210 + 114 + 224 + 273 + 80 + 6 + 3 = 1010 properties
            # mol_property_array = np.zeros(28)
            mol_property_array = np.zeros(1010)

            # Molecular weight
            # 18th property (17 index)
            mol_property_array[0] = Chem.rdMolDescriptors.CalcExactMolWt(molecule)

            # Number of hydrogen bond acceptors
            mol_property_array[1] = Chem.rdMolDescriptors.CalcNumHBA(molecule)

            # Number of hydrogen bond donors
            mol_property_array[2] = Chem.rdMolDescriptors.CalcNumHBD(molecule)

            # Number of rotatable bonds
            mol_property_array[3] = Chem.rdMolDescriptors.CalcNumRotatableBonds(molecule)

            # Molecular lipophilicity log P
            # mol_property_array[4] = np.sum(lipophilicity)
            mol_property_array[4] = Chem.Descriptors.MolLogP(molecule)

            # Net formal charge
            mol_property_array[5] = Chem.GetFormalCharge(molecule)

            # Net Gasteiger charge
            mol_property_array[6] = np.sum(gasteiger)

            # Net EEM charge
            mol_property_array[7] = np.sum(eem)

            # Number of all atoms
            # mol_property_array[8] = len(XYZ)
            mol_property_array[8] = Chem.rdMolDescriptors.CalcNumAtoms(molecule)

            # Number of heavy atoms
            mol_property_array[9] = Chem.rdMolDescriptors.CalcNumHeavyAtoms(molecule)

            # Count occurences of atoms
            n_atoms_dict = Counter(symbols)

            # Number of boron
            mol_property_array[10] = n_atoms_dict.get('B') or 0

            # Number of bromine
            mol_property_array[11] = n_atoms_dict.get('Br') or 0

            # Number of carbon
            mol_property_array[12] = n_atoms_dict.get('C') or 0

            # Number of chlorine
            mol_property_array[13] = n_atoms_dict.get('Cl') or 0

            # Number of fluorine
            mol_property_array[14] = n_atoms_dict.get('F') or 0

            # Number of iodine
            mol_property_array[15] = n_atoms_dict.get('I') or 0

            # Number of nitrogen
            mol_property_array[16] = n_atoms_dict.get('N') or 0

            # Number of oxygen
            mol_property_array[17] = n_atoms_dict.get('O') or 0

            # Number of phosphorous
            mol_property_array[18] = n_atoms_dict.get('P') or 0

            # Number of sulfur
            mol_property_array[19] = n_atoms_dict.get('S') or 0

            # Number of chiral centres
            chiral_centres = Chem.FindMolChiralCenters(molecule, includeCIP=False)
            mol_property_array[20] = len(chiral_centres)

            # Number of ring systems
            mol_property_array[21] = Chem.rdMolDescriptors.CalcNumRings(molecule)

            # Double Cubic Lattice (Volume) method
            # Requires rdkit 2024, morse_tune_lgb_5 environment
            dclm = Chem.rdMolDescriptors.DoubleCubicLatticeVolume(molecule)

            # Surface Area
            mol_property_array[22] = dclm.GetSurfaceArea()

            # vdW volume
            mol_property_array[23] = dclm.GetVDWVolume()

            # Volume
            mol_property_array[24] = dclm.GetVolume()

            # Compactness
            mol_property_array[25] = dclm.GetCompactness()

            # TPSA (2D descriptor)
            mol_property_array[26] = Chem.rdMolDescriptors.CalcTPSA(molecule)

            # PBF descriptor
            mol_property_array[27] = Chem.rdMolDescriptors.CalcPBF(molecule)

            # USR
            mol_property_array[28:28+12] = Chem.rdMolDescriptors.GetUSR(molecule)

            # USRCAT
            mol_property_array[40:40+60] = Chem.rdMolDescriptors.GetUSRCAT(molecule)

            # WHIM descriptors
            mol_property_array[310:310+114] = Chem.rdMolDescriptors.CalcWHIM(molecule)

            # PCA principal component values
            mol_property_array[1001:1001+3] = pca.explained_variance_

            # PCA principal components value ratios
            mol_property_array[1004:1004+3] = pca.explained_variance_ratio_

            # Principle component axis skews
            mol_property_array[1007:1007+3] = paxis_skews

            # Concatenate scalar features
            feature_array = np.concatenate( (quantile_array, mol_property_array) )
    
        else:
            feature_array = None
            # print('SDF not read')

        # print(final_df)
        return feature_array
