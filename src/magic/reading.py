import numpy as np
import pandas as pd
from mendeleev.fetch import fetch_table
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, rdEHTTools
import multiprocessing as mp
from multiprocessing import active_children
from joblib.externals.loky import get_reusable_executor
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects


class SdfReader:
    """
    A class for reading point cloud data and chemical features from sdf files
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

    def reading(self,
                active_filename='actives_final.lowE.prepped.sdf',
                inactive_filename='decoys_final.lowE.prepped.sdf'):
        """
        A function that reads point cloud data and chemical features from sdf files for actives
        and inactives
        """
        target = self.target_name
        print(target)

        target_dict = {}
        target_dict[target] = {}
        
        # active 
        #target_dict[target]['active'] = self.reading_one_class(self.sdf_folder + active_filename)
        target_dict[target]['active'] = self.reading_one_class_parallel(self.sdf_folder + active_filename)

        # inactive 
        #target_dict[target]['inactive'] = self.reading_one_class(self.sdf_folder + inactive_filename)
        target_dict[target]['inactive'] = self.reading_one_class_parallel(self.sdf_folder + inactive_filename)

        return target_dict

    def reading_one_class(self, file_path):
        """
        A function that reads point cloud data and chemical features from one set of sdf files
        """
        target = self.target_name
        chem_table = self.chem_table

        feature_list = []
        con_file=Chem.ForwardSDMolSupplier(file_path, removeHs=False)

        for molecule in con_file:
            if molecule.GetProp('IX_CONF_ID')=='1':
                # Get atom cordinates
                XYZ = molecule.GetConformer(0).GetPositions()
                XYZ = np.array(XYZ)

                # Centre them using numpy
                #XYZ = XYZ - np.mean(XYZ, axis=0)

                # Get the atomic symbols
                symbols = [molecule.GetAtomWithIdx(idx).GetSymbol()
                                for idx in range(len(XYZ))]
                
                # Get the partial charges
                # Unkown method
                # partial_charge = np.array(molecule.GetProp('IX_Q').split('\n')).astype(float) 
                partial_charge = molecule.GetProp('IX_Q').split('\n')

                # Get the lipophilicities
                # Crippen alogp?
                # crippen_mol = Chem.rdMolDescriptors._CalcCrippenContribs(molecule)
                # alogp = [alogp for alogp, apol in crippen_mol]
                # lipophilicity = np.array(molecule.GetProp('IX_ALOGP').split('\n')).astype(float) 
                lipophilicity = molecule.GetProp('IX_ALOGP').split('\n')

                # Get the atomic contributions to the Crippen molar refractivity
                # similar to the polarisability
                # mh = Chem.AddHs(molecule) # for implicit H contribution
                crippen = Chem.rdMolDescriptors._CalcCrippenContribs(molecule)
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
                passed, result = Chem.rdEHTTools.RunMol(molecule)
                huckel = result.GetAtomicCharges()

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

                # for i in dir(Chem.rdMolDescriptors):
                #   print(i)

                # Property names
                #a = molecule.GetPropNames()
                #print([i for i in a])

                # SMILES
                # smiles = Chem.MolToSmiles(molecule)
                # print(smiles)

                # Create a dataframe with the previous quantities as columns
                df = pd.DataFrame({'symbol':pd.Series(symbols, dtype='str'),
                                    'x':pd.Series(XYZ[:,0], dtype='float'),
                                    'y':pd.Series(XYZ[:,1], dtype='float'),
                                    'z':pd.Series(XYZ[:,2], dtype='float'),
                                    'partial_charge':pd.Series(partial_charge, dtype='float'),
                                    'lipophilicity':pd.Series(lipophilicity, dtype='float'),
                                    'molar_refractivity':pd.Series(apols, dtype='float'),
                                    'asa':pd.Series(asa, dtype='float'),
                                    'tpsa':pd.Series(tpsa, dtype='float'),
                                    'gasteiger_q':pd.Series(gasteiger, dtype='float'),
                                    'eem_q':pd.Series(eem, dtype='float'),
                                    'yaehmop_q':pd.Series(huckel, dtype='float')# problematic?
                                    })
                
                # Centre coordinates using pandas
                df.loc[:, ["x", "y", "z"]] = (df[["x", "y", "z"]] - df.mean(axis=0, numeric_only=True )[0:3])

                # Merge the dataframe with the dataframe with information about the atomic
                # van der Waals radii and atomic numbers
                feature_list.append(pd.merge(df,
                                                chem_table,
                                                on='symbol',
                                                how='left'))

        return feature_list

    def reading_one_class_parallel(self, file_path):
        """
        A function that reads point cloud data and chemical features from one set of sdf files
        """
        # Number of available cores
        num_cores = mp.cpu_count()
        print('cores =', num_cores)

        feature_list = []
        
        #con_file = Chem.ForwardSDMolSupplier(file_path, removeHs=False)
        con_file = Chem.SDMolSupplier(file_path, removeHs=False)
        #con_file = Chem.MultithreadedSDMolSupplier(file_path, removeHs=False)
        N_mols = len(con_file)
        print('Number of molecules:', N_mols)
        
        #set_loky_pickler('pickle')

        #feature_list = Parallel(n_jobs=num_cores,
        #                        backend='loky')(
        #                            delayed(self._reading_one_sdf_file)(molecule)
        #                                    for molecule in con_file)

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
                              #molecule
                              #con_file,
                              file_path,
                              idx
                              ):
        """
        A function that reads point cloud data and chemical features from one sdf file
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
            #XYZ = XYZ - np.mean(XYZ, axis=0)

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
            huckel = list( np.zeros_like(eem) )

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

            # for i in dir(Chem.rdMolDescriptors):
            #   print(i)

            # Property names
            #a = molecule.GetPropNames()
            #print([i for i in a])

            # SMILES
            # smiles = Chem.MolToSmiles(molecule)
            # print(smiles)

            # Create a dataframe with the previous quantities as columns
            df = pd.DataFrame({'symbol':pd.Series(symbols, dtype='string'),
                                'x':pd.Series(XYZ[:,0], dtype='float'),
                                'y':pd.Series(XYZ[:,1], dtype='float'),
                                'z':pd.Series(XYZ[:,2], dtype='float'),
                                'partial_charge':pd.Series(partial_charge, dtype='float'),
                                'lipophilicity':pd.Series(lipophilicity, dtype='float'),
                                'molar_refractivity':pd.Series(apols, dtype='float'),
                                'asa':pd.Series(asa, dtype='float'),
                                'tpsa':pd.Series(tpsa, dtype='float'),
                                'gasteiger_q':pd.Series(gasteiger, dtype='float'),
                                'eem_q':pd.Series(eem, dtype='float'),
                                'yaehmop_q':pd.Series(huckel, dtype='float')#problematic
                                })
            
            # Centre coordinates using pandas
            df.loc[:, ["x", "y", "z"]] = (df[["x", "y", "z"]] - df.mean(axis=0, numeric_only=True )[0:3])

            # Merge the dataframe with the dataframe with information about the atomic
            # van der Waals radii and atomic numbers
            final_df = pd.merge(df,
                                chem_table,
                                on='symbol',
                                how='left')
        else:
            final_df = None
            # print('SDF not read')

        # print(final_df)
        return final_df
