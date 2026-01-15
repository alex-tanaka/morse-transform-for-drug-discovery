"""
MAGIC is a library for creating Morse Augmented
Geometric Invariant Classification feature
vectors
"""

# Import version info
from .version_info import VERSION_INT, VERSION  

# Import main classes           
from .baseline import Baseline    
from .reading import SdfReader  
from .complex import MoleculeComplex  
# from .complex_rand import MoleculeComplexRand   
# from .complex_all_top import MoleculeComplexAllTop
# from .complex_noise import MoleculeComplexNoise   
from .augmentation import DataAugmentation 
from .alignment import DataAlignment         
from .saving import FeatureSave  
