from __future__ import print_function
from describe.descriptors import MBTR
from describe.core import System
import numpy as np
from scipy.sparse import lil_matrix, save_npz
from read_binary import *

data = read_b('../binary/database-mulliken-ccsd-spd.bin')

decay_factor = 0.5
mbtr = MBTR( 
    atomic_numbers=[1, 6, 7, 8, 9],
    k=[1, 2, 3],
    periodic=False,
    grid={
        "k1": {
            "min": 0,
            "max": 10,
            "sigma": 0.1,
            "n": 11,
        },
        "k2": {
            "min": 1/5,
            "max": 1.2,
            "sigma": 0.01,
            "n": 50,
        },
        "k3": {
            "min": -1.0,
            "max": 1.0,
            "sigma": 0.05,
            "n": 50,
        }
    },
    weighting={
        "k2": {
            "function": lambda x: np.exp(-decay_factor*x),
            "threshold": 1e-3
        },
        "k3": {
            "function": lambda x: np.exp(-decay_factor*x),
            "threshold": 1e-3
        },
    },
    flatten=True)


mbtr_nfeat = mbtr.get_number_of_features()
elements_list = [1, 6, 7, 8, 9]
max_data_count = 10000
desc = lil_matrix((max_data_count, mbtr_nfeat))

chg_count = np.zeros(len(elements_list), dtype='int')

for atom_ind, atoms in enumerate(data[:max_data_count]):
    if (atom_ind/float(max_data_count-1)*10)%1 == 0 and max_data_count != 1:
        print('\r' + '({0:-<10})'.format('>'*int(atom_ind/float(max_data_count-1)*10)) + ('\n' if (atom_ind+1)==max_data_count else ''), end='')
    atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
    desc[atom_ind] = mbtr.create(atoms_sys)
    
save_npz('../energy.input.mbtr.npz', desc.tocsr())
