from __future__ import print_function
from describe.descriptors import LMBTR
from describe.core import System
from describe.data.element_data import numbers_to_symbols
import numpy as np
from scipy.sparse import lil_matrix, save_npz
from read_binary import *

data = read_b('../binary/database-mulliken-ccsd-spd.bin')

decay_factor = 0.5
mbtr = LMBTR( 
    atom_index = 1,
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
            "min": 1/7,
            "max": 1.5,
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
chg = np.empty(len(elements_list), dtype='object')
max_chg_count = [10000]*4+[3314]
for i, j in enumerate(max_chg_count):
    chg[i] = lil_matrix((j, mbtr_nfeat))

chg_count = np.zeros(len(elements_list), dtype='int')

for atom_ind, atoms in enumerate(data):
    atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
    elements_req = np.array(elements_list)[chg_count != max_chg_count].tolist()
    print('\r {}'.format(chg_count), end = '')
    for element in elements_req:
        element_ind = elements_list.index(element)
        if chg_count[element_ind] != max_chg_count[element_ind] and element in atoms.Zs:
        
            element_indx_atoms = np.where(atoms.Zs == element)[0]
            
            len_added = min(element_indx_atoms.shape[0], max_chg_count[element_ind]-chg_count[element_ind])
            
            for i in range(chg_count[element_ind], chg_count[element_ind]+len_added):
                mbtr.atom_index = element_indx_atoms[i - chg_count[element_ind]]
                chg[element_ind][i] = mbtr.create(atoms_sys)
                
            chg_count[element_ind] += len_added
        
    if np.sum(chg_count) == sum(max_chg_count):
        break 
        
for i, j in enumerate(elements_list):
    save_npz('../charge.{}.input.mbtr'.format(j), chg[i].tocsr())
