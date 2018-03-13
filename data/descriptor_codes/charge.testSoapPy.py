from __future__ import print_function
import soapPy
from read_binary import *
from describe.core import System
import numpy as np
from numpy import genfromtxt

data = read_b('../binary/database-mulliken-ccsd-spd.bin')
elements_list = [1, 6, 7, 8, 9]
NradBas=5
Lmax=5
rCutHard=8.0
soap_nfeat = len(elements_list)*NradBas**2*(Lmax + 1)
chg = np.empty(len(elements_list), dtype='object')
#chg[0] = np.zeros((1230122, soap_nfeat)) #H's in data
#chg[1] = np.zeros((846557, soap_nfeat)) #C's in data
#chg[2] = np.zeros((139764, soap_nfeat)) #N's in data
#chg[3] = np.zeros((187996, soap_nfeat)) #O's in data
#chg[4] = np.zeros((3314, soap_nfeat)) #F's in data

max_chg_count = [10000]*4+[3314]
for i, j in enumerate(max_chg_count):
    chg[i] = np.zeros((j, soap_nfeat))

chg_count = np.zeros(len(elements_list), dtype='int')

for atom_ind, atoms in enumerate(data):
    elements_req = np.array(elements_list)[chg_count != max_chg_count].tolist()
    print('\r {}'.format(chg_count), end = '')
    for element in elements_req:
        element_ind = elements_list.index(element)
        if chg_count[element_ind] != max_chg_count[element_ind] and element in atoms.Zs:
            element_indx_atoms = atoms.Zs == element
            Hpos = atoms.coords[element_indx_atoms][:max_chg_count[element_ind]-chg_count[element_ind]]
            len_added = Hpos.shape[0]
            atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
            x = soapPy.soap(atoms_sys, Hpos, rCutHard, NradBas, Lmax, elements_list) #rCutSoft = rCutHard - 3.0

            for i in range(chg_count[element_ind], chg_count[element_ind]+len_added):
                chg[element_ind][i] = x[i-chg_count[element_ind]]
                
            chg_count[element_ind] += len_added
        
    if np.sum(chg_count) == sum(max_chg_count):
        break 
        
for i, j in enumerate(elements_list):
    np.save('../charge.{}.input.soap.npy'.format(j), chg[i])
