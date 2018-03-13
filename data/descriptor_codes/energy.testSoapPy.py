from __future__ import print_function
import soapPy
from read_binary import *
import numpy as np
from describe.core import System
from numpy import genfromtxt

data = read_b('../binary/database-mulliken-ccsd-spd.bin')
elements_list = [1, 6, 7, 8, 9]
NradBas=5
Lmax=5
rCutHard=8.0
soap_nfeat = len(elements_list)*NradBas**2*(Lmax + 1)
soap_ndata = 10000
soap = np.zeros((soap_ndata, soap_nfeat))

## Mean
#for atom_ind, atoms in enumerate(data[:soap_ndata]):
#    if (atom_ind/float(soap_ndata-1)*10)%1 == 0 and soap_ndata != 1:
#        print('\r' + '({0:-<10})'.format('>'*int(atom_ind/float(soap_ndata-1)*10)) + ('\n' if (atom_ind+1)==soap_ndata else ''), end='')
#    Hpos = atoms.coords
#    atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
#    soap[atom_ind] = np.mean(soapPy.soap(atoms_sys, Hpos, rCutHard, NradBas, Lmax, elements_list), axis=0)
        
#np.save('../energy.input.soap.mean.npy', soap)

## Centre
#for atom_ind, atoms in enumerate(data[:soap_ndata]):
#    if (atom_ind/float(soap_ndata-1)*10)%1 == 0 and soap_ndata != 1:
#        print('\r' + '({0:-<10})'.format('>'*int(atom_ind/float(soap_ndata-1)*10)) + ('\n' if (atom_ind+1)==soap_ndata else ''), end='')
#    Hpos = np.mean(atoms.coords, axis = 0)
#    atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
#    soap[atom_ind] = soapPy.soap(atoms_sys, Hpos, rCutHard, NradBas, Lmax, elements_list)[0]
        
#np.save('../energy.input.soap.centre.npy', soap)

## sum
for atom_ind, atoms in enumerate(data[:soap_ndata]):
    if (atom_ind/float(soap_ndata-1)*10)%1 == 0 and soap_ndata != 1:
        print('\r' + '({0:-<10})'.format('>'*int(atom_ind/float(soap_ndata-1)*10)) + ('\n' if (atom_ind+1)==soap_ndata else ''), end='')
    Hpos = atoms.coords
    atoms_sys = System(positions=atoms.coords, numbers=atoms.Zs)
    soap[atom_ind] = np.sum(soapPy.soap(atoms_sys, Hpos, rCutHard, NradBas, Lmax, elements_list), axis=0)
        
np.save('../energy.input.soap.sum.npy', soap)
