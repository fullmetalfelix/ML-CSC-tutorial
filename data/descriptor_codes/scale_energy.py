import numpy as np
from read_binary import read_b

energy = np.load('../energy.output.npy')
data = read_b('../binary/database-mulliken-ccsd-spd.bin')
ndata = data.shape[0]
energy_correction = {1: -0.499058367914, 6: -37.7605599562, 7: -54.4968293735, 8: -74.9321577151, 9: -99.5549325226}

for atom_ind, atoms in enumerate(data):
    if (atom_ind/float(ndata-1)*10)%1 == 0 and ndata != 1:
        print('\r' + '({0:-<10})'.format('>'*int(atom_ind/float(ndata-1)*10)) + ('\n' if (atom_ind+1)==ndata else ''), end='')
    energy[atom_ind] -= sum([energy_correction[ii] for ii in atoms.Zs])
    
np.save('../energy.corrected.output.npy', energy)
