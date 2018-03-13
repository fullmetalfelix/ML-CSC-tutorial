#! /usr/bin/python

import numpy as np
from struct import unpack #unpacks C type binaries

class Molecule:
    pass
    
def read_b(filename):
    '''
    Reads:
    Format follows:

    INT32     number of molecules

    --- for each molecule ---

        INT32     number of atoms in this molecule

        --- foreach atom ---

            INT32         atomic number

            DBL64*3    X,Y,Z coordinates

        --- end loop over atoms ---

    --- end loop over molecules ---
    returns:
    numpy object array, of size of number of molecules, each object being a class, similar to as saved in binary; reffered as 'flp format'
    '''
    with open(filename, 'rb') as f:
        nrec = unpack('i', f.read(4))[0]
        data = np.empty(nrec, dtype=object)
        
        for m in range(nrec):
            data[m] = Molecule()
            data[m].natm = unpack('i', f.read(4))[0]
            data[m].Zs = np.zeros(data[m].natm, dtype='int')
            data[m].coords = np.zeros((data[m].natm,3))
            data[m].qspd = np.zeros((data[m].natm,3))
            
            for i in range(data[m].natm):
                data[m].Zs[i] = unpack('i', f.read(4))[0]
                data[m].coords[i] = [unpack('d',f.read(8))[0], unpack('d',f.read(8))[0], unpack('d',f.read(8))[0]]
            for i in range(data[m].natm):
                data[m].qspd[i] = [unpack('d',f.read(8))[0], unpack('d',f.read(8))[0], unpack('d',f.read(8))[0]]
                
    return data





