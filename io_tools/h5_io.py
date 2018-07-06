import h5py
import numpy as np
import sys,os

import matplotlib.pyplot as plt


def h5_file_pointer(fname,what,path=''):
    f = h5py.File(fname,'r')
    location = path+'/'+what
    result = f[location]
    return result

def data_reader(fname, start, stop, what=None, path=''):
    f = h5py.File(fname,'r')
    results = {}
    for item in what:
        location = path+'/'+item
        tmp = f[location][start:stop,:,:]
        results[item]=tmp
    f.close()
    return results


if __name__ == "__main__":
    tst(sys.argv[1])
    

