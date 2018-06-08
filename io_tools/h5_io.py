import h5py
import numpy as np
import sys,os

import matplotlib.pyplot as plt

def data_reader(fname, start, stop, what=None, path=''):
    f = h5py.File(fname,'r')
    results = {}
    for item in what:
        location = path+'/'+item
        tmp = f[location][start:stop,:,:]
        results[item]=tmp
    return results

def read_packaged_data(fname,start,stop):
    data = data_reader(fname,start,stop,['cart_data','polar_data', 'polar_mask'], 'data' )
    return data

def tst(fname):
    data = read_packaged_data(fname,0,10)
    keys = ['cart_data','polar_data', 'polar_mask']
    for key in keys:
        assert data.has_key(key)  
    print('Ok')


if __name__ == "__main__":
    tst(sys.argv[1])
    

