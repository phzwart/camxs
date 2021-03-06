import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

from utils import plotters

def make_mask_rect(im_shape, mask_params):
    mask = np.zeros( im_shape ) + 1
    if type(mask_params) != type([]):
        mask_params = [mask_params]
    for m_params in mask_params:
        tmp_mask = np.zeros( im_shape ) + m_params.outside
        tmp_mask[ m_params.xstart:m_params.xstop,  m_params.ystart:m_params.ystop ] = m_params.inside
        mask = mask * tmp_mask
    return mask


class mask_def(object):
    def __init__(self, vals):
        self.xstart  = vals[0] 
        self.xstop   = vals[1]
        self.ystart  = vals[2]
        self.ystop   = vals[3]
        self.inside  = 0.0+vals[4]
        self.outside = 1.0-self.inside

def quick_mask(shape, params):
    f = open(params,'r')
    defs = []
    for line in f:
        keys = line.split()
        tmp = []
        for key in keys:
            tmp.append( int(key) )
        defs.append( mask_def(tmp) )
    f.close()
    mask = np.zeros(shape) + 1
    for md in defs:
        mask = mask*make_mask_rect(shape, md)
    return mask


def run(f,params):
    img = h5py.File(f,'r')['/template/median'].value
    defs =  []
    f = open(params,'r')
    for line in f:
        keys = line.split()
        tmp = []
        for key in keys:
            tmp.append( int(key) )
        defs.append( mask_def(tmp) )
    f.close()

    mask = img*0.0+1
    for md in defs:
        mask = mask*make_mask_rect(img.shape, md)
    np.save( 'mask', mask )
    fname= 'tmp_mask.png'
    plotters.plot_equalized_template( mask*img, fname, True  ) 



if __name__ == "__main__":
    run( sys.argv[1], sys.argv[2] ) 
