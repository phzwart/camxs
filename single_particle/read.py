import h5py
import sys,os
import numpy as np

from utils import panel_tools, plotters, fast_median_calculator, mask_tools, geometrized_panel, fix_asics
from io_tools import basic_parser, h5_io, saxs_tools

import matplotlib.pyplot as plt

from multiprocessing import Pool

default_parameters = """
[data]
filename            = None
data_path           = /photonConverter/pnccdBack/
data_field          = photonCount

[mask]
mask_def = mask.params

[geometry]
split            = [ ((0,260),(0,257)) ]
cxcydz           = [ (130, 128.5, 0.581) ]
pixel            = 300e-6
energy           = 1600.0

[setup]
n_cores          = 24 
"""

def write_img(img,name):
    f = open(name,'w')
    for tt in img.flatten():
        print >> f, tt
    f.close()

def run(config):
    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()

    ff = h5_io.h5_file_pointer( fname = params.data.filename,
                                what  = params.data.data_field,
                                path  = params.data.data_path ) 


    N,Nx,Ny = ff.shape
    print N,Nx,Ny
    imgs    = ff.value

    mean_img = np.mean( imgs, axis=0 )
    eq_mean  = panel_tools.equalize(mean_img)
    #plt.imshow(eq_mean);plt.show()


    mask = mask_tools.quick_mask( (Nx,Ny), params.mask.mask_def)
    #plt.imshow(eq_mean*mask);plt.show()

    #
    center     = (params.geometry.cxcydz[0][0], params.geometry.cxcydz[0][1])
    distance   = params.geometry.cxcydz[0][2]
    wavelength = 12398.0 / params.geometry.energy  
    det_panel  = geometrized_panel.detector_panel( (Nx,Ny), params.geometry.pixel, distance, wavelength, center, 2) 
    dxdy       = det_panel.mimimum_variance_saxs_center(mean_img,mask,power=0.33)
  
    saxs_curves = [] 

    q,mean_polar = det_panel.all_rings(mean_img,mask, dxdy[0],dxdy[1], 129)
    f = params.data.filename+'_q.dat'
    f = open(f,'w')
    for qq in q:
        print >> f, qq
    f.close()

    for ii in range(N):
        img = imgs[ii,:,:]
        s = det_panel.get_saxs(img,mask,False,dxdy[0],dxdy[1])
        Rg,LnI0 = saxs_tools.rg_estimate(s)
        if not np.isnan(Rg):
            print ii, Rg,LnI0
            saxs_curves.append( s.I  ) 
            q, ring = det_panel.all_rings(img,mask, dxdy[0],dxdy[1], 129)
            write_img( ring, params.data.filename+'_%i.dat'%ii )



if __name__ == "__main__":
    inputs = None
    if len(sys.argv)>1:
        inputs = sys.argv[1]
        if os.path.isfile(inputs):
            inputs = open(inputs,'r')
    run( inputs )
