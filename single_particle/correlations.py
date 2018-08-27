import h5py
import sys,os
import numpy as np

from utils import panel_tools, plotters, mask_tools, geometrized_panel, correlation
from io_tools import basic_parser, h5_io, saxs_tools

import matplotlib.pyplot as plt

from multiprocessing import Pool

default_parameters = """
[data]
filename            = None
data_path           = /photonConverter/pnccdBack/
data_field          = photonCount
output_base         = result

[mask]
mask_def = mask.params

[geometry]
split            = [ ((0,260),(0,257)) ]
cxcydz           = [ (130, 128.5, 0.581) ]
pixel            = 300e-6
energy           = 1600.0
n_phi            = 128
n_q              = 128

[selection]
filename = None 

"""

def write_img(img,name):
    f = open(name,'w')
    for tt in img.flatten():
        print >> f, tt
    f.close()



def run(config):
    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()

    f = open(params.selection.filename,'r')
    index = []
    Rgs = []
    I0s = []
    for line in f:
        keys = line[:-1].split()
        ii = int(keys[0])
        Rg = float(keys[1])
        LnI0 = float(keys[2])
        Rgs.append(Rg)
        I0s.append(LnI0)
        index.append(ii)
    f.close()
    index = np.array(index).astype(np.int)
    scales = np.array(I0s)
    scales = np.exp(scales)
    scales = scales / np.max(scales)

    ff = h5_io.h5_file_pointer( fname = params.data.filename,
                                what  = params.data.data_field,
                                path  = params.data.data_path ) 
    imgs    = ff[index,:,:]
    N,Nx,Ny = imgs.shape
    
    mean_img = np.mean( imgs, axis=0 )
    eq_mean  = panel_tools.equalize(mean_img)
    #plt.imshow(eq_mean);plt.show()


    mask = mask_tools.quick_mask( (Nx,Ny), params.mask.mask_def)
    #plt.imshow(eq_mean*mask);plt.show()

    #
    center     = (params.geometry.cxcydz[0][0], params.geometry.cxcydz[0][1])
    distance   = params.geometry.cxcydz[0][2]
    wavelength = 12398.0 / params.geometry.energy  
    det_panel  = geometrized_panel.detector_panel( (Nx,Ny), params.geometry.pixel, 
                                                   distance, wavelength, center, 0, params.geometry.n_q )
     
    dxdy       = det_panel.mimimum_variance_saxs_center(mean_img,mask,power=0.33)
  
    saxs_curves = [] 

    q,mean_polar = det_panel.all_rings(mean_img,mask, dxdy[0],dxdy[1], params.geometry.n_phi )
    
    # get the mask
    q,mask_polar = det_panel.all_rings(mask, mask, dxdy[0],dxdy[1], params.geometry.n_phi )
    sel = mask_polar < 0.9
    mask_polar[sel]=0



    f = params.data.output_base+'_q.dat'
    f = open(f,'w')
    for qq in q:
        print >> f, qq
    f.close()

    
    c2_obj = correlation.correlation_accumulator( params.geometry.n_q, params.geometry.n_phi, mask_polar )

    for ii in range(N):
        print ii, scales[ii]
        img = imgs[ii,:,:]
        s = det_panel.get_saxs(img,mask,False,dxdy[0],dxdy[1])
        saxs_curves.append( s.I  ) 
        q, ring = det_panel.all_rings(img,mask, dxdy[0],dxdy[1], params.geometry.n_phi)
        scale = 1 #scales[ii]
        write_img( ring, params.data.output_base+'_img_%i.dat'%index[ii] )
        c2_obj.update( ring*scale )
    c2_final = c2_obj.finish_up()
    np.save( params.data.output_base+'_c2.npy', c2_final ) 
    

if __name__ == "__main__":
    inputs = None
    if len(sys.argv)>1:
        inputs = sys.argv[1]
        if os.path.isfile(inputs):
            inputs = open(inputs,'r')
    run( inputs )
