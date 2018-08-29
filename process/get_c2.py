import matplotlib
#matplotlib.use('Agg')

import h5py
import numpy as np
from io_tools import basic_parser, h5_io, saxs_tools
from utils import plotters, panel_tools, mask_tools, fix_asics, correlation, geometrized_panel
from scipy.signal import medfilt2d as mf2d
from scipy.spatial import ConvexHull
import sys, os
import matplotlib.pyplot as plt
from scipy.optimize import linprog


default_parameters = """
[data]
filename            = None
data_path           = data
data_field          = adu_front

[template]
filename = template.h5
scores   = /template/cc_scores
ints     = /template/tot_ints
offset   = /template/asic_offset
mask     = /template/mask
template = /template/median

[selection]
peak_fraction   = 0.75
selected_frames = selected_frames.npy

[geometry]
split            = [ ((0,1024),(0,512)), ((0,1024),(512,1024)) ]
cxcydz           = [ (562, 563, 0.29715), (567, -80, 0.29531)  ]
pixel            = 75e-6
energy           = 1700.0
Nq               = 500
Nphi             = 2048


[output]
filename = c2.npy

[refinement]
reference_curve  = None
power            = 0.33

"""


def in_area(mask, xa, ya, X, Y):
    dx = xa[1]-xa[0]
    dy = ya[1]-ya[0]
    minx = xa[0]
    miny = ya[0]
    Xi = np.floor( (X-minx)/dx ).astype(np.int)
    Yi = np.floor( (Y-miny)/dy ).astype(np.int)
    res = mask[Xi,Yi]
    sel = res > 0.5
    return sel

def run(config, interactive=False):
    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()

    
    # first get all stuff needed for correlation calculations
    tot_ints = h5_io.h5_file_pointer( fname = params.template.filename,
                                      what  = params.template.ints,
                                      path  = '').value

    # this is an array of indices of frames that we subject to selection. 
    # it will change while we remove outliers etc
    master_indices = np.arange(len(tot_ints))

    tot_scores=h5_io.h5_file_pointer( fname = params.template.filename,
                                      what  = params.template.scores,
                                      path  = '').value

    mask     = h5_io.h5_file_pointer( fname = params.template.filename,
                                      what  = params.template.mask,
                                      path  = '').value     

    template = h5_io.h5_file_pointer( fname = params.template.filename,
                                      what  = params.template.template,
                                      path  = '').value
 
    offset   = h5_io.h5_file_pointer( fname = params.template.filename,
                                      what  = params.template.offset,
                                      path  = '').value    
 
    data_f = h5_io.h5_file_pointer( fname = params.data.filename,
                                    what  = params.data.data_field,
                                    path  = params.data.data_path )
    N_imgs,Nx,Ny = data_f.shape


    # display the template
    equalized_template = panel_tools.equalize( template ) #+offset )

    #-------------------------------------------------
    # Here we do some selection stuff
    #
    print "---- Starting Selection Procedure ---- \n\n"  
    # first make a 2d histogram of the data
    errors = np.geterr()
    np.seterr(invalid='ignore') # suppress negative logs
    log_ints = np.log( tot_ints )
    np.seterr(invalid=errors['invalid'] ) # go back to previous settings
 
    sel = np.isnan( log_ints )
    log_ints = log_ints[~sel]
    these_scores = tot_scores[~sel]
    master_indices = master_indices[~sel] 

    sel = these_scores > 0.5  # if worse than 0.5 we have an issue (I guess)
    log_ints = log_ints[sel]
    these_scores = these_scores[sel]
    master_indices = master_indices[sel]

    hist = plt.hist2d( log_ints, these_scores, bins=[500,500] )
    plt.clf() # no need for the figure to ghang around
    dens = hist[0]

    # run a median filter to smoothen things a bit
    dens = mf2d(dens,3)
    log_int_axis = hist[1]
    cc_axis      = hist[2]
    log_int_axis = log_int_axis[0:-1]
    cc_axis      = cc_axis[0:-1]
    LI,CC = np.meshgrid( log_int_axis, cc_axis )
    ind = np.argmax( dens )

    # normalize the histogram such that the maximum value is 1
    trunc_dens = dens /np.max(dens)
    # select according to specifics
    sel = trunc_dens > params.selection.peak_fraction
    trunc_dens[~sel] = 0
    trunc_dens[sel]  = 1.0
    these_log_ints =  LI[sel].flatten()
    these_ccs = CC[sel].flatten()

    # calculate which images are within the decent area 
    selection_array = in_area( trunc_dens, log_int_axis, cc_axis, log_ints, these_scores  )
    leftover_master_indices = master_indices[selection_array]
    print '%i frames left after selection'%len(leftover_master_indices)    
    print 'Writing selection array as numpy array, with filename %s'%params.selection.selected_frames
    np.save( params.selection.selected_frames, leftover_master_indices ) 


    # now that we have selected frames, lets extract rings
    
    # build a c2_prep object
    # this guy does the polar transformation and static mask polar conversion etc

    c2_prep_obj = correlation.c2_prep( mask, params.geometry  )
    p_template = c2_prep_obj.to_polar( template )

    # now build a c2 accumulator
    c2_obj = correlation.correlation_accumulator( c2_prep_obj.geom_def.nq,
                                                  c2_prep_obj.geom_def.nphi,
                                                  c2_prep_obj.final_polar_mask )  
    
    # now we can compute C2's
    for this_image_index in leftover_master_indices:
        print this_image_index
        img = data_f[this_image_index,:,:]
        p_img = c2_prep_obj.to_polar(img)
        
        c2_obj.update_ac_only( p_img )    
    
    ac = c2_obj.finish_up_ac() 
    np.save( params.output.filename, ac )
    






if __name__ == "__main__":
    inputs = None
    interactive = False
    if len(sys.argv)>1:
        inputs = sys.argv[1]
        if os.path.isfile(inputs):
            inputs = open(inputs,'r')
        if len(sys.argv)>2:
            if sys.argv[2] =='pilot':
               interactive=True
    run( inputs, interactive )
