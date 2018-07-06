import h5py
from sklearn.mixture import GMM, DPGMM
import scipy.signal
import numpy as np
from io_tools import basic_parser, h5_io
from utils import fast_median_calculator
import sys, os

import matplotlib.pyplot as plt




default_parameters = """
[data]
filename         = ../PBCV_5.0e11.h5
data_field       = adu_front
data_path        = data
int_field        = front_sum_int
int_path         = summary
mask             = mask.npy
template_matches = pbcv_cc_scores.npy

[cc_selection]
cc_low              = 0.68
cc_high             = 1.2

[intensity_selection]
delta_i_ratio       = 0.25
intensity_selection = auto
minimum_peak_height = 0.05



[output]
filename = selected_PBCV
data_fields = ( adu_front, adu_back )
"""


def run(config):
    print config
    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()  

    # get the total intensity please
    tot_ints = h5_io.h5_file_pointer( fname = params.data.filename,
                                      what  = params.data.int_field,
                                      path  = params.data.int_path ).value
    histogram, bins = np.histogram( tot_ints, bins=50)
    peaks = scipy.signal.find_peaks_cwt( histogram, np.arange(3,6) )
    heights = histogram[ peaks ]
    norma = 1.0*np.sum(heights)
    heights = heights / norma
    sel =  heights > params.intensity_selection.minimum_peak_height
    peaks   = np.array(peaks)[sel]
    heights = heights[sel]
    this_peak = peaks[-1]
    this_intensity = bins[this_peak]
    if params.intensity_selection.intensity_selection =='auto':
        print "We will focus on images with intensity of %4.3e"%(this_intensity)
        print " +/- %4.3e"%(params.intensity_selection.delta_i_ratio*this_intensity)

    else:
       that_intensity = this_intensity*1.0
       this_intensity = float(params.intensity_selection.intensity_selection)
       print "The intensity bin selected by the user is %4.3e"%(this_intensity)
       print " +/- %4.3e"%(params.intensity_selection.delta_i_ratio*this_intensity)
       print "    The auto-selection would give %4.3e"%that_intensity
       print "    user supplied / auto selection = %4.3e"(this_intensity/that_intensity)


    delta_i = params.intensity_selection.delta_i_ratio*this_intensity
    int_sel = ( tot_ints  > this_intensity-delta_i ) & ( tot_ints < this_intensity+delta_i)


    # read in the template match scores
    template_scores = np.load( params.data.template_matches )
   
    cc_sel = (template_scores > params.cc_selection.cc_low ) & ( template_scores < params.cc_selection.cc_high )  

    combo_sel  = int_sel & cc_sel
    these_ccs  = template_scores[ combo_sel ]
    these_ints = tot_ints[ combo_sel ]
    indices = np.arange( 0, len(tot_ints) ) 
    indices = indices[combo_sel]

    print "Exporting %i images with a mean score of %4.3f"%(len(indices),np.mean(these_ccs))

    # make a new file please
    data_path   = params.data.data_path
    data_field  = params.data.data_field
    f_out = h5py.File(params.output.filename,'w')

    # we need provenance fields to be copied

    exp_id      =  h5_io.h5_file_pointer( fname = params.data.filename, what = 'exp_id', path = 'provenance' ).value
    time_points =  h5_io.h5_file_pointer( fname = params.data.filename, what = 'event_time', path = 'provenance' ).value
    fiducials   =  h5_io.h5_file_pointer( fname = params.data.filename, what = 'event_fiducials', path = 'provenance' ).value
    time_points =  time_points[combo_sel]
    fiducials   =  fiducials[combo_sel]
 
    prov        = f_out.create_group('provenance')
    dt = h5py.special_dtype(vlen=bytes)
    prov.create_dataset('exp_id'           , data = exp_id , dtype=dt)
    prov.create_dataset('event_time'       , data = time_points,  dtype='uint64')
    prov.create_dataset('event_fiducials'  , data = fiducials,    dtype='uint64')
   
    # make a field that will contain the data
    data_group  = f_out.create_group(data_path)
    
    export_data = data_group.create_dataset(params.data.data_field, (len(indices), 1024, 1024), dtype='float32') 


    # get a point to the data
    data_f = h5_io.h5_file_pointer( fname = params.data.filename,
                                    what  = params.data.data_field,
                                    path  = params.data.data_path )
    
    for jj,this_index in enumerate(indices):
        print jj, this_index
        export_data[jj,:,:] = data_f[this_index,:,:]

    # I want to export the total intensities as well, and the template scores
    data_group.create_dataset( 'mask', data =np.load( params.data.mask  ) , dtype='float32')
    summary = f_out.create_group('summary')
    summary.create_dataset('tot_int',  data = these_ints, dtype='float32')
    summary.create_dataset('template_scores', data = these_ccs, dtype='float32')


    f_out.close()

if __name__ == "__main__":
    inputs = None
    if len(sys.argv)>1:
        inputs = sys.argv[1]
        if os.path.isfile(inputs):
            inputs = open(inputs,'r')
    run( inputs )
