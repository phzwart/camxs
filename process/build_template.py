import h5py
from sklearn.mixture import GMM, DPGMM
import scipy.signal
from multiprocessing import Pool
import numpy as np
from io_tools import basic_parser, h5_io
from utils import fast_median_calculator, plotters
import sys, os

import matplotlib.pyplot as plt




default_parameters = """
[data]
filename            = None
data_path           = data
data_field          = adu_front
int_path            = summary
int_field           = front_sum_int
delta_i_ratio       = 0.01
intensity_selection = auto
minimum_peak_height = 0.15


[output]
filename_base = template
 
"""


def image_score(a,b,mask=None):
    score = 0
    if mask is not None:
        sel = mask > 0.5
        aa = a[sel].flatten()
        bb = b[sel].flatten()
        score = np.corrcoef(aa,bb)[0][1] 
    else:
        score = np.corrcoef(a.flatten(),b.flatten())[0][1]
     
    return score

def run(config, interactive=False):
    print config
    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()  
    # get the total intensity please
    tot_ints = h5_io.h5_file_pointer( fname = params.data.filename,
                                      what  = params.data.int_field,
                                      path  = params.data.int_path ).value
    # make a histogram please
    histogram, bins = np.histogram( tot_ints, bins=50)
    #   
    peaks = scipy.signal.find_peaks_cwt( histogram, np.arange(3,6) )
    heights = histogram[ peaks ]
    norma = 1.0*np.sum(heights)
    heights = heights / norma
    sel =  heights > params.data.minimum_peak_height
    peaks   = np.array(peaks)[sel]
    heights = heights[sel]
    this_peak = peaks[-1]
    this_intensity = bins[this_peak]
    if params.data.intensity_selection =='auto':
        print "We will focus on images with intensity of %4.3e"%(this_intensity)
        print " +/- %4.3e"%(params.data.delta_i_ratio*this_intensity)

    else:
       that_intensity = this_intensity*1.0
       this_intensity = float(params.data.intensity_selection)
       print "The intensity bin selected by the user is %4.3e"%(this_intensity)
       print " +/- %4.3e"%(params.data.delta_i_ratio*this_intensity)
       print "    The auto-selection would give %4.3e"%that_intensity
       print "    user supplied / auto selection = %4.3e"%(this_intensity/that_intensity)


    delta_i = params.data.delta_i_ratio*this_intensity
    sel = ( tot_ints  > this_intensity-delta_i ) & ( tot_ints < this_intensity+delta_i)
    indices = np.arange(0,len(tot_ints))
    indices = indices[sel]
    M_indices = len( indices )
    print "In total, %i images have been selected for template construction"%M_indices 
    plotters.intensity_histogram( tot_ints, 50, 'Integrated Intensity','Occurance','%s'%params.data.filename, params.data.filename+'_int_hist.png', (this_intensity,delta_i), interactive )



    data_f = h5_io.h5_file_pointer( fname = params.data.filename,
                                    what  = params.data.data_field,
                                    path  = params.data.data_path )
    N_imgs,Nx,Ny = data_f.shape

    median_image = fast_median_calculator.Fast_Median_Image(100,0)
    var_image    = fast_median_calculator.ReservoirSampler( (Nx,Ny), min(500,len(indices)//2) )

    for nn in indices:
        print nn
        img = data_f[nn,:,:]
        median_image.update(img)
        var_image.update(img)
    
    median_img = median_image.current_median()
    sig_image = var_image.sigma()

    n_bins = 1000
    image_histogram, image_bins = np.histogram( median_img.flatten() , bins=n_bins, normed=True  )
    for ii in range(1,n_bins):
        image_histogram[ii] += image_histogram[ii-1]
    bin_centers = image_bins[0:-1] + image_bins[1:]
    bin_centers = bin_centers / 2.0
    equalized_image = np.interp(median_img, bin_centers, image_histogram)
    low_lim = np.percentile( median_img.flatten(), 10.0 )
    high_lim = np.percentile( median_img.flatten(), 85.0 )
   
    np.save(params.output.filename_base+'_equalized',equalized_image) 
    np.save(params.output.filename_base+'_median',median_img)
    np.save(params.output.filename_base+'_sigma',sig_image)

    plotters.plot_equalized_template(equalized_image, params.data.filename+'_eq_template.png', interactive)



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
