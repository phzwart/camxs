import h5py
from sklearn.mixture import GMM, DPGMM
import scipy.signal
from multiprocessing import Pool
import numpy as np
from io_tools import basic_parser, h5_io
from utils import fast_median_calculator, plotters, panel_tools, mask_tools
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

[mask]
mask_def = mask.params

[output]
filename = template.h5

"""


def z_score(img,mean,sig,mask=None, return_img=True):
    m = np.sum( img*mask ) / np.sum(mask )
    k = np.sum( mean*mask ) / np.sum(mask )
    tmp = k*img/m
    sel = (mask > 0.5) & (sig > 0)


    score = np.abs(tmp[sel] - img[sel])
    score = (score*score)/(1e-12+sig[sel]*sig[sel])
    score = np.sqrt(np.abs( score) )
    score = np.sum(score)/np.sum(mask[sel])
    if return_img:
        return score, np.abs( (tmp - mean)/sig )
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
    plotters.intensity_histogram( tot_ints, 
                                  50, 
                                  'Integrated Intensity',
                                  'Occurance',
                                  '%s'%params.data.filename, 
                                  params.data.filename+'_int_hist.png', 
                                  (this_intensity,delta_i), 
                                  interactive )

    data_f = h5_io.h5_file_pointer( fname = params.data.filename,
                                    what  = params.data.data_field,
                                    path  = params.data.data_path )
    N_imgs,Nx,Ny = data_f.shape

    median_image = fast_median_calculator.Fast_Median_Image(100,0)
    var_image    = fast_median_calculator.ReservoirSampler( (Nx,Ny), min(500,len(indices)//2) )

    for nn in indices:
        print 'Processing image ', nn
        img = data_f[nn,:,:]
        median_image.update(img)
        var_image.update(img)
    
    median_img = median_image.current_median()
    sig_image = var_image.sigma()

    
    equalized_image = panel_tools.equalize( median_img )
    equalized_sigma = panel_tools.equalize( sig_image )
    
    #np.save(params.output.filename_base+'_equalized',equalized_image) 
    #np.save(params.output.filename_base+'_median',median_img)
    #np.save(params.output.filename_base+'_sigma',sig_image)

    plotters.plot_equalized_template(equalized_image, params.data.filename+'_eq_median.png', interactive)
    plotters.plot_equalized_template(equalized_sigma, params.data.filename+'_eq_sigma.png', interactive)

    # Let's make a new data filed in the exisiting h5 file where we store the template and its sigma
    template_h5 = h5py.File(params.output.filename,'w')
    grp = template_h5.create_group('template')
    grp.create_dataset('median', data=median_img, dtype=median_img.dtype )
    grp.create_dataset('sigma',  data=sig_image,  dtype=sig_image.dtype)
    int_range_sel = np.array([this_intensity-delta_i, this_intensity+delta_i])
    grp.create_dataset('intensity_selection_range', data=int_range_sel, dtype=int_range_sel.dtype)
    # we now need the mask
    mask = mask_tools.quick_mask( sig_image.shape, params.mask.mask_def)   
    grp.create_dataset('mask',  data=mask,  dtype=mask.dtype)
 
    # now we can do scoring of all images in the file
    z_scores = [ ]
    indices = np.arange(0,len(tot_ints)) 
    for ii in indices:
        img = data_f[ii]
        # now we need to score these images
        score,zimg = z_score(img,median_img,sig_image,mask, True)
        print ii, '--->', score
        z_scores.append( score )
        #plt.imshow( zimg, vmin=0, vmax=6, interpolation='none'); plt.colorbar(); plt.show()
    z_scores = np.array(z_scores)
    
    grp.create_dataset('z_scores',  data=z_scores,  dtype=z_scores.dtype)
    template_h5.close() 
    



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
