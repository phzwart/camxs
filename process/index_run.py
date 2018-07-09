import matplotlib
matplotlib.use('Agg')

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
filename   = None
data_path  = data
data_field = adu_front

[template]
median_file         = None
sigma_file          = None
display_range       = (0.0,4.0)
display_bins        = 40 


[output]
filename_base = None

"""


def equalize(median_img):
    n_bins = 1000
    image_histogram, image_bins = np.histogram( median_img.flatten() , bins=n_bins, normed=True  )
    for ii in range(1,n_bins):
        image_histogram[ii] += image_histogram[ii-1]
    bin_centers = image_bins[0:-1] + image_bins[1:]
    bin_centers = bin_centers / 2.0
    equalized_image = np.interp(median_img, bin_centers, image_histogram)    
    return equalized_image 

class template_classes(object):
    def __init__(self, low_score, high_score, n_bins):
        tmp = np.linspace( low_score, high_score, n_bins + 1)
        self.n_bins = n_bins
        self.low_score  = low_score
        self.high_score =  high_score
        self.lim_low  = tmp[ 0  : -1 ]
        self.lim_high = tmp[  1 :    ]
        self.NN = self.lim_high*0.0
        self.center   = ( self.lim_low + self.lim_high )/2.0 
         
        self.accumulators = []
        for ii in range(n_bins):
            tmp = fast_median_calculator.Fast_Median_Image()
            self.accumulators.append(tmp)

    def update(self, score, img):
        if score > self.low_score:
            if score  < self.high_score:
                a = self.lim_low[0]
                d = self.lim_low[1]-self.lim_low[0]
                index = int(np.floor((score-a)/d +0.5) )
                if index < self.n_bins:
                    self.NN[index] += 1
                    self.accumulators[index].update( img )

    def show_all(self, filename_base, display):
        for cc,obj,NN in zip( self.center, self.accumulators, self.NN) :
            tmp = obj.current_median()
            if tmp is not None:
                etmp = equalize(tmp)
                title = ' Z-score to template = %4.3f ; N = %3.2e\n'%(cc,NN)
                filename = filename_base + '_template_cc_%4.3f'%cc+'.png'
                plotters.plot_equalized_template(etmp, filename, display, title=title)


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

def z_score(img,mean,var,mask=None):
    m = np.sum( img*mask ) / np.sum(mask )
    k = np.sum( mean*mask ) / np.sum(mask )
    tmp = k*img/m
    sel = (mask > 0.5) & (var > 0)

    
    score = np.abs(tmp[sel] - img[sel])
    score = (score*score)/(1e-12+var[sel])
    score = np.sqrt(np.abs( score) )
    score = np.sum(score)/np.sum(mask[sel])
    return score

def run(config, interactive):
    print config, interactive

    params = basic_parser.read_and_parse( config, default_parameters  )
    params.show()  
    data_f = h5_io.h5_file_pointer( fname = params.data.filename,
                                    what  = params.data.data_field,
                                    path  = params.data.data_path )
    N_images = data_f.shape[0]
    
 
    # read in the template
    template_img = np.load( params.template.median_file )
    variance_img = np.load( params.template.sigma_file )
    variance_img = variance_img*variance_img

    mask_img     = np.load( params.template.mask )
    cc_scores = []  
    z_scores  = []
    cc_name = params.output.filename_base + '_cc_scores' 
    z_name = params.output.filename_base + '_z_scores'

    median_range_object = template_classes(    params.template.display_range[0], 
                                               params.template.display_range[1], 
                                           int(params.template.display_bins)     )
    

    for nn in range(N_images):
        img = data_f[nn,:,:]
        tmp_score = image_score( template_img, img , mask_img )
        tmp_z_score =  z_score(img,template_img,variance_img, mask_img)
        print nn, tmp_score , tmp_z_score
        median_range_object.update( tmp_z_score, img )
        cc_scores.append( tmp_score )
        z_scores.append(tmp_z_score)

    median_range_object.show_all( params.data.filename , interactive )

    np.save( cc_name , cc_scores )
    np.save( z_name, z_scores ) 
    filename = params.data.filename 
    plotters.plot_run_summaries(cc_scores, filename, interactive)

    



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
