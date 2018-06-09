import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys

from io_tools import h5_io, basic_parser
from geometry import panel_tools, mask_tools, find_center

"""
prototype input file

data: filename=~/cluster-iterate/data/run_59_front.h5,  data_path=/data/cart_data, xstart=1000, xstop=1010;
use_mask: type=rectangle, xstart=400, xstop:700, ystart=400, y_stop=700;



"""

def run(inputs):
   inputs = basic_parser.read_and_parse(inputs)
   filename = inputs['data']['filename']
   Nstart   = inputs['data']['Nstart']
   Nstop    = inputs['data']['Nstop']
   data_path= inputs['data']['data_path']

   # define the added mask
   mask_params = inputs['mask']

   cart_data = h5_io.data_reader(filename, Nstart, Nstop, what=[data_path])[data_path]
   mean_img = np.abs(stats.stats.trim_mean(cart_data,0.2,axis=0))
   mean_img = panel_tools.massage(mean_img, inputs['massage']['factor'])


   # build use mask
   Nx,Ny = mean_img.shape
   masks = []
   for these_params in mask_params:
       m = mask_tools.build_it['rectangle'](Nx,Ny, these_params)
       masks.append(m)
   use_mask = masks[0]
   for m in masks[1:]:
       use_mask = use_mask*m
   auto_mask = mask_tools.auto_mask(mean_img)  
   total_mask = auto_mask*use_mask 

   panels = panel_tools.splitter( mean_img, panel_tools.default_split  )
   masks  = panel_tools.splitter( total_mask, panel_tools.default_split  )
   results = {}

   for key in panels.keys():
       this_mask  = masks[key]
       plt.figure(figsize=(15,15)); plt.imshow( np.log(panels[key]*this_mask) ) ; plt.show()
       # define the prior where we can find the beam center 
       xyrange = panel_tools.default_center_prior[key]
       x_prior = xyrange[0]
       y_prior = xyrange[1]
       tmp = find_center.slow_center_finder( panels[key], this_mask, x_prior, y_prior  ) 
       results[ key ] = tmp
   for key in results:
       r = results[key][2]
       I = results[key][3]
       plt.plot( r, np.log(I) )
       for rr,II in zip(r,I):
           print rr,II
   plt.show()
  


if __name__ == "__main__":
    run(sys.argv[1])


