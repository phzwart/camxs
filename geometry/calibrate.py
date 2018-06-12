import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import pickle

from io_tools import h5_io, basic_parser, saxs_tools
from geometry import panel_tools, mask_tools, find_center, refine_distance

"""
prototype input file

   mask           :
      ystop = 700                       ,
      inside = 1                        ,
      outside = 0                       ,
      xstop = 800                       ,
      xstart = 300                      ,
      ystart = 300                      ,
  ;

  mask           :
      ystop = 540                       ,
      inside = 0                        ,
      outside = 1                       ,
      ystart = 500                      ,
      xstart = 0                        ,
      xstop = 1024                      ,
  ;

  endstation     :
      wavelength = 7.289                ,
      high_d = 1.5                      ,
      pixel_size = 7.5e-05              ,
      low_d = 0.1                       ,
  ;

  data           :
      Nstart = 5000                     ,
      data_path = /data/cart_data       ,
      Nstop = 6000                      ,
      filename = /reg/neh/home/phzwart/cluster-iterate/data/run_59_front.h5,
  ;

  massage        :
      factor = 0.33                     ,
  ;

  reference      :
      filename = /reg/neh/home/phzwart/camxs_code/camxs/pbcv.dat,
  ;


"""

def run(inputs):
   inputs = basic_parser.read_and_parse(inputs)
   basic_parser.show_params( inputs )
   filename = inputs['data']['filename']
   Nstart   = inputs['data']['Nstart']
   Nstop    = inputs['data']['Nstop']
   data_path= inputs['data']['data_path']

   reference_data = saxs_tools.read_saxs(inputs['reference']['filename'])
   reference_saxs_refinery = refine_distance.distance_refinery( reference_data, 
                                                                wavelength=inputs['endstation']['wavelength'],
                                                                pixel_size=inputs['endstation']['pixel_size']  )

   # define the added mask
   mask_params = inputs['mask']

   cart_data   = h5_io.data_reader(filename, Nstart, Nstop, what=[data_path])[data_path]
   mean_img    = stats.stats.trim_mean(cart_data,0.2,axis=0)
   massage_img = panel_tools.massage( np.abs(mean_img)+1e-12, inputs['massage']['factor'])
   f = open('mean_img.pickle','w')
   pickle.dump(mean_img,f)
   f.close()

   # build use mask, combining masks as instructed
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
 
   # split the image and masks according to the parameters defined as defaults. Could be made specific if need be
   panels = panel_tools.splitter( mean_img, panel_tools.default_split  )
   massage_panels = panel_tools.splitter( massage_img, panel_tools.default_split  )
   masks  = panel_tools.splitter( total_mask, panel_tools.default_split  )
   results = {}

   # for each panel guess a beam center
   for key in panels.keys():
       this_mask  = masks[key]
       #plt.figure(figsize=(15,15)); plt.imshow( np.log(massage_panels[key]*this_mask) ) ; plt.show()
       # define the prior where we can find the beam center 
       xyrange = panel_tools.default_center_prior[key]
       x_prior = xyrange[0]
       y_prior = xyrange[1]
       cx,cy,saxs = find_center.slow_center_finder( massage_panels[key], this_mask, x_prior, y_prior  ) 
       results[ key ] = (cx,cy,saxs)

   panel_results = []

   for key in results.keys():
       this_panel = panels[ key ]
       this_mask  = masks[ key ]
       p_obj      = find_center.to_polar( this_panel, this_mask )
       cx         = results[key][0]
       cy         = results[key][1]
       saxs       = p_obj.get_saxs(cx,cy)
       best_distance = reference_saxs_refinery.guess_distance( saxs, 
                             inputs['endstation']['low_d'],
                             inputs['endstation']['high_d'], 1e5 )
       calibration_results         = {}
       calibration_results['name'] = key
       calibration_results[ 'cx' ] = cx
       calibration_results[ 'cy' ] = cy
       calibration_results[ 'distance' ] = best_distance
       calibration_results[ 'split'    ] = str( panel_tools.default_split[ key ]  ) 
       panel_results.append( calibration_results )

   results = {'panels': panel_results}
   print panel_results
   print results
   print
   basic_parser.show_params( results )   

if __name__ == "__main__":
    run(sys.argv[1])


