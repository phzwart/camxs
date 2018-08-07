import numpy as np
from io_tools import basic_parser
import sys,os
import matplotlib.pyplot as plt

def slab_data(pnccd_np):
    pnccd_ij = np.zeros((1024,1024), dtype=pnccd_np.dtype)
    pnccd_ij[0:512, 0:512] = pnccd_np[0]
    pnccd_ij[512:1024, 0:512] = pnccd_np[1][::-1, ::-1]
    pnccd_ij[512:1024, 512:1024] = pnccd_np[2][::-1, ::-1]
    pnccd_ij[0:512, 512:1024] = pnccd_np[3]
    return pnccd_ij


def native_data(pnccd_ij):
    pnccd_np = np.zeros((4,512,512), dtype=pnccd_ij.dtype)
    pnccd_np[0][:, :] = pnccd_ij[:512, :512]
    pnccd_np[1][:, :] = pnccd_ij[512:1024, :512][::-1, ::-1]
    pnccd_np[2][:, :] = pnccd_ij[512:1024, 512:1024][::-1, ::-1]
    pnccd_np[3][:, :] = pnccd_ij[:512, 512:1024]
    return pnccd_np


asics = { '00': ( (  0, 512), (   0, 128) ) ,
          '01': ( (  0, 512), ( 128, 256) ) ,
          '02': ( (  0, 512), ( 256, 384) ) ,
          '03': ( (  0, 512), ( 384, 512) ) ,

          '10': ( (512,1024), (   0, 128) ) ,
          '11': ( (512,1024), ( 128, 256) ) ,
          '12': ( (512,1024), ( 256, 384) ) ,
          '13': ( (512,1024), ( 384, 512) ) ,

          '20': ( (0,512), ( 512, 640) ) ,
          '21': ( (0,512), ( 640, 768) ) ,
          '22': ( (0,512), ( 768, 896) ) ,
          '23': ( (0,512), ( 896,1024) ) ,  

          '30': ( (512,1024), ( 512, 640) ) ,
          '31': ( (512,1024), ( 640, 768) ) ,
          '32': ( (512,1024), ( 768, 896) ) ,
          '33': ( (512,1024), ( 896,1024) ) ,
        } 


borders = {  '00': {'01': ((0,512),(126,130))                            },
             '01': {'00': ((0,512),(126,130)), '02': ((0,512),(254,258)) },
             '02': {'01': ((0,512),(254,258)), '03': ((0,512),(382,386)) },
             '03': {'03': ((0,512),(382,386))                            },
           
             '20': {'21': ((512,1024),(638,642))                               },
             '21': {'20': ((512,1024),(638,642)), '22': ((512,1024),(766,770)) },
             '22': {'21': ((512,1024),(766,770)), '23': ((512,1024),(894,898)) },
             '23': {'23': ((512,1024),(894,898))                               },

             '10': {'11': ((512,1024),(126,130))                              },
             '11': {'10': ((512,1024),(126,130)), '12': ((  0,512),(254,258)) },
             '12': {'11': ((512,1024),(254,258)), '13': ((  0,512),(382,386)) },
             '13': {'13': ((512,1024),(382,386))                              },

             '30': {'31': ((0,512),(638,642))                            },
             '31': {'30': ((0,512),(638,642)), '32': ((0,512),(766,770)) },
             '32': {'31': ((0,512),(766,770)), '33': ((0,512),(894,898)) },
             '33': {'33': ((0,512),(894,898))                            }
          }

def build_asic_mask():
    asic_masks = {}
    for key in asics:
        tmp = np.zeros((1024,1024)) 
        sel = asics[key]
        sel_x = sel[0]
        sel_y = sel[1]
        tmp[ sel_x[0]:sel_x[1], sel_y[0]:sel_y[1] ] = 1.0
        asic_masks[ key ] = tmp
    return asic_masks



def equalize(median_img):
    n_bins = 1000
    image_histogram, image_bins = np.histogram( median_img.flatten() , bins=n_bins, normed=True  )
    for ii in range(1,n_bins):
        image_histogram[ii] += image_histogram[ii-1]
    bin_centers = image_bins[0:-1] + image_bins[1:]
    bin_centers = bin_centers / 2.0
    equalized_image = np.interp(median_img, bin_centers, image_histogram)
    return equalized_image





if __name__ == "__main__":
    img = np.load(sys.argv[1])
    plt.imshow(  equalize(img) , interpolation='none');plt.show()
    params = sys.argv[2]
    params = basic_parser.read_and_parse(open(params,'r'))
    params.show()
    mask = np.load( sys.argv[3] )
    img = fix_image(img, params.adjust)
    plt.imshow(  equalize(img)*mask , interpolation='none' );plt.show()    

