import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from io_tools import h5_io
import sys


default_split = { 'front_A'    : ( slice(0,1024,1), slice(0  , 512,1), ),
                    'front_B'    : ( slice(0,1024,1), slice(512,1024,1), )
                }

# here are the beam center domains with indices after splitting
default_center_prior = { 'front_A' :  ( (550,570), (560 , 570)  ),
                         'front_B' :  ( (550,570), (-90 ,-70)   ) 
                       }

def massage(img, factor):
    result = np.power(img, factor )
    return result

def splitter(img, split_def):
    if type(split_def) == type({}):
        result = {}
        for key in split_def:
             result[key] = img[ split_def[key][0], split_def[key][1] ]
        return result
    else:
        result = img[ split_def[0], split_def[1] ]
        return result





def tst(filename,split_def):
    data = h5_io.data_reader( filename,stop=1010,start=1000,what=['data/cart_data'] )['data/cart_data']    
    data = np.sum( data, axis=0)
    plt.imshow(data);plt.show()
    res = splitter( data, split_def)
    for item in res.keys():
       plt.imshow(res[item]);plt.show()


if __name__ == "__main__":
    tst(sys.argv[1], default_split )

