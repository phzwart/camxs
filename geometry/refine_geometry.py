import numpy as np
import matplotlib.pyplot as plt
import sys

from io_tools import h5_io
from geometry import panel_tools, mask_tools

def run(filename,data_path,Nstart,Nstop):
   cart_data = h5_io.data_reader(filename, Nstart, Nstop, what=[data_path])[data_path]
   mean_img = np.sum(cart_data,axis=0)
   panels = panel_tools.splitter( mean_img, panel_tools.default_split  )
   masks  = {}
   for key in panels.keys():
       plt.imshow( np.log(panels[key])  ) ; plt.show()
       this_mask  = mask_tools.auto_mask(panels[key])
       masks[key] = this_mask 



if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2],int(sys.argv[3]), int(sys.argv[4]))


