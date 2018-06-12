import numpy as np
import matplotlib.pyplot as plt
import sys
from io_tools import saxs_tools


class distance_refinery(object):
    def __init__(self, reference_curve, pixel_size=75*1e-6, wavelength = 7.23):
        self.pixel_size      = pixel_size  # in meter)
        self.reference_curve = reference_curve
        self.wavelength      = wavelength # in Angstrom, like the q of the reference data

    def guess_distance(self,saxs,d_low,d_high,M=1e6):
        """
        We need to relate the distance in pixels to the q value

        given a saxs curve in pixel coordinates (p) , set

            r = p*k

        k is the pixel size

        using 
        
            theta = 0.5* atan(r / d )
  
        with d the unknown detector distance, and 

            q = 4 pi sin(theta)/lambda
 
        with lambda the wavelength


        we get

            q  =  sin( atan(0.5 p k /d) ) 4 pi / lambda

        Optimize via grid search.

        """
        r = saxs.q
        r = r*self.pixel_size
        scores = []
        d_range = np.linspace(d_low,d_high,M)
        for d in d_range:
            sin_theta = np.sin( 0.5*np.arctan(r / d))
            

            stol      = sin_theta / self.wavelength
            q_trial = stol*2.0*2.0*np.pi
            min_q = np.max( np.min(q_trial), np.min(self.reference_curve.q) )
            max_q = np.min( np.max(q_trial), np.max(self.reference_curve.q) )
            # we now have to interpolate the reference curve to the scale q curve
            # we have to maker sure that the q limits are observed
            sel_r   = (self.reference_curve.q >= min_q ) & (self.reference_curve.q <= max_q )
            sel_r_q = self.reference_curve.q[sel_r]
            sel_r_I = self.reference_curve.I[sel_r]

            sel_t   = (q_trial >= min_q ) & (q_trial <= max_q )
            sel_t_q = q_trial[sel_t]
            sel_t_I = saxs.I[sel_t]
            
            inter_I = np.interp(sel_t_q, sel_r_q, sel_r_I)
            score = np.corrcoef( inter_I, sel_t_I )
            score = score[0][1]
            scores.append(score)
        scores = np.array(scores)
        this_loc = np.argmax(scores)
        best_d = d_range[this_loc]
        return best_d


def tst(panel, reference,d_low=0.1, d_high=1.5,M=1e6):
    a   =  saxs_tools.read_saxs(panel)
    ref =  saxs_tools.read_saxs(reference)
    obj =  distance_refinery(ref)
    best_distance = obj.guess_distance(a,d_low,d_high,100000)
    print best_distance

if __name__ ==  "__main__":
    tst(sys.argv[1], sys.argv[2])    



