import numpy as np
import h5py 
import matplotlib.pyplot as plt
import sys,os
from io_tools import basic_parser, h5_io, saxs_tools
from utils import fast_median_calculator, plotters, panel_tools, mask_tools, geometrized_panel, cointegration
from utils.golden_section_search import gss 
from utils import piecewise_fit

class template_object(object):
    def __init__(self,mask,median,sigma,ref_curve, q_min=0.025, q_max=0.05):
        
        self.mask   = mask
        self.median = median
        self.sigma  = sigma
        self.q_max  = q_max
        self.q_min  = q_min 

        self.ref_curve = ref_curve


    def split_asics(self, geometry):
        full_asic_masks = panel_tools.build_asic_mask()
        split_asic_masks = []
        for split_def in geometry.split:        
            tmp = {}
            for asic in full_asic_masks.keys():
                asic_mask = full_asic_masks[ asic ]
                sel_x = slice( split_def[0][0],split_def[0][1], 1 )
                sel_y = slice( split_def[1][0],split_def[1][1], 1 )
                sam   = asic_mask[ sel_x, sel_y ]
                if np.max(sam) > 0.5:
                    tmp[ asic ] = sam
            split_asic_masks.append( tmp )
        return split_asic_masks

    def variance_score(self, x, img, asic_mask, mask, geom_panel):
        tmp_img = asic_mask*mask*x + img
        curve = geom_panel.get_saxs(tmp_img, mask, 0, 0, no_corrections=True)
        ws = curve.s*curve.s*curve.q*curve.q
        var_score = np.sum( ws )
        return var_score

    

    def fix_panel(self, geom_panel, img, mask, asic_masks, offset_mask=None,bracket=(-40,40), eps=1.0):
       print "Panel Iteration"
       print "Searching for offsets in Bracket (%5.2f, %5.2f)"%(bracket[0],bracket[1]) 
       panel_offset = img*0.0 
       if offset_mask is None:
           offset_mask = img*0.0

       keys = asic_masks.keys()
       np.random.shuffle( keys )
       print "We use a random order of ASICS pedestals to be determined. We adjust one ASIC offset at a time."
       for asic in keys:
           am = asic_masks[asic]
           combo_mask = am*mask
           if np.max(combo_mask)> 0.5:
               this_offset = gss( self.variance_score, bracket[0], bracket[1], [img+offset_mask+panel_offset, am, mask, geom_panel],N=7,eps=eps)
               print "Correction for ASIC", asic ,"  ---> %5.3f "%this_offset 
               panel_offset = panel_offset+this_offset*am

       converged = False
       if np.max(np.abs(panel_offset) ) <= eps:
           converged = True
       max_offset = np.max(np.abs(panel_offset))
       return panel_offset+offset_mask, converged, max_offset


    def fixit(self, geometry, max_iter=30):
        self.split_median   = []
        self.split_sigma    = []
        self.split_geometry = []
        self.split_mask     = []
        self.split_splits   = geometry.split      
 
        self.split_asic_masks = self.split_asics(geometry)

        self.offset_masks   = []
        self.beam_offsets   = []


        for split_def, geom  in zip(geometry.split,geometry.cxcydz):
            sel_x = slice( split_def[0][0],split_def[0][1], 1 )
            sel_y = slice( split_def[1][0],split_def[1][1], 1 )
            this_panel_median = self.median[ sel_x, sel_y ]
            this_panel_sigma  = self.sigma[ sel_x, sel_y ]
            this_panel_mask   = self.mask[ sel_x, sel_y ]
            #plt.imshow( np.log(this_panel_median*this_panel_mask)   ); plt.show()
            wavelength        = 12398.0 / geometry.energy
            this_geometry     = geometrized_panel.detector_panel( this_panel_mask.shape, geometry.pixel, geom[2], wavelength, (geom[0], geom[1]) )
            saxs_curve        = this_geometry.get_saxs( this_panel_median, this_panel_mask, 0, 0, True )
            saxs_curve.show()
            self.split_median.append( this_panel_median )
            self.split_sigma.append( this_panel_sigma )
            self.split_geometry.append( this_geometry )
            self.split_mask.append( this_panel_mask )

        for img,mask,geom,asics in zip(self.split_median, 
                                       self.split_mask, 
                                       self.split_geometry,
                                       self.split_asic_masks
                                       ):
            print "---------- FIXING PANEL ----------"
            if self.ref_curve is not None:
                geom.refine_geometry( img, mask, self.ref_curve )

            offset_mask=None
            converged=False
            bracket=(-40,40)
            for ii in range(max_iter):
                offset_mask, converged, max_offset = self.fix_panel(geom, img, mask, asics, offset_mask, bracket)
                if converged:
                    break
                bracket = (-max_offset*1.2,max_offset*1.2)
            self.offset_masks.append( offset_mask ) 

            print "----------- PANEL FIXED ----------"
            print "\n\n\n" 

        print "Writing out the pedestals to %s"%('None yet')


        # now that we have fixed the panels, we need to find the offset between them
        saxs_curves =  []
        min_range   =  -1.0e8
        max_range   =   1.0e8
        N_points    =   1.0e8
        for img,mask,offset,geom in zip( self.split_median,
                                         self.split_mask,
                                         self.offset_masks,
                                         self.split_geometry ):
            saxs_corrected  = geom.get_saxs(img+offset,mask,0,0, no_corrections=True, q_min=self.q_min, q_max = self.q_max)
            saxs_curves.append( saxs_corrected )

            #print np.min(saxs_corrected.q), np.max(saxs_corrected.q), saxs_corrected.q.shape
            min_range = max( min_range , np.min(saxs_corrected.q)  )
            max_range = min( max_range , np.max(saxs_corrected.q)  )
            N_points  = min( N_points  , saxs_corrected.q.shape[0] )

     
        tmp_new_q_range = np.linspace( min_range, max_range, N_points )
        new_curves = []
        mean_curve = None
        for curve in saxs_curves:
            new_I = np.interp( tmp_new_q_range, curve.q, curve.I  )
            new_curves.append( new_I )
            if mean_curve is None:
                mean_curve = new_I*0
            mean_curve += new_I
        bp, p1, p2 = piecewise_fit.fit_two_pieces(  new_curves[0], new_curves[1] ) 

        self.offset_masks[1] = self.offset_masks[1] - p2[1]        
        tot_offset = self.build_offset_image()
        return tot_offset

        
    def build_offset_image(self):
        offsets = np.zeros((1024,1024))
        for pedestal, split_def in zip(self.offset_masks, self.split_splits):
            sel_x = slice( split_def[0][0],split_def[0][1], 1 )
            sel_y = slice( split_def[1][0],split_def[1][1], 1 ) 
            offsets[sel_x,sel_y] += pedestal 

            

        delta = np.mean(offsets)
        offsets = offsets-delta
        #get final saxs curves
        self.compute_best_saxs()
       
        return offsets

    def compute_best_saxs(self):
       self.template_saxs_curves = []
       self.template_intensity = []
       for img, mask, offset, geom in zip( self.split_median,
                                           self.split_mask, 
                                           self.offset_masks,
                                           self.split_geometry ):
           
           saxs_curve = geom.get_saxs( img+offset, mask, 0, 0, True, q_min=self.q_min, q_max=self.q_max ) # assume the central geometry is the best one
           self.template_saxs_curves.append(saxs_curve)
           self.template_intensity.append( np.sum( (img+offset)*mask  )  )
           saxs_curve.show()

    def score_image(self,img):
        curves = []
        score = 0
        z_score = 0
        this_int = 0.0
        for split_def, geom, offset, mask, curve, part_int in zip(self.split_splits, 
                                                  self.split_geometry,
                                                  self.offset_masks,
                                                  self.split_mask, 
                                                  self.template_saxs_curves,
                                                  self.template_intensity) :
            sel_x = slice( split_def[0][0],split_def[0][1], 1 )
            sel_y = slice( split_def[1][0],split_def[1][1], 1 )   
            part = img[sel_x,sel_y]
            saxs = geom.get_saxs( part + offset, mask, 0, 0, True, q_min=self.q_min, q_max=self.q_max )
            this_int += np.sum( (part + offset)*mask  )
            score += np.corrcoef(  saxs.I, curve.I )[0][1] #cointegration.engle_granger( saxs.I, curve.I)
            curves.append( saxs )
            ab = np.polyfit( saxs.I, curve.I, 1  )
            tmp = saxs.I*ab[0]+ab[1]
            z_score += np.mean(np.abs( tmp - curve.I )/curve.s)

            #ring = geom.q_ring( 25, part + offset, mask, 0,0  )

        merged_curve = saxs_tools.merge_curves( curves[0], curves[1] )

        score = score / 2.0
        z_score = z_score / 2.0
        return score, z_score, merged_curve, this_int

