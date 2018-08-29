import numpy as np
import matplotlib.pyplot as plt

from utils.geometrized_panel import detector_panel

class correlation_accumulator(object):
    def __init__(self, N_q, N_phi, mask):
        self.N_q      = N_q
        self.N_phi    = N_phi
        self.c2       = np.zeros( (N_q,N_q,N_phi) )
        self.ac       = 0
        
        self.mean_img   = np.zeros((N_q,N_phi)) 
        self.mask       = mask
        self.mask_for_mean = mask*1+0
        self.FT_mask    = np.fft.rfft( mask, axis=1 )
        self.counter    = 0
        self.ac_counter = 0 

    def finish_up(self):
        # first we need to compute the correlation funcion of the mean
        c2_mean = self.update(self.mean_img/self.counter, self.mask_for_mean,store=False  )
        mean_c2 = self.c2/self.counter - c2_mean
        return mean_c2

    def finish_up_ac(self):
        # first we need to compute the correlation funcion of the mean
        ac_mean = self.update_ac_only(self.mean_img/self.ac_counter, self.mask_for_mean,store=False  )
        mean_ac = self.ac/self.ac_counter - ac_mean
        return mean_ac


    def update_ac_only(self, img, mask=1,eps=1e-8, store=True):
        tot_mask = self.mask*mask
        self.mean_img += img*mask
        self.mask_for_mean = self.mask_for_mean*mask
        FT_img  = np.fft.rfft( img*tot_mask, axis=1 )
        FT_mask = self.FT_mask
        if mask is not 1:
            FT_mask = np.fft.rfft( tot_mask, axis=1 )
        A = FT_img.conjugate()*FT_img
        M = FT_mask.conjugate()*FT_mask
        this_C2 = np.fft.irfft(A, axis=1) 
        this_M2 = np.fft.irfft(M, axis=1)
        # avoid divide by 0
        sel = this_M2 < eps
        this_M2[sel] = eps
        fin_AC  = this_C2/this_M2
        self.ac_counter += 1
        if store:
            self.ac += fin_AC
        else:
            return fin_AC

        return self.ac_counter

    def update(self, img, mask=1,eps=1e-8, store=True):

        tot_mask = self.mask*mask
        self.mean_img += img*mask
        self.mask_for_mean = self.mask_for_mean*mask

        _tmp_ = 0
        if not store:
           _tmp_ = self.c2*0

        # we now have an image and a mask, do the FFT for both
        FT_img  = np.fft.rfft( img*tot_mask, axis=1 )
        FT_mask = self.FT_mask
        if mask is not 1:
            FT_mask = np.fft.rfft( tot_mask, axis=1 )


        # now that we have rings, we can build the correlation functions
        for iq in range(self.N_q):
            Ai      = FT_img[iq,:]
            AiAj    = Ai.conjugate()*FT_img
            Mi      = FT_mask[iq,:]
            MiMj    = Mi.conjugate()*FT_mask
            C2i     = np.fft.irfft(AiAj, axis=1)#.real
            M2i     = np.fft.irfft(MiMj, axis=1)#.real
            sel     = np.abs(M2i) < eps 
            M2i[sel]=eps
            C2i     = C2i / M2i
            if store:
                self.c2[iq,:,:]+=C2i
            else:
                _tmp_[iq,:,:]+=C2i
            
        if not store:
            return _tmp_

        self.counter += 1
        return self.counter

class c2_prep(object):
    def __init__(self, static_mask, geom_def, eps=1e-12):
        self.eps = eps
        self.static_mask = static_mask
        self.geom_def = geom_def 


        self.splits   = []
        self.centers  = []
        self.distance = []
        self.energy = self.geom_def.energy
        self.wavelength = 12398.0 / self.energy
        self.pixel_scale = self.geom_def.pixel

        for split_def,cxcydz in zip(self.geom_def.split,self.geom_def.cxcydz) :
            sel_x = slice( split_def[0][0],split_def[0][1], 1 )
            sel_y = slice( split_def[1][0],split_def[1][1], 1 )
            self.splits.append( (sel_x, sel_y)  )
            self.distance.append( cxcydz[2])
            self.centers.append( (cxcydz[0],cxcydz[1]) )
        # now split the mask and build the geometrized_panels
        self.split_masks = self.split_img(self.static_mask)
        self.geom_panels = []

        self.q_bins = None 

        for split_mask, distance, cxcy in zip(self.split_masks,
                                              self.distance,
                                              self.centers) :
            geom_panel = detector_panel( img_shape  = split_mask.shape, 
                                         pixel_size = self.pixel_scale,
                                         distance   = distance,
                                         wavelength = self.wavelength,
                                         center     = cxcy,
                                         delta_pixel= 0,
                                         Nq         = self.geom_def.nq 
                                       )
            self.geom_panels.append( geom_panel )    
        
        # now we have to build a polar mask 
        self.split_polar_masks, self.final_polar_mask, self.q_bins = self.build_polar_mask()        

    def split_img(self,img):
       split_img = []
       for sel_pair in self.splits:
           sel_x = sel_pair[0]
           sel_y = sel_pair[1]
           tmp = img[sel_x, sel_y]
           split_img.append(tmp) 
       return split_img

    def build_polar_mask(self):
        these_polar_masks = []
        final_polar_mask  = 0
        these_bins = []
        bins = []
        for mask , geom in zip(self.split_masks, self.geom_panels):
           q_bins, p_mask = geom.all_rings( mask, mask*0+1, 0, 0, self.geom_def.nphi )
           sel = p_mask > 0.5
           p_mask[~sel]=0
           these_polar_masks.append( p_mask )
           final_polar_mask += p_mask
           these_bins.append( q_bins )
        return these_polar_masks, final_polar_mask, these_bins


    def to_polar(self, img):
        these_rings = []
        panels = self.split_img(img)
        result = 0
        for panel,mask,geom in zip(panels, self.split_masks, self.geom_panels):
            q_bins, panel_rings = geom.all_rings( panel, mask, 0, 0, self.geom_def.nphi )
            these_rings.append( panel_rings  )

        for p_mask,p_img in zip(self.split_polar_masks, these_rings):
            corrector = np.sum( p_mask*p_img, axis=1 ) / ( np.sum(p_mask, axis=1)+self.eps)
            p_img = p_img.T - corrector
            p_img = p_img.T*p_mask
            result += p_img
        
        return result

     
         





