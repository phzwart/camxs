import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import pickle
from utils import panel_tools
from io_tools import h5_io, basic_parser, saxs_tools


class selection_builder(object):
    def __init__(self,q_object,phi_object,corrections,N=500):
        self.q_object   = q_object    # do i need to keep this around?
        self.phi_object = phi_object
        self.corrections= corrections
        self.N          = N
        self.q_lims     = ( np.min(self.q_object), np.max(self.q_object) )
        self.q_bins     = np.linspace( self.q_lims[0] , self.q_lims[1] , self.N+1 )
        self.q = (self.q_bins[0:-1]+self.q_bins[1:])/2.0
        self.sel        = [] 
        for ii in range(self.N):
            low_q   = self.q_bins[ii] 
            high_q  = self.q_bins[ii+1]
            tmp_sel = (self.q_object > low_q) & (self.q_object < high_q)
            self.sel.append(tmp_sel.flatten())
        
    def get_saxs(self, img, mask, no_corrections=False, q_min = 0, q_max = 1000):
        I = []
        mI= []
        s = []

        top1   = img*mask
        if not no_corrections:
            top1 = top1*self.corrections
        top1   = top1.flatten()
 
        top2   = img*img*mask
        if not no_corrections:
            top2 = top2*self.corrections
        top2   = top2.flatten()
        bottom = mask.flatten()

        sel = (self.q < q_max) & (self.q > q_min)
        inds = np.arange(self.N)
        inds = inds[sel]
        this_q = self.q[sel]
        N = len(this_q)
        #print "qqqq", this_q

        for ii in inds:
            this_sel    = self.sel[ii]
            mean_I = np.sum( top1[this_sel]  ) / np.sum( bottom[this_sel] + 1e-12 )
            s_I    = np.sum( top2[this_sel]  ) / np.sum( bottom[this_sel] + 1e-12)
            I.append( mean_I )
            s.append( np.sqrt(np.abs(s_I-mean_I*mean_I))/np.sum( bottom[this_sel] + 1e-12) )
            #these_Is =  bottom[sel]
            #these_Is = these_Is > 0.5
            #median_I =  top1[sel]
            #median_I = median_I[ these_Is ]
            #median_I = np.mean( median_I )
            #mI.append(median_I)
        I = np.array(I)
        
        mean_curve   = saxs_tools.curve(this_q,I,s)
        return mean_curve

    def q_ring( self, q_bin, img, mask, Nphi ):
        bphi = np.arange(Nphi)
        delta_phi = np.pi*2.0/Nphi
        result = np.zeros(Nphi)
        mask_mask = np.zeros(Nphi) + 0
        sel       = self.sel[q_bin]
        these_I   = img*mask
        mask_vals = mask.flatten()[sel]
        these_I   = these_I.flatten()[sel]
        these_phi = self.phi_object.flatten()[sel] 

        order = np.argsort(these_phi)
        these_phi = these_phi[ order ] + np.pi
        phi_bins  = (np.floor(these_phi/delta_phi) ).astype(np.int)

        these_I   = these_I[ order ]
        # place the observed intensities
        result[phi_bins] = these_I

        #indicate where the mask is
        mask_vals = mask_vals[order]
        msel = mask_vals > 0.9
        masked_phi_bins = phi_bins[ msel ] 
        mask_mask[ masked_phi_bins ] = 1
        result = result * mask_mask  - (1.0-mask_mask) * 1000
        return result
          
 
class q_container(object):
    def __init__(self,Q,Phi,Corrections,selections):
        self.Q           = Q
        self.Phi         = Phi
        self.Corrections = Corrections
        self.selections  = selections

class detector_panel(object):
    def __init__(self, img_shape, pixel_size, distance, wavelength, center, delta_pixel=1, Nq=None):
        # Store setup parameters
        self.pixel_size  =  pixel_size     # in meter
        self.distance    =  distance       # in meter
        self.wavelength  =  wavelength     # in Angstrom
        self.mean_center =  center         # in pixels
        self.img_shape   =  img_shape      # image shape
        self.delta_pixel =  delta_pixel
 
        # Build X and Y coordinate images
        self.x           = np.linspace( 0, self.img_shape[1]-1, self.img_shape[1]  )
        self.y           = np.linspace( 0, self.img_shape[0]-1, self.img_shape[0]  )
        self.Y,self.X    = np.meshgrid(self.x,self.y)
        self.Nq = Nq
        if self.Nq is None:
            self.Nq = int(self.img_shape[1]//2)

        self.delta_indices, self.Q_Phi_Corr_Sel  = self.compute_Q_Phi_maps(self.delta_pixel)

    def this_q_map(self,dx,dy):
        dX = self.X - self.mean_center[0] - dx
        dY = self.Y - self.mean_center[1] - dy
        R  = np.sqrt(  dX*dX + dY*dY )
        Q  = np.sin( np.arctan( 0.5*(R*self.pixel_size)/self.distance ) )*4.0*np.pi/self.wavelength
        Phi = np.arctan2( dY, dX)  
        pd = self.distance/self.pixel_size
        Solid_Angle = self.pixel_size*self.distance/( np.power(dX*dX +dY*dY + pd*pd, 1.5 ) )
        Solid_Angle = Solid_Angle / np.max(Solid_Angle)
        sel_object  = selection_builder( Q, Phi, Solid_Angle, self.Nq )
        return Q,Phi,Solid_Angle, sel_object 


    def compute_Q_Phi_maps(self, cdelta=0):
        x_range = range( -cdelta , cdelta+1)
        y_range = range( -cdelta , cdelta+1)
        # index lookup
        delta_indices = {}
        count = 0
        
        Q_Phi_maps = []

        for dx in x_range:
            for dy in y_range:
                # update the lookup
                delta_indices[ (dx,dy) ] = count
                count +=1                   
                tmp_Q, tmp_Phi, tmp_SA,tmp_sel = self.this_q_map(dx,dy)
                tmp_qc = q_container( tmp_Q, tmp_Phi, tmp_SA, tmp_sel )
                Q_Phi_maps.append( tmp_qc )
                print ('next_one')
                #plt.imshow( tmp_Q  ); plt.colorbar(); plt.show()
                #plt.imshow( tmp_Phi); plt.colorbar(); plt.show()
                #plt.imshow( tmp_SA ); plt.colorbar(); plt.show()
        return delta_indices, Q_Phi_maps 

    def get_saxs(self, img, mask, dx, dy , no_corrections=False, q_min=0, q_max=1):
        sel_obj = self.Q_Phi_Corr_Sel[ self.delta_indices[ (dx,dy) ] ]
        mean = sel_obj.selections.get_saxs(img, mask, no_corrections=no_corrections, q_min=q_min, q_max=q_max)
        return mean

    def q_ring(self, q_bin, img, mask, dx, dy, Nphi ): #delta_phi=2.0*np.pi/2048 ):
        sel_obj = self.Q_Phi_Corr_Sel[ self.delta_indices[ (dx,dy) ] ]
        ring    = sel_obj.selections.q_ring( q_bin, img, mask, nphi )
        return ring

    def all_rings(self, img, mask, dx, dy,  nphi):
        delta_phi= 2.0*np.pi/nphi
        sel_obj = self.Q_Phi_Corr_Sel[ self.delta_indices[ (dx,dy) ] ]
        these_q_bins = sel_obj.selections.q
        rings = []
        M = len(these_q_bins)
        phi = None
        for mm in range(M):
            I = sel_obj.selections.q_ring( mm, img, mask, nphi ) 
            rings.append( I )
        rings = np.array(rings).reshape((M,nphi))
        #plt.imshow(np.log(rings+1e-1),interpolation='none'); plt.show()
        return these_q_bins,rings

        


    def refine_distance(self, saxs_curve, ref_curve, delta_d=0.010,N=10003):
        dd = np.linspace( -delta_d, delta_d, N  )
        distance = self.distance + dd
        scores =  []        
        for this_d in distance:
            scale     = self.distance / this_d
            tmp_curve = saxs_tools.curve( saxs_curve.q*scale, saxs_curve.I, saxs_curve.I*0.03 )    
            #plt.semilogy(tmp_curve.q, tmp_curve.I); plt.semilogy(ref_curve.q, ref_curve.I); plt.show()
            score     = saxs_tools.compare_curves( ref_curve, tmp_curve, False ) 
            scores.append(score)
        ii = np.argmax(scores)
        return distance[ii],scores[ii]

    def refine_geometry(self,img,mask, ref_curve, use_q_fraction=(0.1,0.9), q_min=0, q_max=10 ):
        dxy = np.arange(-self.delta_pixel,self.delta_pixel+1)
        for dx in dxy:
            for dy in dxy:
                curve = self.get_saxs( img, mask, dx, dy, True, q_min=q_min, q_max=q_max )
                Nl = int(curve.q.shape[0]*use_q_fraction[0])
                Nh = int(curve.q.shape[0]*use_q_fraction[1])
                curve = saxs_tools.curve( curve.q[Nl:Nh],curve.I[Nl:Nh], curve.s[Nl:Nh])
                distance,score = self.refine_distance( curve, ref_curve)         
                print dx,dy,distance,score


    def mimimum_variance_saxs_center(self,img,mask,power=0.33):
        dxy = np.arange(-self.delta_pixel,self.delta_pixel+1)
        scores = []
        displacement_pairs = []
        for dx in dxy:
            for dy in dxy:
                displacement_pairs.append( (dx,dy)  ) 
                s = self.get_saxs( img, mask, dx, dy, True)
                score = np.sum( np.power( s.q, power )*s.s*s.s )
                print dx, dy, score
                scores.append(score)
        tt = np.argmin( scores )
        return displacement_pairs[ tt ] 



        


if __name__ == "__main__":
    run(sys.argv[1])
