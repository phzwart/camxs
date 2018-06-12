import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import pickle

from io_tools import h5_io, basic_parser, saxs_tools
from geometry import panel_tools, mask_tools, find_center, refine_distance


class selection_builder(object):
    def __init__(self,q_object,phi_object,corrections,N=500):
        self.q_object   = q_object    # do i need to keep this around?
        self.phi_object = phi_object
        self.corrections= corrections
        self.N          = N
        self.q_lims     = ( np.min(self.q_object), np.max(self.q_object) )
        self.q_bins     = np.linspace( self.q_lims[0] , self.q_lims[1] , self.N+1 )
        self.q          = np.linspace( (self.q_bins[ 0 ] + self.q_bins[ 1 ])*0.5, 
                                       (self.q_bins[-1 ] + self.q_bins[-2 ])*0.5
                                     )
        self.sel        = [] 
        for ii in range(self.N):
            low_q   = self.q_bins[ii] 
            high_q  = self.q_bins[ii+1]
            tmp_sel = (self.q_object > low_q) & (self.q_object < high_q)
            self.sel.append(tmp_sel.flatten())
        
    def get_saxs(self, img, mask):
        I = []
        s = []

        top1   = img*mask*self.corrections
        top1   = top1.flatten()
 
        top2   = img*img*mask*self.corrections
        top2   = top2.flatten()
        bottom = mask.flatten()
        for ii in range(self.N):
            sel    = self.sel[ii]
            mean_I = np.sum( top1[sel]  ) / np.sum( bottom[sel] + 1e-12 )
            s_I    = np.sum( top2[sel]  ) / np.sum( bottom[sel] + 1e-12)
            I.append( mean_I )
            s.append( np.sqrt(s_I-mean_I*mean_I) )
        I = np.array(I)
        
        plt.plot(np.log(I)); plt.show()
        return 0,1,2 

  
  
          
 
class q_container(object):
    def __init__(self,Q,Phi,Corrections,selections):
        self.Q           = Q
        self.Phi         = Phi
        self.Corrections = Corrections
        self.selections  = selections



class detector_panel(object):
    def __init__(self, img_shape, pixel_size, distance, wavelength, center):
        # Store setup parameters
        self.pixel_size  =  pixel_size     # in meter
        self.distance    =  distance       # in meter
        self.wavelength  =  wavelength     # in Angstrom
        self.mean_center =  center         # in pixels
        self.img_shape   =  img_shape      # image shape
        
        # Build X and Y coordinate images
        self.x           = np.linspace( 0, self.img_shape[1]-1, self.img_shape[1]  )
        self.y           = np.linspace( 0, self.img_shape[0]-1, self.img_shape[0]  )
        self.Y,self.X    = np.meshgrid(self.x,self.y)

        self.delta_indices, self.Q_Phi_Corr_Sel  = self.compute_Q_Phi_maps()

    def this_q_map(self,dx,dy):
        dX = self.X - self.mean_center[0] + dx
        dY = self.Y - self.mean_center[1] + dy
        R  = np.sqrt(  dX*dX + dY*dY )
        Q  = np.sin( np.arctan( 0.5*(R*self.pixel_size)/self.distance ) )*4.0*np.pi/self.wavelength
        Phi = np.arctan2( dY, dX)  
        pd = self.distance/self.pixel_size
        Solid_Angle = self.pixel_size*self.distance/( np.power(dX*dX +dY*dY + pd*pd, 1.5 ) )
        Solid_Angle = Solid_Angle / np.max(Solid_Angle)
        sel_object  = selection_builder( Q, Phi, Solid_Angle )
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

    def get_saxs(self, img, mask, dx, dy ):
        sel_obj = self.Q_Phi_Corr_Sel[ self.delta_indices[ (dx,dy) ] ]
        q,I,s = sel_obj.selections.get_saxs(img, mask)



def img_compare(A,B,sel):
    AA = A[sel]
    BB = B[sel]
    score = np.corrcoef(AA,BB)
    score = score[0][1]
    return score
    


def run(inputs):
   # get inputs first
   inputs = basic_parser.read_and_parse(inputs)

   # get data 
   mask_params = inputs['mask']
   cart_data   = h5_io.data_reader(inputs['data']['filename'], 
                                   inputs['data']['mean_start'], 
                                   inputs['data']['mean_stop'],
                                   what=[inputs['data']['data_path'] ] 
                                  )[ inputs['data']['data_path'] ]
   mean_img    = stats.stats.trim_mean(cart_data,0.3,axis=0)

   # make mask
   Ny,Nx = mean_img.shape
   final_mask = mask_tools.auto_mask( mean_img )

   if type(mask_params) == type([]):
       for mp in mask_params:
           mask = mask_tools.build_it['rectangle']( Ny,  Nx,  mp ) 
           final_mask = final_mask*mask 
   else:
       mask = mask_tools.build_it['rectangle']( Ny,  Nx,  mask_params  )
       final_mask = final_mask*mask

   # Display image and mask    
   #plt.imshow( np.log(final_mask*mean_img) ); plt.show()   
 
   # make panel objects
   panel_names     = []
   panel_objects   = []
   panel_strippers = []
   panel_means     = []
   panel_masks     = []
   panel_mask_sel  = []

   for panel in inputs['panels']:
       p_name = panel['name'].strip() 
       pixel_size = panel['pixel_size'] 
       distance = panel['distance']
       center = ( panel['cx'], panel['cy']  )
       wavelength = inputs['end_station']['wavelength']
       split_def  = {}
       split_def[ p_name ] = panel['split']
       panel_img  = panel_tools.splitter( mean_img   , split_def ) 
       panel_mask = panel_tools.splitter( final_mask , split_def )
       #plt.imshow(np.log(panel_img[p_name]*panel_mask[p_name])); plt.colorbar(); plt.show()

       this_panel = panel_img[p_name]
       this_mask  = panel_mask[p_name] 

       det_panel_object = detector_panel(this_panel.shape, pixel_size, distance, wavelength, center)   
       # store panel geometry object and mean images
       panel_names.append(     p_name            )
       panel_objects.append(   det_panel_object  )
       panel_strippers.append( panel['split']    )
       panel_means.append(      this_panel        )
       panel_masks.append(      this_mask         )
       panel_mask_sel.append(   this_mask > 0.1 )


   # now walk over all images, we have to read them from file, cant keep all in memory

   img_scores   = []
   tot_ints     = []
   N_panels     = len(panel_names) 

   tot_int_mean = 0
   for kk in range(N_panels):
       this_panel = panel_means[kk]
       this_mask  = panel_masks[kk]
       tot_int_mean += np.sum( this_panel*this_mask )


   #--------------------------------------------------
   # here we actually process the data
   process_images = h5_io.h5_file_pointer( inputs['data']['filename'], 
                                            what=inputs['data']['data_path'], path='' )

   # these parameters govern selection of images
   cc_lims  = (inputs['selection']['cc_low'], inputs['selection']['cc_high']) 
   int_lims = (inputs['selection']['I_count_low'], inputs['selection']['I_count_high'])


   selected_images = []

   for KK in range( inputs['data']['process_start'] , inputs['data']['process_stop'] ):
       img = process_images[KK,:,:]
       # first we split the data
       panel_scores = []
       img_score = 0
       tot_int   = 0
       for JJ in range(N_panels):
           mean_panel = panel_means[JJ]
           this_panel = panel_tools.splitter( img, panel_strippers[ JJ ] )
           this_mask  = panel_masks[JJ]
           this_sel   = panel_mask_sel[JJ]
           #plt.imshow( this_panel*this_mask ); plt.show()        
           pscore = img_compare( this_panel, mean_panel, this_sel)
           img_score += pscore
           tot_int   += np.sum( this_panel*this_mask ) 

       img_score = img_score/2.0
       #print cc_lims[0],cc_lims[1],int_lims[0],int_lims[1], img_score, tot_int
       if img_score > cc_lims[0]:
           if img_score < cc_lims[1]:
               if tot_int > int_lims[0] :
                   if tot_int < int_lims[1]:
                       selected_images.append( KK  ) 

       img_scores.append( img_score  )
       tot_ints.append( tot_int  )
       print KK, img_score, tot_int, len(selected_images)


   sel_imgs = open('selected_images.pickle','w')
   pickle.dump( selected_image, sel_imgs)
   sel_imgs.close()

   #plt.plot( range(len(tot_ints)), tot_ints/tot_int_mean,    '.' ); plt.show()
   #plt.plot( range(len(img_scores)), img_scores, '.' ); plt.show()    
   #plt.plot( tot_ints/tot_int_mean, img_scores, '.'         ); plt.show()
   
   print len(selected_images)



if __name__ == "__main__":
    run(sys.argv[1])
