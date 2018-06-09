import numpy as np
import matplotlib.pyplot as plt

class to_polar(object):
    def __init__(self, img, mask):
        self.mask = mask
        self.img = img
        self.Ny, self.Nx = mask.shape
        self.x = np.linspace(0,self.Nx-1,self.Nx)
        self.y = np.linspace(0,self.Ny-1,self.Ny)
        self.Y,self.X = np.meshgrid(self.x,self.y)

    def build_r_map(self,cx,cy):
        dX = self.X-cx
        dY = self.Y-cy
        R  = dX*dX + dY*dY
        R  = np.sqrt( R )
        return R

    def try_beam_center_stupid(self, cx, cy ):
        R_map = self.build_r_map(cx,cy) 
        # downsample the R map to only include the masked area 
        sel = self.mask > 0.5    
        R_map   = R_map[sel]     
        sel_img = self.img[sel]
        TR_map = R_map-np.min(R_map)
        TR_map = TR_map/np.max(TR_map)
        score = np.corrcoef( np.exp(-(TR_map.flatten()) )  , sel_img.flatten() ) 
        return score[0][1]

    def get_saxs(self, cx, cy, bins=100):
        R_map = self.build_r_map(cx,cy)
        sel = self.mask > 0.5
        R_map   = R_map[sel].flatten()
        sel_img = self.img[sel].flatten()
        min_r = np.min( R_map )
        max_r = np.max( R_map )
        r_range = np.linspace(min_r,max_r,bins)
        r_val   = r_range[0:-1]+r_range[1:]
        r_val   = r_val / 2.0
        I_val   = np.zeros(bins-1)
        sig_val = np.zeros(bins-1) 
        for ii in range(0,bins-1):
            low_r  = r_range[ii]
            high_r = r_range[ii+1]
            sel    = (R_map > low_r ) & (R_map < high_r) 
            mean   = np.mean( sel_img[sel]  )
            var    = np.mean( sel_img[sel]*sel_img[sel] ) - mean*mean
            I_val[ii]     = mean
            sig_val[ii]   =  np.sqrt(var)

        return r_val, I_val,sig_val

    def try_beam_center_variance(self, cx, cy, bins=300):
        r,I,s = self.get_saxs(cx,cy,bins) 
        score = np.sum( s*s )
        return -score 



def grid_search(pobj_function,cxs,cys, show=False):
    X,Y = np.meshgrid(cxs,cys)
    scores = np.zeros(Y.shape) 
    for ii,xx in enumerate(cxs):
        for jj,yy in enumerate(cys):
            score = pobj_function( xx,yy  )
            scores[jj][ii] = score*1.0+0.0
    uu = np.argmax(scores.flatten())

    if show:
        plt.imshow( scores ); plt.colorbar(); plt.show()

    best_x = X.flatten()[uu] 
    best_y = Y.flatten()[uu]
    return best_x,best_y

def grid_maker(x_range, y_range,N):
    cxs = range(x_range[0], x_range[1], N)
    cys = range(y_range[0], y_range[1], N)
    return cxs, cys

def slow_center_finder( img, mask, cx_range, cy_range,trials=4):
    polar_object = to_polar( img, mask )

    M  = 32
    best_x = None
    best_y = None
    print('First we do a very rough probing of where the beamcenter lies.')
    print('This is relatively fast.')
    for ii in range(trials):
        cxs, cys      = grid_maker(cx_range, cy_range,M)
        best_x,best_y = grid_search(polar_object.try_beam_center_stupid,cxs,cys)
        cx_range = (best_x-M*4, best_x+M*4)
        cy_range = (best_y-M*4, best_y+M*4)
        M = M//2 

    M = 2 
    cx_range = (best_x-M, best_x+M+1)
    cy_range = (best_y-M, best_y+M+1)
    print('We found %i %i'%(best_x,best_y))
    print('Now we try and improve on this. This can take a while...')
    for ii in range(500):
        cxs, cys = grid_maker(cx_range, cy_range,1)
        n_best_x,n_best_y = grid_search(polar_object.try_beam_center_variance,cxs,cys)
        cx_range = (best_x-M, best_x+M)
        cy_range = (best_y-M, best_y+M)
        d = np.abs(n_best_x-best_x) + np.abs(n_best_y-best_y)
        if d < 1:
            break
        best_x = n_best_x
        best_y = n_best_y
        if ii%100==0:
            print('Now at Iteration', ii)
     
    r,I,s = polar_object.get_saxs(best_x,best_y)
    plt.plot(r,np.log(I) ); plt.plot(r,np.log(s)); plt.show();
    print 'Best beam center found:', best_x, best_y
    return best_x, best_y, r,I,s


    

