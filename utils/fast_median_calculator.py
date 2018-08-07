import numpy as np
import matplotlib.pyplot as plt

class Fast_Median_Image(object):
    def __init__(self, min_step=1, eps=0):
        self.M = None
        self.step = None
        self.count = 0
        self.min_step = min_step
        self.eps = 0
        if eps > 0:
            self.eps = eps

    def update(self,data, no_os=True):
        if self.count == 0:
            self.M = data+0
            self.step = np.abs(data/2.0)
            sel = self.step < self.min_step
            self.step[sel ] = self.min_step

        else:
            # above
            sel_above = self.M > data
            sel_below = self.M < data
            sel_no_os = []

            if no_os:
                sel_no_os = np.abs(data - self.M) < self.step
            

            self.M[ sel_above ] = self.M[ sel_above ] - self.step[ sel_above ]
            self.M[ sel_below ] = self.M[ sel_below ] + self.step[ sel_below ]
            if no_os:
                self.M[ sel_no_os ] = data[sel_no_os]


            # step size
            sel = np.abs(data - self.M) < self.step
            self.step[sel] = self.step[sel]/2.0
            if self.eps > 0:
                self.step = self.step*(1+self.eps)

        self.count += 1

    def current_median(self):
        return self.M

    def current_step(self):
        return self.step


class ReservoirSampler(object):
    def __init__(self, shape, N ):
        self.reservoir =  np.zeros( (N, shape[0], shape[1])  )
        self.N = N
        self.count = 0

    def update(self, img):
        if self.count < self.N:
            self.reservoir[ self.count, :, : ] = img
            self.count += 1
        else:
            tau = np.random.random()
            if tau < self.N / self.count: # use this new guy
                this_index = np.random.randint(0,self.N,1)
                self.reservoir[ this_index, :, : ] = img 
                self.count +=1
            else:  # ignore this new guy
                self.count +=1
    

    def current_median(self):
        assert self.count >= self.N
        median = np.median( self.reservoir, axis=0 )
        return median

    def percentiles(self, p):
        assert self.count >= self.N
        percentile_low  = np.percentile( self.reservoir, p, axis=0)
        percentile_high = np.percentile( self.reservoir, 100-p, axis=0)
        return percentile_low,percentile_high

    def sigma(self):
        a,b = self.percentiles(25)
        d = b-a
        sigma_image = d/1.34
        return sigma_image


if __name__ == "__main__":
    N = 3
    M = 3
    FMI = Fast_Median_Image(1e-3,0)    
    K = 1000
    RS = ReservoirSampler( (N,M), K )

    data = []
    for ii in range(100000):
       X = np.random.normal(100.0,1,(N,M))
       data.append( X )
       FMI.update(X,False)
       RS.update( X )

    tmp_data = np.array(data)
    tmp_data = tmp_data.reshape( (ii+1,N,M) )    
    this_median = np.median(tmp_data, axis=0)
    Median = FMI.current_median()
    assert np.mean(  np.abs(100 - Median)/100 ) < 0.05
 

    RS_median = RS.current_median()
    assert np.mean(  np.abs(100 - RS_median)/100 ) < 0.05

    si = RS.sigma()
    assert np.mean(  np.abs(si-1) ) < 0.05

    print 'OK'

