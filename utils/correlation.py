import numpy as np
import matplotlib.pyplot as plt



class correlation_accumulator(object):
    def __init__(self, N_q, N_phi, mask):
        self.N_q      = N_q
        self.N_phi    = N_phi
        self.c2       = np.zeros( (N_q,N_q,N_phi) )
        self.mean_img = np.zeros((N_q,N_phi)) 
        self.mask     = mask
        self.mask_for_mean = mask*1+0
        self.FT_mask  = np.fft.rfft( mask, axis=1 )
        self.counter  = 0

    def finish_up(self):
        # first we need to compute the correlation funcion of the mean
        c2_mean = self.update(self.mean_img/self.counter, self.mask_for_mean,store=False  )
        mean_c2 = self.c2/self.counter - c2_mean
        return mean_c2


    def update(self, img, mask=1,eps=1e-8, store=True):

        tot_mask = self.mask*mask
        self.mean_img += img*mask
        self.mask_for_mean = self.mask_for_mean*mask

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
                return C2i

        self.counter += 1
        return self.counter


