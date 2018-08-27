import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import linregress


def guinier(q,Rg,LnI0, factor=1.2):
    y_calc = LnI0 - Rg*Rg*q*q/3.0
    sel = q *Rg < factor
    return y_calc, sel

def rg_estimate(curve,window=8):
    q = curve.q
    I = curve.I
    sel = I!=0
    q = q[sel]
    I = I[sel]

    x = q*q
    y = np.log(I)

    Rgs   = []
    lnI0s = []
    scores = []
    
    for ii in range(0, len(x)-window):
        slope,intercept,r,p,stderr = linregress(x[ii:ii+window],y[ii:ii+window])
        Rg = np.sqrt( -slope*3.0 )
        LnI0 = intercept
        if not np.isnan(Rg):
            y_calc,sel = guinier(q,Rg,LnI0)
            score_vals  = np.median( np.abs( (y[sel]-y_calc[sel]) / np.abs(y[sel]) )    )
            scores.append( score_vals )
            Rgs.append(Rg)
            lnI0s.append(LnI0)
    scores = np.array(scores)
    Rgs = np.array(Rgs)
    lnI0s = np.array(lnI0s)
    Rg_final = 0
    LnIo_final = -100
    Rg_final   = np.sum(Rgs/(1e-5+scores)) / np.sum( 1.0/(scores+1e-5) )
    LnIo_final = np.sum(lnI0s/(1e-5+scores)) / np.sum( 1.0/(scores+1e-5) )
    return Rg_final, LnIo_final
 


class curve(object):
   def __init__(self, q, I, s):
       self.q = np.array(q)
       self.I = np.array(I)
       self.s = np.array(s)
       # fix nan
       sel = np.isnan(self.I)
       self.I = self.I[~sel]
       self.q = self.q[~sel]
       self.s = self.s[~sel]
       
   def show(self):
       plt.figure(figsize=(12,8), facecolor='w')
       plt.plot(self.q,np.log(self.I))
       plt.show()


def compare_curves(A,B, log=True):
    A_min_q  = np.min(A.q) 
    A_max_q  = np.max(A.q)
    B_min_q  = np.min(B.q)
    B_max_q  = np.max(B.q)
    N_min_q  = max(A_min_q, B_min_q)
    N_max_q  = min(A_max_q, B_max_q)
    N_points = min( A.q.shape[0], B.q.shape[0] )

    new_q    = np.linspace( N_min_q, N_max_q, N_points )  
    new_A    = np.interp( new_q, A.q, A.I )
    new_B    = np.interp( new_q, B.q, B.I )
    new_sA   = np.interp( new_q, A.q, A.s ) 
    new_sB   = np.interp( new_q, B.q, B.s )
    if log :
        new_A = np.log( new_A - np.min(new_A) + 1 )
        new_B = np.log( new_B - np.min(new_B) + 1 )
    score    = np.corrcoef(new_A,new_B)[0][1]
    return score

def merge_curves(A,B):
    A_min_q  = np.min(A.q)
    A_max_q  = np.max(A.q)
    B_min_q  = np.min(B.q)
    B_max_q  = np.max(B.q)
    N_min_q  = max(A_min_q, B_min_q)
    N_max_q  = min(A_max_q, B_max_q)
    N_points = min( A.q.shape[0], B.q.shape[0] )

    new_q    = np.linspace( N_min_q, N_max_q, N_points )
    new_A    = np.interp( new_q, A.q, A.I )
    new_B    = np.interp( new_q, B.q, B.I )
    new_sA   = np.interp( new_q, A.q, A.s )
    new_sB   = np.interp( new_q, B.q, B.s )
    merged_I = 0.5*(new_A+new_B)
    merged_s = 0.5*(new_sA*new_sA + new_sB*new_sB)
    merged   = curve(new_q, merged_I, merged_s)
    return merged

def read_saxs(filename):
    f = open(filename,'r')
    q = []
    I = []
    s = []
    comments = []
    for line in f:
        if '#' not in line:
            keys = line[:-1].split()
            q.append( float(keys[0]) )
            I.append( float(keys[1]) )
            if len(keys)>2:
                s.append( float(keys[2]) )
    f.close()
    q = np.array(q)
    I = np.array(I)
    if len(s)>0:
        s = np.array(s)
    else:
        s = I*0.05
    saxs = curve(q,I,s)
    return saxs

