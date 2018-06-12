import numpy as np
import matplotlib.pyplot as plt


class curve(object):
   def __init__(self, q, I, s):
       self.q = q
       self.I = I
       self.s = s

   def show(self):
       plt.figure(figsize=(10,10))
       plt.plot(self.q,np.log(self.I))
       plt.show()
 

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

