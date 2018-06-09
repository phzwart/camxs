import numpy as np


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
    return q,I,s

