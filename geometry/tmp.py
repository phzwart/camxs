import numpy as np
import matplotlib.pyplot as plt

def g(x,a):
    t = np.exp( -(x-a)*(x-a) )
    return t

def shift(x,f,g):
    tmp  = (f-g)*f
    norm = np.sum(tmp)
    a    = np.sum(tmp*x)/norm
    return -a

x = np.linspace(-10,10,1000)
f = g(x,0)
h = g(x,5)
for ii in range(10):
    a = shift(x,f,h)
    print(ii,a)
    h = g(x,a)
    plt.plot(x,f); plt.plot(x,h);plt.show()




