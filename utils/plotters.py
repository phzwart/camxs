import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def intensity_histogram( data, bins, x_label, y_label, legend, filename, markers , display=False):
    plt.figure(figsize=(18,10), facecolor='w')
    # first make a simple histogram
    plt.hist( data, bins, alpha=0.7, label=legend)

    # now plot the marker lines as vertical lines
    low  = markers[0] - markers[1]  
    high = markers[0] + markers[1]
    plt.axvline(x=low,color='k', linestyle='--')
    plt.axvline(x=high,color='k', linestyle='--')

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.yscale('log', nonposy='clip')
    plt.legend()
    plt.savefig(filename)
    if display:
        plt.show() 


def plot_equalized_template(img, filename, display=False, title=None):
    #fig = plt.figure(figsize=(13,10), facecolor='w')
    fig, ax = plt.subplots(1, figsize=(13,10), facecolor='w')
    cntrs = ax.imshow( img, interpolation='none' )
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    fig.colorbar(cntrs, ax=ax)
    if title is not None:
        ax.set_title(title, fontsize=18)
    fig.savefig(filename)
    if display:
        fig.show()
    

def plot_run_summaries(ccs, filename, display=False):
    N = len(ccs)
    t = np.arange(N)
  
    # first we plot a trace
 
    fig,ax = plt.subplots( 1, figsize=(18,10), facecolor='w')
    ax.plot( t, ccs, '.' )
    ax.set_xlabel('Image Index', fontsize=20 )
    ax.set_ylabel('Template correlation', fontsize=20 )
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    fig.savefig(filename+'.cc_trace.png')
    if display:
        fig.show()

    # now we do a histogram
    fig2,ax2 = plt.subplots(1, figsize=(18,10), facecolor='w')
    # first make a simple histogram
    ax2.hist( ccs, bins=100, alpha=0.7, label='Template correlation')

    # now plot the marker lines as vertical lines
    ax2.set_xlabel('Correlation', fontsize=20)
    ax2.set_ylabel('Occurance', fontsize=20)

    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=12)

    ax2.set_yscale('log', nonposy='clip')
    ax2.legend()
    fig2.savefig(filename+'.cc_histogram.png')
    if display:
        fig2.show()    
    

