import pickle
import numpy as np
import matplotlib.pyplot as plt

def fit_two_pieces(x,y,low_cut=2,high_cut=2,D=5):
    print "IN FITTING PW"
    sel =  np.argsort(x)
    nx = x[sel]
    ny = y[sel]
    nx = nx[low_cut:-high_cut]
    ny = ny[low_cut:-high_cut]

    bps = nx[D:-D]

    scores = []
    print "BPS", bps
    print "nX", nx
    print "nY", ny
    for bp in bps:
        sel = nx <= bp
        x1 = nx[sel]
        y1 = ny[sel]
        x2 = nx[~sel]
        y2 = ny[~sel]
        ab1 = np.polyfit(x1, y1, 1, full=True)
        ab2 = np.polyfit(x2, y2, 1, full=True)    
        s1 = np.corrcoef(x1, y1)[0][1]
        s2 = np.corrcoef(x2, y2)[0][1]
        scores.append( (s1 + s2)/2.0 )
    this_bp_index = np.argmax( scores  )
    best_bp = bps[ this_bp_index ]
    if best_bp > 2:
        sel = nx < best_bp
        x1 = nx[sel]
        y1 = ny[sel]
        x2 = nx[~sel]
        y2 = ny[~sel]
        piece_1 = np.polyfit(x1, y1, 1)
        piece_2 = np.polyfit(x2, y2, 1)
        return best_bp, piece_1, piece_2 
    else:
        piece_2 = np.polyfit(nx, ny, 1)
        return best_bp, piece_2, piece_2

