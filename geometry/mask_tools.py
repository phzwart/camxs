import numpy as np
import matplotlib.pyplot as plt

default_circle = {'cx'     : 512,
                  'cy'     : 512,
                  'radius' : 50,
                  'inside' : 0,
                  'outside': 1
                 }

default_rectangle = { 'xstart'  : 400,
                      'xstop'   : 512,
                      'ystart'  : 300,
                      'ystop'   : 400,
                      'inside'  : 1,
                      'outside' : 0 
                    }

def parse_circle(line):
    pairs = line.split(',')
    result = {'cx'     : 0,
              'cy'     : 0,
              'radius' : 0,
              'inside' : 0,
              'outside': 1
             }
    for pair in pairs:
        tmp = pair.split('=')
        assert len(tmp)==2
        key = tmp[0]
        val = int(tmp[1])
        result[key]=val
    return result

def parse_rectangle(line):
    pairs = line.split(',')
    result = {        'xstart'  : 0,
                      'xstop'   : 1024,
                      'ystart'  : 0,
                      'ystop'   : 1024,
                      'inside'  : 1,
                      'outside' : 0
                    }
    for pair in pairs:
        tmp = pair.split('=')
        assert len(tmp)==2
        key = tmp[0]
        val = int(tmp[1])
        result[key]=val
    return result


def parse_mask_defs(defs):
    lines = defs.split(';')
    mask_params = {'circle'    : [], 
                   'rectangle' : [] }
    for line in lines:
        keys = line.split(':')
        if len(keys)==2:
            print(keys[0], keys[1])
            params = parse_it[ keys[0] ]( keys[1] )
            mask_params[ keys[0] ].append(  params )
    return mask_params


def circle(Nx,Ny,params):
    print params
    cx      = params['cx']
    cy      = params['cy']
    radius  = params['radius']
    inside  = params['inside']
    outside = params['outside']
    print cx,cy
    x = np.linspace(0,Nx-1,Nx)
    y = np.linspace(0,Ny-1,Ny)
    X,Y = np.meshgrid(x,y)
    dx = X-cx
    dy = Y-cy
    r  = np.sqrt(dx*dx+dy*dy)    
    sel = r < radius
    mask = np.zeros((Nx,Ny))+outside
    mask[sel]=inside
    return mask

def rectangle(Nx,Ny,params):
    print params
    xstart  = params['xstart' ]
    xstop   = params['xstop'  ]
    ystart  = params['ystart' ]
    ystop   = params['ystop'  ]
    inside  = params['inside' ] 
    outside = params['outside']

    mask = np.zeros( (Nx,Ny) )
    mask = mask+outside
    selx = slice(xstart,xstop,1) 
    sely = slice(ystart,ystop,1)
    mask[selx,sely]=inside
    return mask



def auto_mask(img,val=0):
    sel = (img==val)
    mask = img*0.0+1
    mask[sel]=0
    return mask

def make_mask(panel, mask_defs):
    Ny,Nx = panel.shape
    params = parse_mask_defs( mask_defs )
    print params
    masks = []
    for key in params.keys():
        for this_param in params[key]:
            mask = build_it[key](Nx,Ny,this_param)            
            masks.append(mask)
    result = masks[0]
    if len(masks)>1:
        for m in masks[1:]:
            result = result*m

    return result

parse_it = {'rectangle':parse_rectangle, 'circle':parse_circle}
build_it = {'rectangle':rectangle, 'circle':circle}

def tst():
    circle_def = "circle:cx=200,cy=200,radius=50,inside=0,outside=1;"
    rect_def = "rectangle:xstart=10,xstop=100,ystart=40,ystop=200,inside=0,outside=1;"
    mm = np.zeros( (400,400) )
    mask = make_mask(mm,rect_def+circle_def)
    plt.imshow(mask);plt.colorbar(); plt.show()



if __name__ == "__main__":
    tst()
    

