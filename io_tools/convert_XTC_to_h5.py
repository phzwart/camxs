"""
This code reads in an XTC stream and produces an h5 file with panel separated.
The H5 file contains both front and back panel, an intensity sum and a median 
image. 

"""

import numpy as np
import h5py
import psana
import sys
from scipy import ndimage
import os.path
import time
from io_tools import basic_parser
from utils import panel_tools, fast_median_calculator

import matplotlib.pyplot as plt




default_parameters ="""
[data]
experiment        = None
run               = None
index_start       = 0
index_max         = 10
index_stride      = 1
common_mode_front = (3, 100, 100, 128)
common_mode_front = (3, 2000, 2000, 128)

[output]
filename          = None
comments          = None
"""

def getEventID(evt):
    try: 
        # timestamp
        evtid = evt.get(psana.EventId)
        seconds = evtid.time()[0]
        nanoseconds = evtid.time()[1]
        fiducials = evtid.fiducials()

        # photon energy
        ebeam = evt.get(psana.Bld.BldDataEBeamV7,psana.Source('BldInfo(EBeam)'))
        photonEnergy = ebeam.ebeamPhotonEnergy()

        return seconds, nanoseconds, fiducials, photonEnergy
    except: pass

    return None, None, None, None
        

def getMask(evt,det,assemble=False):

    mask = (det.mask(evt, calib=False, edges=True, central=True,\
            unbond=True, unbondnbrs=True)).copy()
    
    if assemble:
        mask = det.image(evt,mask)

    return mask

def getAssembledImg(evt,det,cmpars=None,assemble=False):
    calib = np.zeros((4,512,512))
    success=False
    try:

        # raw data
        raw = det.raw(evt)

        # calibrated data = pedestals, common mode, gain mask, gain, 
        # pixel status mask applies
        if cmpars is None:
            calib = det.calib(evt)
        else:
            calib = det.calib(evt,cmpars)

        if assemble:
            calib = det.image(evt,calib)
        success=True
    except: pass
    if calib is None:
        calib = np.zeros((4,512,512))
        success=False
    return calib, success


def run(config) :
    params = basic_parser.read_and_parse(  config , default_parameters  )
    params.show()

    h5_fname    = params.output.filename 
    exp_name    = params.data.experiment
    run_number  = str(params.data.run)
    max_events  = params.data.index_max
    start_index = params.data.index_start
    event_stride = params.data.index_stride

    exprun = 'exp='+exp_name+':run='+run_number+':idx'
    ds = psana.DataSource(exprun)
    env = ds.env()
    run = ds.runs().next()
    times = run.times()
    print 'total number of events in XTC file %d' %(len(times))
    max_events = min((len(times)-start_index), max_events)
    print 'We will only export %i of these events'%max_events
    print '   starting at %i with a stride of %i'%(start_index, event_stride)

    stop_index  = start_index+max_events
    my_event_indices = np.array( range( start_index, stop_index, event_stride ) )
    event_list = []

    fiducial_record = []
    time_record     = []

    for this_index in my_event_indices:
        event_list.append( times[ this_index ] )
        print (times[ this_index ].fiducial(), times[ this_index ].time() )
        fiducial_record.append( times[ this_index ].fiducial() )
        time_record.append( times[ this_index ].time() )
    fiducial_record = np.array( fiducial_record  )
    time_record = np.array( time_record )

    # (use ds.events().next().keys() to get src and alias
    srcList = ['Camp.0.pnCCD.0','Camp.0.pnCCD.1']    
    aliasList = ['pnccdFront','pnccdBack']   
    detList = [psana.Detector(src,env) for src in aliasList]


    # name of h5 file
    grp_name = 'data'
    adu_front_datafield = 'adu_front'
    mask_front_datafield = 'mask_front'
    adu_back_datafield = 'adu_back'
    mask_back_datafield = 'mask_back'

    if os.path.isfile(h5_fname):
            f = h5py.File(h5_fname,'r+')
    
            if grp_name in f:
                del f[grp_name]
    
            if mask_front_datafield in f:
                del f[mask_front_datafield]
    
            if mask_back_datafield in f:
                del f[mask_back_datafield]

            if mask_front_datafield in f:
                del f[adu_front_datafield]

            if mask_back_datafield in f:
                del f[adu_back_datafield]

            f.close()

    # parallel h5py file processing
    f = h5py.File(h5_fname,'w')
    grp = f.create_group(grp_name)

    # lets get the comon mode parameters
    front_comm_mode_pars = params.data.common_mode_front
    back_comm_mode_pars = params.data.common_mode_back


    # lets make a provenance field that contains some info on what data this is
    dt = h5py.special_dtype(vlen=bytes)
    prov          = f.create_group('provenance')
    prov.create_dataset('exp_id'           , data = 'exp='+exp_name+':run='+run_number , dtype=dt)
    prov.create_dataset('common_mode_front', data = front_comm_mode_pars               , dtype='uint8')
    prov.create_dataset('common_mode_back' , data = back_comm_mode_pars                , dtype='uint8')
    prov.create_dataset('event_fiducials'  , data = fiducial_record                    , dtype='uint64')
    prov.create_dataset('event_time'       , data = time_record                        , dtype='uint64')
    prov.create_dataset('user_comments'    , data = params.output.comments             , dtype=dt)

    adu_front_ds = None
    mask_front = None
    adu_back_ds = None
    mask_back = None
   
    # make two median objects
    front_median_object = fast_median_calculator.Fast_Median_Image()
    back_median_object  = fast_median_calculator.Fast_Median_Image()
    front_sum_int = []
    back_sum_int = []
    faults = 0
    for n,evt in enumerate(event_list):
        print n,evt.time()
        evt = run.event(evt)
        sec,ns,fid,phot_en = getEventID(evt)
        adu_back  = np.zeros((1024,1024))
        adu_front = np.zeros((1024,1024))
        if sec is not None:
            if n == 0:
                mask_front = panel_tools.slab_data( getMask(evt,detList[0],assemble=False) )
                mask_back  = panel_tools.slab_data( getMask(evt,detList[1],assemble=False) )
            adu_front, ok  = getAssembledImg(evt,detList[0],cmpars=front_comm_mode_pars,assemble=False)
            if not ok:
                faults += 1

            adu_front  = panel_tools.slab_data(adu_front).astype('float32')

            #adu_back, ok   = getAssembledImg(evt,detList[1],cmpars=back_comm_mode_pars,assemble=False)
            #adu_back       = panel_tools.slab_data(adu_back).astype('float32')
        else:
            faults += 1
        if adu_front_ds is None:
            adu_front_ds = grp.create_dataset(adu_front_datafield,(len(event_list), 1024, 1024), dtype='float32')
        if adu_back_ds is None:
            adu_back_ds = grp.create_dataset(adu_back_datafield,(len(event_list),1024, 1024), dtype='float32')
        adu_front_ds[n,:,:] = adu_front
        adu_back_ds[n,:,:] = adu_back
        front_sum_int.append( np.sum( adu_front.flatten() ) )
        back_sum_int.append( np.sum( adu_back.flatten() ) )
        # update median estimates
        front_median_object.update( adu_front  )
        back_median_object.update(  adu_back   )
        
        
          


    grp.create_dataset(mask_front_datafield,data=mask_front,dtype='uint8')
    grp.create_dataset(mask_back_datafield,data=mask_back,dtype='uint8')


    median = f.create_group('summary')
    median.create_dataset('front_median' , data = front_median_object.current_median(), dtype='float32')
    median.create_dataset('back_median'  , data = back_median_object.current_median() , dtype='float32')
    median.create_dataset('front_sum_int', data = front_sum_int                       , dtype='float32')
    median.create_dataset('back_sum_int' , data = back_sum_int                        , dtype='float32')
    

    f.close()


if __name__ == "__main__":
    run(sys.argv[1])
