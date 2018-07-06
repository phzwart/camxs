import numpy as np

def slab_data(pnccd_np):
    pnccd_ij = np.zeros((1024,1024), dtype=pnccd_np.dtype)
    pnccd_ij[0:512, 0:512] = pnccd_np[0]
    pnccd_ij[512:1024, 0:512] = pnccd_np[1][::-1, ::-1]
    pnccd_ij[512:1024, 512:1024] = pnccd_np[2][::-1, ::-1]
    pnccd_ij[0:512, 512:1024] = pnccd_np[3]
    return pnccd_ij


def native_data(pnccd_ij):
    pnccd_np = np.zeros((4,512,512), dtype=pnccd_ij.dtype)
    pnccd_np[0][:, :] = pnccd_ij[:512, :512]
    pnccd_np[1][:, :] = pnccd_ij[512:1024, :512][::-1, ::-1]
    pnccd_np[2][:, :] = pnccd_ij[512:1024, 512:1024][::-1, ::-1]
    pnccd_np[3][:, :] = pnccd_ij[:512, 512:1024]
    return pnccd_np
