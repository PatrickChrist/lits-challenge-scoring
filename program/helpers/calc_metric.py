from medpy import metric
from surface import Surface
import numpy as np



def get_scores(pred,label,vxlspacing):
    '''
    Calculates metrics scores from numpy arrays and returns an dict.
    :param pred: numpy.array
    :param label: numpy.array
    :param vxlspacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voew, rvd, assd, and msd
    '''
    # Test whether arrays have equal size
    if pred.shape!=label.shape:
        print 'Shapes to not match! Pred %s and Label %s' % (pred.shape,label.shape)
        raise AttributeError


    volscores = {}

    volscores['dice'] = metric.dc(pred,label)
    volscores['jaccard'] = metric.binary.jc(pred,label)
    volscores['voe'] = 1. - volscores['jaccard']
    volscores['rvd'] = metric.ravd(label,pred)

    if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
        volscores['assd'] = 0
        volscores['msd'] = 0
    else:
        evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
        volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
        volscores['msd'] = evalsurf.get_maximum_symmetric_surface_distance()

    return volscores
