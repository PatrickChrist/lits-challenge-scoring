from medpy import metric
from surface import Surface
import numpy as np




def get_tumorburden_metric(pred,label):
    '''Calculates the tumorburden and evalutes the tumor burden metrics RMSE and max error
    :param pred: numpy.array
    :param label: numpy.array
    :param vxlspacing: list with x,y and z spacing
    :return: dict with RMSE and Max error
    '''
    tumorburden_label=calc_tumorburden(label)
    tumorburden_pred = calc_tumorburden(pred)

    tumorburden_diff = tumorburden_label - tumorburden_pred
    return tumorburden_diff

def calc_tumorburden(vol):
    num_liv_pix=np.count_nonzero(vol>=1)
    num_les_pix=np.count_nonzero(vol==2)

    tumorburden = np.divide(num_les_pix,num_liv_pix)
    return tumorburden

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
    volscores['precision'] = metric.binary.precision(label,pred)
    volscores['recall'] = metric.binary.recall(label,pred)
    volscores['obj_tpr'] =metric.binary.obj_tpr(label,pred)
    volscores['obj_fpr'] =metric.binary.obj_fpr(label,pred)

    if np.count_nonzero(pred) ==0 or np.count_nonzero(label)==0:
        volscores['assd'] = 0
        volscores['msd'] = 0
    else:
        evalsurf = Surface(pred,label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
        volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()
        volscores['msd'] = evalsurf.get_maximum_symmetric_surface_distance()

    return volscores
