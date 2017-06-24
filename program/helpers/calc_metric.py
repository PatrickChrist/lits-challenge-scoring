from medpy import metric
from surface import Surface
import numpy as np


def detect_lesions(prediction_mask, reference_mask, min_overlap=0.5):
    """
    Produces a mask containing predicted lesions that overlap by at least
    `min_overlap` with the ground truth. The label IDs in the output mask
    match the label IDs of the corresponding lesions in the ground truth.
    
    :param prediction_mask: numpy.array, int or bool
    :param reference_mask: numpy.array, int or bool
    :param min_overlap: float in range [0., 1.]
    :return: integer mask (same shape as input masks)
    """
    
    # Get available IDs
    prediction_ids = np.unique(prediction_mask)[1:]
    groundtruth_ids = np.unique(reference_mask)[1:]

    nb_pred_ids = len(prediction_ids)
    nb_true_ids = len(groundtruth_ids)

    # Compute the overlap between each pair of IDs
    overlap_matrix = np.zeros((nb_pred_ids, nb_true_ids), dtype="int")
    for i in range(nb_pred_ids):
        for j in range(nb_true_ids):
            overlap_matrix[i, j] = np.sum(( \
                        np.logical_and((prediction_mask==prediction_ids[i]),
                                       (reference_mask==groundtruth_ids[j])) ))
        
    # Produce output mask of detected lesions.
    detected_mask = np.zeros(prediction_mask.shape, dtype=np.uint32)
    for i in range(nb_pred_ids):
        for j in range(nb_true_ids):
            overlap_fraction = \
                float(overlap_matrix[i,j]) / np.sum(reference_mask==j)
            if overlap_fraction > min_overlap:
                detected_mask[prediction_mask==i] = j
                
    return detected_mask



def compute_tumor_burden(prediction_mask, reference_mask):
    """
    Calculates the tumor_burden and evalutes the tumor burden metrics RMSE and
    max error.
    
    #TODO: How are RMSE and max supposed to be derived from this measure?
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :return: dict with RMSE and Max error
    """
    def calc_tumor_burden(vol):
        num_liv_pix=np.count_nonzero(vol>=1)
        num_les_pix=np.count_nonzero(vol==2)
        tumor_burden = num_les_pix/float(num_liv_pix)
        return tumor_burden
    tumor_burden_label= calc_tumor_burden(label)
    tumor_burden_pred = calc_tumor_burden(pred)

    tumor_burden_diff = tumor_burden_label - tumor_burden_pred
    return tumor_burden_diff


def compute_segmentation_scores(prediction_mask, reference_mask,
                                voxel_spacing):
    """
    Calculates metrics scores from numpy arrays and returns an dict.
    
    Assumes that each object in the input mask has an integer label that 
    defines object correspondence between prediction_mask and 
    reference_mask.
    
    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """
    
    scores = {'dice': [],
              'jaccard': [],
              'voe': [],
              'rvd': [],
              'assd': [],
              'rmsd': [],
              'msd': []}
    
    for obj_id in np.unique(prediction_mask):
        p = prediction_mask==obj_id
        r = reference_mask==obj_id
        if np.count_nonzero(p) and np.count_nonzero(r):
            scores['dice']=np.append(scores['dice'],metric.dc(p,r))
            scores['dice']=np.asarray(scores['dice'])
            scores['jaccard']=np.append(scores['jaccard'],scores['dice']/(2.-scores['dice']))
            scores['jaccard']=np.asarray(scores['jaccard'])
            scores['voe']= np.append(scores['voe'],(1.-scores['jaccard']))
            scores['rvd'].append(metric.ravd(r,p))
            evalsurf = Surface(p, r,
                               physical_voxel_spacing=voxel_spacing,
                               mask_offset=[0.,0.,0.],
                               reference_offset=[0.,0.,0.])
            assd = evalsurf.get_average_symmetric_surface_distance()
            rmsd = evalsurf.get_root_mean_square_symmetric_surface_distance()
            msd = evalsurf.get_maximum_symmetric_surface_distance()
            scores['assd'].append(assd)
            scores['rmsd'].append(rmsd)
            scores['msd'].append(msd)
        else:
            # There are no objects in the prediction, in the reference, or both
            scores['dice'].append(0)
            scores['jaccard'].append(0)
            scores['voe'].append(1.)
            
            # Surface distance (and volume difference) metrics between the two
            # masks are meaningless when any one of the masks is empty. Assign 
            # maximum (infinite) penalty. The average score for these metrics,
            # over all objects, will thus also not be finite as it also loses 
            # meaning.
            scores['rvd'].append(np.inf)
            scores['assd'].append(np.inf)
            scores['rmsd'].append(np.inf)
            scores['msd'].append(np.inf)
              
    return scores
