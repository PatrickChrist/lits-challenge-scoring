from medpy import metric
import numpy as np
from scipy import ndimage
import time

from .surface import Surface


def dice(input1, input2):
    return metric.dc(input1, input2)


def detect_lesions(prediction_mask, reference_mask, min_overlap=0.5):
    """
    Produces a mask for predicted lesions and a mask for reference lesions,
    with label IDs matching lesions together. A (set of) lesion(s) in the
    reference is considered detected if a set of blobs in the prediction
    overlaps the set of blobs in the reference by `min_overlap` (intersection
    over union).
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :param min_overlap: float in range [0, 1.]
    :return: prediction mask (int),
             reference mask (int),
             num_detected,
             reduction in number of reference lesions due to merging,
             reduction in number of predicted lesions due to merging
    """
    
    # Initialize
    detected_mask = np.zeros(prediction_mask.shape, dtype=np.uint8)
    mod_reference_mask = np.zeros(prediction_mask.shape, dtype=np.uint8)
    num_detected = 0
    if not np.any(reference_mask):
        return detected_mask, num_detected
    
    if not min_overlap>0 and not min_overlap<=1:
        raise ValueError("min_overlap must be in [0, 1.]")
    
    # Get available IDs (excluding 0)
    # 
    # To reduce computation time, check only those lesions in the prediction 
    # that have any overlap with the ground truth.
    p_id_list = np.unique(prediction_mask[reference_mask.nonzero()])[1:]
    g_id_list = np.unique(reference_mask)[1:]
    
    # To reduce computation time, get views into reduced size masks.
    reduced_prediction_mask = prediction_mask.copy()
    for p_id in np.unique(prediction_mask):
        if p_id not in p_id_list:
            reduced_prediction_mask[p_id] = 0
    target_mask = np.logical_or(reference_mask, reduced_prediction_mask)
    bounding_box = ndimage.find_objects(target_mask)[0]
    r = reference_mask[bounding_box]
    p = prediction_mask[bounding_box]
    d = detected_mask[bounding_box]
    m = mod_reference_mask[bounding_box]

    # Compute intersection of predicted lesions with reference lesions.
    intersection_matrix = np.zeros((len(p_id_list), len(g_id_list)),
                                    dtype=np.int32)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = np.count_nonzero(np.logical_and(p==p_id, r==g_id))
            intersection_matrix[i, j] = intersection
    
    def sum_dims(x, axis, dims):
        '''
        Given an array x, collapses dimensions listed in dims along the 
        specified axis, summing them together. Returns the reduced array.
        '''
        x = np.array(x)
        if len(dims)==0:
            return x
        
        # Initialize output
        new_shape = list(x.shape)
        new_shape[axis] -= len(dims)-1
        x_ret = np.zeros(new_shape, dtype=x.dtype)
        
        # Sum over dims on axis
        sum_slices = [slice(None)]*x.ndim
        sum_slices[axis] = dims
        dim_sum = np.sum(x[sum_slices])
        
        # Remove all but first dim in dims
        mask = np.ones(x.shape, dtype=np.bool)
        mask_slices = [slice(None)]*x.ndim
        mask_slices[axis] = dims[1:]
        mask[mask_slices] = 0
        x_ret.ravel()[...] = x[mask]
        
        # Put dim_sum into array at first dim
        replace_slices = [slice(None)]*x.ndim
        replace_slices[axis] = [dims[0]]
        x_ret[replace_slices] = dim_sum
        
        return x_ret
            
    # Merge and label reference lesions that are connected by predicted
    # lesions.
    num_g_merged = 0
    for i, p_id in enumerate(p_id_list):
        # Merge columns, as needed
        g_id_intersected = g_id_list[intersection_matrix[i].nonzero()]
        num_g_merged += len(g_id_intersected)-1
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=1,
                                       dims=g_id_intersected-1)
        g_id_list = np.delete(g_id_list, obj=g_id_intersected[1:]-1)
        for g_id in g_id_intersected:
            m[r==g_id] = g_id_intersected[0]
    
    # Match each predicted lesion to a single (merged) reference lesion.
    max_val = np.max(intersection_matrix, axis=1)
    max_indices = np.argmax(intersection_matrix, axis=1)
    intersection_matrix[...] = 0
    intersection_matrix[np.arange(len(p_id_list)), max_indices] = max_val
    
    # Merge and label predicted lesions that are connected by reference
    # lesions.
    num_p_merged = 0
    for j, g_id in enumerate(g_id_list):
        # Merge rows, as needed
        p_id_intersected = p_id_list[intersection_matrix[:,j].nonzero()]
        num_p_merged += len(p_id_intersected)-1
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=0,
                                       dims=p_id_intersected-1)
        p_id_list = np.delete(p_id_list, obj=p_id_intersected[1:]-1)
        for p_id in p_id_intersected:
            d[p==p_id] = p_id_intersected[0]
    
    # Trim away lesions deemed undetected.
    num_detected = len(p_id_list)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = intersection_matrix[i, j]
            union = np.count_nonzero(np.logical_or(d==p_id, m==g_id))
            overlap_fraction = float(intersection)/union
            if overlap_fraction <= min_overlap:
                d[d==p_id] = 0
                num_detected -= 1
                
    return detected_mask, mod_reference_mask, \
           num_detected, num_g_merged, num_p_merged


def compute_tumor_burden(prediction_mask, reference_mask):
    """
    Calculates the tumor_burden and evalutes the tumor burden metrics RMSE and
    max error.
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :return: dict with RMSE and Max error
    """
    def calc_tumor_burden(vol):
        num_liv_pix=np.count_nonzero(vol>=1)
        num_les_pix=np.count_nonzero(vol==2)
        if num_liv_pix:
            return num_les_pix/float(num_liv_pix)
        return np.inf
    tumor_burden_r = calc_tumor_burden(reference_mask)
    tumor_burden_p = calc_tumor_burden(prediction_mask)

    tumor_burden_diff = tumor_burden_r - tumor_burden_p
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
    
    for i, obj_id in enumerate(np.unique(prediction_mask)):
        if obj_id==0:
            continue    # 0 is background, not an object; skip

        # Limit processing to the bounding box containing both the prediction
        # and reference objects.
        target_mask = (reference_mask==obj_id)+(prediction_mask==obj_id)
        bounding_box = ndimage.find_objects(target_mask)[0]
        p = (prediction_mask==obj_id)[bounding_box]
        r = (reference_mask==obj_id)[bounding_box]
        if np.any(p) and np.any(r):
            dice = metric.dc(p,r)
            jaccard = dice/(2.-dice)
            scores['dice'].append(dice)
            scores['jaccard'].append(jaccard)
            scores['voe'].append(1.-jaccard)
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
