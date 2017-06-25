#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import nibabel as nb
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import glob

import helpers.calc_metric
from helpers.calc_metric import (detect_lesions,
                                 compute_segmentation_scores,
                                 compute_tumor_burden)



# Check input directories.
submit_dir = os.path.join(sys.argv[1], 'res')
truth_dir = os.path.join(sys.argv[1], 'ref')
if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))
    sys.exit()
if not os.path.isdir(truth_dir):
    sys.exit()

# Create output directory.
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize results dictionaries
lesion_detection_stats = {'TP': 0, 'FP': 0, 'FN': 0}
lesion_segmentation_scores = {}
liver_segmentation_scores = {}
dice_per_case = {'lesion': [], 'liver': []}
dice_global_x = {'lesion': {'I': 0, 'S': 0},
                 'liver':  {'I': 0, 'S': 0}} # 2*I/S
tumor_burden_list = []

# Iterate over all volumes in the reference list.
reference_volume_list = glob.glob(truth_dir+'/*.nii')
for reference_volume in reference_volume_list:
    print("Starting with volume {}".format(reference_volume))
    reference_volume_path = reference_volume
    submission_volume_path = os.path.join(submit_dir,
                                          os.path.basename(reference_volume))
    if os.path.exists(submission_volume_path):
        print("Found corresponding submission file {} for reference file {}"
              "".format(reference_volume_path, submission_volume_path))

        # Load reference and submission volumes with Nibabel.
        reference_volume = nb.load(reference_volume_path)
        submission_volume = nb.load(submission_volume_path)

        # Get the current voxel spacing.
        voxel_spacing = reference_volume.header.get_zooms()[:3]

        # Get Numpy data and compress to int8.
        reference_volume_data = (reference_volume.get_data()).astype(np.int8)
        submission_volume_data = (submission_volume.get_data()).astype(np.int8)
        
        # Ensure that the shapes of the masks match.
        if submission_volume_data.shape!=reference_volume_data.shape:
            raise AttributeError("Shapes to not match! Prediction mask {}, "
                                 "ground truth mask {}"
                                 "".format(submission_volume_data.shape,
                                           reference_volume_data.shape))
        
        # Create lesion and liver masks with labeled connected components.
        # (Assuming there is always exactly one liver - one connected comp.)
        pred_mask_lesion = \
            label_connected_components(submission_volume_data==2)[0]
        true_mask_lesion = \
            label_connected_components(reference_volume_data==2)[0]
        pred_mask_liver = submission_volume_data>=1
        true_mask_liver = reference_volume_data>=1
        
        # Begin computing metrics.
        print("Start calculating metrics for submission file {}"
              "".format(submission_volume_path))
        
        # Identify detected lesions.
        detected_mask_lesion = detect_lesions(prediction_mask=pred_mask_lesion,
                                              reference_mask=true_mask_lesion,
                                              min_overlap=0.5)
        
        # Count true/false positive and false negative detections.
        TP = len(np.unique(detected_mask_lesion))
        FP = len(np.unique(pred_mask_lesion[detected_mask_lesion==0]))
        FN = len(np.unique(true_mask_lesion[detected_mask_lesion==0]))
        lesion_detection_stats['TP']+=TP
        lesion_detection_stats['FP']+=FP
        lesion_detection_stats['FN']+=FN
        
        # Compute segmentation scores for DETECTED lesions.
        lesion_scores = compute_segmentation_scores( \
                                          prediction_mask=detected_mask_lesion,
                                          reference_mask=true_mask_lesion,
                                          voxel_spacing=voxel_spacing)
        for metric in lesion_scores:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(lesion_scores[metric])
        
        # Compute liver segmentation scores. 
        liver_scores = compute_segmentation_scores( \
                                          prediction_mask=pred_mask_liver,
                                          reference_mask=true_mask_liver,
                                          voxel_spacing=voxel_spacing)
        for metric in liver_scores:
            if metric not in liver_segmentation_scores:
                liver_segmentation_scores[metric] = []
            liver_segmentation_scores[metric].extend(liver_scores[metric])
            
        # Compute per-case (per patient volume) dice.
        dice_per_case['lesion'].append(calc_metric.dice(pred_mask_lesion,
                                                        true_mask_lesion))
        dice_per_case['liver'].append(calc_metric.dice(pred_mask_liver,
                                                       true_mask_liver))
        
        # Accumulate stats for global (dataset-wide) dice score.
        dice_global_x['lesion']['I'] += np.logical_and(pred_mask_lesion,
                                                       true_mask_lesion).sum()
        dice_global_x['lesion']['S'] += pred_mask_lesion.sum() + \
                                        true_mask_lesion.sum()
        dice_global_x['liver']['I'] += np.logical_and(pred_mask_liver,
                                                      true_mask_liver).sum()
        dice_global_x['liver']['S'] += pred_mask_liver.sum() + \
                                       true_mask_liver.sum()
            
        ##TODO Compute tumor burden.
        tumor_burden = compute_tumor_burden(prediction_mask=pred_mask_lesion,
                                            reference_mask=true_mask_lesion)
        tumor_burden_list.append(tumor_burden)
        
        
# Compute lesion detection metrics.
TP = lesion_detection_stats['TP']
FP = lesion_detection_stats['FP']
FN = lesion_detection_stats['FN']
lesion_detection_metrics = {'precision': float(TP)/(TP+FP),
                            'recall': float(TP)/(TP+FN)}

# Compute lesion segmentation metrics.
lesion_segmentation_metrics = {}
for m in lesion_segmentation_scores:
    lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores)
lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
lesion_segmentation_metrics['dice_global'] = dice_global
    
# Compute liver segmentation metrics.
liver_segmentation_metrics = {}
for m in liver_segmentation_scores:
    liver_segmentation_metrics[m] = np.mean(liver_segmentation_scores)
liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
dice_global = 2.*dice_global_x['liver']['I']/dice_global_x['liver']['S']
liver_segmentation_metrics['dice_global'] = dice_global

##TODO Compute tumor burden.
tumor_burden_rmse = np.mean(np.sqrt(tumor_burden_list**2))
tumor_burden_max = np.max(tumor_burden_list)


# Print results to stdout.
print("Computed LESION DETECTION metrics:")
for metric, value in lesion_detection_metrics:
    print("{}: %.2f".format(metric, value))
print("Computed LESION SEGMENTATION metrics (for detected lesions):")
for metric, value in lesion_segmentation_metrics:
    print("{}: %.2f".format(metric, value))
print("Computed LIVER SEGMENTATION metrics:")
for metric, value in liver_segmentation_metrics:
    print("{}: %.2f".format(metric, value))
print("Computed TUMOR BURDEN: \n"
      "rmse: %.2f\nmax: %.2f".format(tumor_burden_rmse, tumor_burden_max))


# Write metrics to file.
output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'wb')
for metric, value in lesion_detection_metrics:
    output_file.write("lesion_{}: %.2f\n".format(metric, value))
for metric, value in lesion_segmentation_metrics:
    output_file.write("lesion_{}: %.2f\n".format(metric, value))
for metric, value in liver_segmentation_metrics:
    output_file.write("liver_{}: %.2f\n".format(metric, value))

#Tumorburden
output_file.write("RMSE_Tumorburden: %.2f\n".format(tumor_burden_rmse))
output_file.write("MAXERROR_Tumorburden: %.2f\n".format(tumor_burden_max))

output_file.close()
