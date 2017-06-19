#!/usr/bin/env python
import sys, os, os.path
import nibabel as nb
import glob
import helpers.calc_metric
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print "%s doesn't exist" % submit_dir

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')

    reference_vol_list = glob.glob(truth_dir + '/*.nii')


    # Iterate over all volumes in the reference list

    dice_liv=[]
    voe_liv=[]
    rvd_liv=[]
    assd_liv=[]
    mssd_liv=[]
    precision_liv=[]
    recall_liv =[]


    dice_les = []
    voe_les = []
    rvd_les = []
    assd_les = []
    mssd_les = []

    precision_les = []
    recall_les = []
    obj_tpr_les =[]
    obj_fpr_les = []


    tumor_burden_rmse = []
    tumor_burden_max_error = []

    for reference_vol in reference_vol_list:

        print 'Starting with volume %s' % reference_vol

        reference_volume_path = reference_vol
        submission_volume_path = os.path.join(submit_dir, os.path.basename(reference_vol))
        if os.path.exists(submission_volume_path):
            print 'Found corresponding submission file %s for reference file %s' % (reference_volume_path, submission_volume_path)

            # Load Ref and submission with Nibabel
            loaded_reference_volume = nb.load(reference_volume_path)
            loaded_submission_volume = nb.load(submission_volume_path)

            # Get the current voxelspacing
            reference_voxelspacing = loaded_reference_volume.header.get_zooms()[:3]

            # Get Numpy data and compress to int8
            loaded_reference_volume_data = (loaded_reference_volume.get_data()).astype(np.int8)
            loaded_submission_volume_data = (loaded_submission_volume.get_data()).astype(np.int8)


            # Calc metric and store them in dict
            print 'Start calculating metrics for submission file %s' % submission_volume_path

            num_lesion_in_reference = np.count_nonzero(loaded_reference_volume_data == 2)
            print 'Found %s Lesion Pixels in file %s' % (num_lesion_in_reference, submission_volume_path)

            # Check whether ground truth contains lesions
            if num_lesion_in_reference==0:
                print 'No lesions in ground truth'
                num_lesion_in_submission = np.count_nonzero(loaded_submission_volume_data==2)
                if num_lesion_in_submission ==0:
                    print 'No lesions in prediction, well done'

                    current_result_les['dice'] = 1.
                    current_result_les['voe'] = 0.
                    current_result_les['rvd'] = 0.
                    current_result_les['assd'] = 0.
                    current_result_les['msd'] = 0.

                    current_result_les['precision'] = 0.
                    current_result_les['recall'] = 0.
                    current_result_les['obj_tpr'] = 0.
                    current_result_les['obj_fpr'] = 0.


                elif num_lesion_in_submission!=0:
                    loaded_reference_volume_data = np.zeros_like(loaded_reference_volume_data)
                    # Set one pixel to be a lesion pixel to avoid divding through zero
                    loaded_reference_volume_data[0,0,0] = 2
                    current_result_les = helpers.calc_metric.get_scores(loaded_submission_volume_data == 2,
                                                                        loaded_reference_volume_data == 2,
                                                                        reference_voxelspacing)

            else:
                current_result_les = helpers.calc_metric.get_scores(loaded_submission_volume_data==2,
                                                                    loaded_reference_volume_data==2,
                                                                    reference_voxelspacing)

            # Check whether liver exists in Reference

            num_liver_in_submission = np.count_nonzero(loaded_submission_volume_data == 1)

            # Calculate liver metrics
            if num_liver_in_submission==0:
                loaded_submission_volume_data[0,0,0] = 1
                current_result_liv = helpers.calc_metric.get_scores(loaded_submission_volume_data >= 1,
                                                                    loaded_reference_volume_data >= 1,
                                                                    reference_voxelspacing)
            else:
                current_result_liv = helpers.calc_metric.get_scores(loaded_submission_volume_data>=1,
                                                                    loaded_reference_volume_data>=1,
                                                                    reference_voxelspacing)


            # Calculate tumorburden
            if num_liver_in_submission!=0 and num_lesion_in_reference!=0:
                tumorburden_diff=helpers.calc_metric.get_tumorburden_metric(loaded_submission_volume_data,
                                                                            loaded_reference_volume_data)
            else:
                num_lesion_in_submission = np.count_nonzero(loaded_submission_volume_data == 2)
                if num_lesion_in_submission!=0:
                    tumorburden_diff=1.0
                else:
                    tumorburden_diff=0.0

            print 'Found following results for submission file %s: %s\n %s\n %s' % (submission_volume_path,current_result_les,current_result_liv,tumorburden_diff)

            # Saving the current results
            ## Results for lesion

            dice_les.append(current_result_les['dice'])
            voe_les.append(current_result_les['voe'])
            rvd_les.append(current_result_les['rvd'])
            assd_les.append(current_result_les['assd'])
            mssd_les.append(current_result_les['msd'])

            precision_les.append(current_result_les['precision'])
            recall_les.append(current_result_les['recall'])
            obj_tpr_les.append(current_result_les['obj_tpr'])
            obj_fpr_les.append(current_result_les['obj_fpr'])

            ## Results for liver
            dice_liv.append(current_result_liv['dice'])
            voe_liv.append(current_result_liv['voe'])
            rvd_liv.append(current_result_liv['rvd'])
            assd_liv.append(current_result_liv['assd'])
            mssd_liv.append(current_result_liv['msd'])

            precision_liv.append(current_result_liv['precision'])
            recall_liv.append(current_result_liv['recall'])

            ## Results
            tumor_burden_rmse.append(np.abs(tumorburden_diff))
            tumor_burden_max_error.append(tumorburden_diff)

    # Writing Output
    ## Output for lesion
    output_file.write("Dice_Lesion: %.2f\n" % np.mean(dice_les))
    output_file.write("VOE_Lesion: %.2f\n" % np.mean(voe_les))
    output_file.write("RVD_Lesion: %.2f\n" % np.mean(rvd_les))
    output_file.write("ASSD_Lesion: %.2f\n" % np.mean(assd_les))
    output_file.write("MSSD_Lesion: %.2f\n" % np.mean(mssd_les))

    output_file.write("Precision_Lesion: %.2f\n" % np.mean(precision_les))
    output_file.write("Recall_Lesion: %.2f\n" % np.mean(recall_les))
    output_file.write("OBJ_TPR_Lesion: %.2f\n" % np.mean(obj_tpr_les))
    output_file.write("OBJ_FPR_Lesion: %.2f\n" % np.mean(obj_fpr_les))

    ## Output for liver
    output_file.write("Dice_Liver: %.2f\n" % np.mean(dice_liv))
    output_file.write("VOE_Liver: %.2f\n" % np.mean(voe_liv))
    output_file.write("RVD_Liver: %.2f\n" % np.mean(rvd_liv))
    output_file.write("ASSD_Liver: %.2f\n" % np.mean(assd_liv))
    output_file.write("MSSD_Liver: %.2f\n" % np.mean(mssd_liv))

    output_file.write("Precision_Liver: %.2f\n" % np.mean(precision_liv))
    output_file.write("Recall_Liver: %.2f\n" % np.mean(recall_liv))

    ## Output for tumorburden
    output_file.write("RMSE_Tumorburden: %.2f\n" % np.mean(tumor_burden_rmse))
    output_file.write("MAXERROR_Tumorburden: %.2f\n" % np.max(tumor_burden_max_error))
    output_file.close()
