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

	dice=[]
	voe=[]
	rvd=[]
	assd=[]
	mssd=[]


	for reference_vol in reference_vol_list:

		print 'Starting with volume %s' % reference_vol

		reference_volume_path = reference_vol
		submission_volume_path = os.path.join(submit_dir, os.path.basename(reference_vol))
		if os.path.exists(submission_volume_path):
			print 'Found corresponding submission file %s for reference file %s' % (reference_volume_path,submission_volume_path)

			# Load Ref and submission with Nibabel
			loaded_reference_volume=nb.load(reference_volume_path)
			loaded_submission_volume = nb.load(submission_volume_path)

			# Get the current voxelspacing
			reference_voxelspacing= loaded_reference_volume.header.get_zooms()[:3]

			# Get Numpy data and compress to int8
			loaded_reference_volume_data = (loaded_reference_volume.get_data()).astype(np.int8)
			loaded_submission_volume_data = (loaded_submission_volume.get_data()).astype(np.int8)


			# Calc metric and store them in dict
			print 'Start calculating metrics for submission file %s' % submission_volume_path

			print 'Found %s Lesion Pixels in file %s' % (np.count_nonzero(loaded_submission_volume_data==2),submission_volume_path)
			current_result = helpers.calc_metric.get_scores(loaded_submission_volume_data==2,loaded_reference_volume_data==2,reference_voxelspacing)

			print 'Found following results for submission file %s: %s' % (submission_volume_path,current_result)

			dice.append(current_result['dice'])
			voe.append(current_result['voe'])
			rvd.append(current_result['rvd'])
			assd.append(current_result['assd'])
			mssd.append(current_result['msd'])

	output_file.write("ISBIDiceComplete: %.2f\n" % np.mean(dice))
	output_file.write("ISBIVOEComplete: %.2f\n" % np.mean(voe))
	output_file.write("ISBIRVDComplete: %.2f\n" % np.mean(rvd))
	output_file.write("ISBIASSDComplete: %.2f\n" % np.mean(assd))
	output_file.write("ISBIMSSDComplete: %.2f\n" % np.mean(mssd))

	output_file.write("LBDiceComplete: %.2f\n" % np.mean(dice))
	output_file.write("LBVOEComplete: %.2f\n" % np.mean(voe))
	output_file.write("LBRVDComplete: %.2f\n" % np.mean(rvd))
	output_file.write("LBASSDComplete: %.2f\n" % np.mean(assd))
	output_file.write("LBMSSDComplete: %.2f\n" % np.mean(mssd))

	output_file.close()