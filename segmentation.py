#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from segmentation_utils import (
    normalize_image_intensities,
    expectation_maximization,
    full_probability_volumes,
    numpy_solution_from_pr,
    lesion_new
)

def segmentation_patient(patient_num):
    """
    Perform segmentation on the data of a specific patient.
    
    Parameters:
    patient_num (int): The patient number to process.
    
    Returns:
    refined_lesions_image (SimpleITK.Image): The final lesion segmentation image.
    results (dict): A dictionary containing the tissue percentages for each lesion.
    """
    
    base_folder = f"../data/Patient-{patient_num}/"
    images_folder = os.path.join(base_folder, "preprocessed/")
    transforms_folder = os.path.join(base_folder, "transforms/")
    atlas_folder = os.path.join(base_folder, "atlas/")
    segmentation_folder = os.path.join(base_folder, "segmentation/")
    
    # Create directory if it doesn't exist
    os.makedirs(segmentation_folder, exist_ok=True)
    
    # File paths
    dataFile = f"../data/Patient-{patient_num}/{patient_num}_T1_isovox.nii.gz"
    maskFile = f"../data/Patient-{patient_num}/{patient_num}_isovox_fg_mask.nii.gz"
    
    # Load brain image and mask
    brainImage = nib.load(dataFile).get_fdata()
    maskIndices = np.array(nib.load(maskFile).get_fdata(), dtype=bool)
    brainIntensities = brainImage[maskIndices]
    
    # Restore image for visualization/debugging
    restored_image = np.zeros_like(brainImage)
    restored_image[maskIndices] = brainIntensities
    
    # Read and normalize atlases
    atlases = []
    tissue_types = ['csf', 'gray', 'white']
    for key in tissue_types:
        atlas_path = os.path.join(atlas_folder, f"atlas_{key}.nii.gz")
        atlas_data = nib.load(atlas_path).get_fdata()
        atlases.append(atlas_data[maskIndices])
        
    atlases_norm = []
    for atlas in atlases:
        normalized_atlas = normalize_image_intensities(atlas)
        atlases_norm.append(normalized_atlas)

    # Expectation-Maximization
    GMM_means, GMM_variances, GMM_weights, posteriors = expectation_maximization(
        brainIntensities, atlases_norm, 0.6
    )
    
    # Full probability volumes
    nComponents = 3
    full_volumes = full_probability_volumes(brainImage, posteriors, nComponents)
    
    # Convert results to label image
    t1 = sitk.ReadImage(dataFile)  # T1 as reference image to copy metadata
    label_image = numpy_solution_from_pr(full_volumes, maskIndices, t1)
    
    # Load and mask FLAIR image
    flair_path = f"../data/Patient-{patient_num}/{patient_num}_FLAIR_isovox.nii.gz"
    flair = sitk.ReadImage(flair_path)
    brain_mask = sitk.ReadImage(f'../data/Patient-{patient_num}/{patient_num}_isovox_fg_mask.nii.gz')
    brain_mask = sitk.Cast(brain_mask > 0, sitk.sitkFloat32)
    flair_masked = flair * brain_mask
    
    # Refine lesion segmentation
    alpha = 2
    min_size = 5
    ce_radius = 2
    dil_radius = 4
    wm_threshold = 5
    gm_wm_threshold = 80
    factor = 3
    
    refined_lesions_image, results = lesion_new(
        label_image, flair_masked, alpha, factor, min_size, ce_radius, dil_radius,
        wm_threshold, gm_wm_threshold
    )
    sitk.WriteImage(refined_lesions_image, f'../data/Patient-{patient_num}/segmentation/refined_lesions_image.nii.gz')
    
    return refined_lesions_image, results

