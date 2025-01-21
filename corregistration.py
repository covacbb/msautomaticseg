#!/usr/bin/env python
# coding: utf-8

import os
import glob
import nibabel as nib
import ants

from corregistration_utils import(
    multi_affine_registration,
    resample_image,
    read_image
)

def corregistration_patient(patient_num):
    """
    Performs coregistration of the preprocessed images, aligning T1, T2, 
    and the brain mask to the FLAIR reference image using multi-affine registration.

    Parameters:
    - patient_num (int): The patient ID used to locate the data folder.

    Returns:
    - None: The function saves the registered images (T1, T2, and brain mask) to the "transforms" folder 
      under the patient's directory.
    """
    
    base_folder = f"../data/Patient-{patient_num}/"
    images_folder = os.path.join(base_folder, "preprocessed/")
    transforms_folder = os.path.join(base_folder, "transforms/")
    
    # Ensure directory exists
    os.makedirs(transforms_folder, exist_ok=True)
    
    # Read images
    t1 = read_image(os.path.join(images_folder, "t1_corrected.nii.gz"))
    t2 = read_image(os.path.join(images_folder, "t2_corrected.nii.gz"))
    mask = read_image(os.path.join(base_folder, "brain_mask.nii.gz"))
    flair = read_image(os.path.join(images_folder, "flair_corrected.nii.gz"))
    
    # Coregister T1 to FLAIR using multi-affine registration
    t1_to_flair_transform = multi_affine_registration(flair, t1)
    t1_registered = resample_image(flair, t1, t1_to_flair_transform)
    t1_registered_clamped = ants.threshold_image(t1_registered, low_thresh=0, binary=False)
    ants.image_write(t1_registered_clamped, os.path.join(transforms_folder, "t1_registered.nii.gz"))

    # Coregister MASK to FLAIR using the t1_to_flair_transform
    mask_registered = resample_image(flair, mask, t1_to_flair_transform)
    mask_registered_binary = ants.threshold_image(mask_registered, low_thresh=0.5, high_thresh=None, inval=1, outval=0)
    ants.image_write(mask_registered_binary, os.path.join(transforms_folder, "brain_mask_registered.nii.gz"))
    
    # Coregister T2 to FLAIR using multi-affine registration
    t2_to_flair_transform = multi_affine_registration(flair, t2)
    t2_registered = resample_image(flair, t2, t2_to_flair_transform)
    t2_registered_clamped = ants.threshold_image(t2_registered, low_thresh=0, binary=False)
    ants.image_write(t2_registered_clamped, os.path.join(transforms_folder, "t2_registered.nii.gz"))

