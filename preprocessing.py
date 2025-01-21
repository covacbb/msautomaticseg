#!/usr/bin/env python
# coding: utf-8


import os
import glob
import ants
import nibabel as nib

from preprocessing_utils import (
    downsample_image,
    print_memory_usage,
    extract_brain_mask,
    anisotropic_diffusion,
    n4_bias_correction
)


def preprocessing_patient(patient_num):
    """
    Preprocesses medical imaging data for a specific patient by performing skull stripping,
    extracting and applying a brain mask, anisotropic diffusion filtering, and N4 bias correction.

    Parameters:
    - patient_num (int): The patient ID for which data is processed.
    - image_types (list of str): List of image modalities to process (e.g., "pd", "t1").
    """
    
    # Construct paths using the provided base_folder
    base_folder = f"../data/Patient-{patient_num}/"
    baseline_search_pattern = os.path.join(base_folder, "*.nii.gz").replace("\\", "/")
    baseline_processed_folder = os.path.join(base_folder, "preprocessed/")
    os.makedirs(baseline_processed_folder, exist_ok=True)

    print("Brain mask extraction")
    # Identify and process the T1 image for brain mask extraction
    t1_files = [
        f for f in glob.glob(baseline_search_pattern) 
        if 't1' in f.lower() and "lesionseg" not in f.lower()
    ]
    if not t1_files:
        raise FileNotFoundError(f"No T1 image found in folder {base_folder}")
    
    t1_image_path = t1_files[0].replace("\\", "/")
    t1_image = nib.load(t1_image_path)
    brain_mask_output_path = os.path.join(base_folder, "brain_mask_t1.nii.gz")
    extract_brain_mask(t1_image, brain_mask_output_path)

    # Process each image type
    image_types = ["pd", "t1", "t2", "flair"]
    for image_type in image_types:
        print("\t-------------------------------------")
        print(f"\t         Processing {image_type.upper()}        ")
        print("\t-------------------------------------")
    
        file_list = glob.glob(baseline_search_pattern)
        files = [
            file for file in file_list 
            if image_type in file.lower() and "lesionseg" not in file.lower()
        ]
        
        if not files:
            print(f"ERROR: There is no {image_type.upper()} image in folder {base_folder}")
            continue
        
        file_path=files[0]
        print(f"Processing file: {file_path}")
        
        # Apply anisotropic diffusion filter
        #downsample_image(file_path, file_path, scale_factor=0.5)
        adf_file_path = os.path.join(baseline_processed_folder, f"{image_type}_adf.nii.gz")
        print_memory_usage("Before processing")
        image_smooth = anisotropic_diffusion(file_path, adf_file_path)
        print_memory_usage("After processing")

        # Apply N4 bias correction
        mask_path = os.path.join(base_folder, 'brain_mask_t1.nii.gz')
        if not os.path.exists(mask_path):
            print(f"ERROR: Brain mask {mask_path.upper()} not found at base folder {base_folder}")
            continue
            
        mask = ants.image_read(mask_path)
        image_smooth = ants.image_read(adf_file_path)
        corrected_image = n4_bias_correction(image_smooth, mask)
        corrected_file_path = os.path.join(baseline_processed_folder, f"{image_type}_corrected.nii.gz")
        ants.image_write(corrected_image, corrected_file_path)

