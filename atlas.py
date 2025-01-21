#!/usr/bin/env python
# coding: utf-8

import os
import glob
import ants
import SimpleITK as sitk

from atlas_utils import(
    apply_brain_mask,
    extract_brain_mask,
    rigid_registration,
    affine_registration,
    normalize_image,
    histogram_matching,
    calculate_ncc,
    get_atlases
)

def atlas_patient(patient_num):
    """
    Processes atlas-based registration and tissue-specific atlas generation for a given patient.

    Parameters:
    - patient_num (int): The patient ID used to locate the data folder.

    Returns:
    - None: The function saves registered atlas images and similarity maps to the patient's atlas folder.
    """
    base_folder = f"../data/Patient-{patient_num}/"
    transforms_folder = os.path.join(base_folder, "transforms/")
    atlas_folder = os.path.join(base_folder, "atlas/")
    atlases_folder = '../atlases/'
    
    # Create directory if it doesn't exist
    os.makedirs(atlas_folder, exist_ok=True)
    
    # Image paths
    t1_path = os.path.join(transforms_folder, 't1_registered.nii.gz')
    mask_path = os.path.join(transforms_folder, 'brain_mask_registered.nii.gz')
    atlas_template_path = os.path.join(atlases_folder, 'ICBM_Template.nii.gz')
    mask_atlas_path = os.path.join(atlases_folder, 'ICBM_Template_mask.nii.gz')
    
    # Ensure files exist
    for path in [t1_path, mask_path, atlas_template_path, mask_atlas_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    t1 = ants.image_read(t1_path)
    mask = ants.image_read(mask_path)
    atlas = ants.image_read(atlas_template_path)
    mask_atlas = ants.image_read(mask_atlas_path)
    
    #atlas_template = nib.load(atlas_template_path)
    #mask_atlas = extract_brain_mask(atlas_template, '../atlases/ICBM_Template_mask.nii.gz')
    
    # Apply brain masks
    t1_masked = apply_brain_mask(t1, mask)
    atlas_masked = apply_brain_mask(atlas, mask_atlas)
    
    # Atlas registration to T1 image
    rigid_transforms, _ = rigid_registration(t1_masked, atlas_masked)
    affine_transforms, _ = affine_registration(t1_masked, atlas_masked, initial_transform=rigid_transforms)
    #bspline_transforms, _ = bspline_multi_registration(mask_image, moving_image, initial_transform=affine_transforms)
    
    affine_transform_file_path = os.path.join(atlas_folder, "affine_transforms.mat")
    ants.write_transform(affine_transforms[0], affine_transform_file_path)
    
    atlas_moved = ants.apply_transforms(fixed=t1_masked, moving=atlas_masked, transformlist=affine_transforms,
                                                   interpolator='linear', default_value=0)
    atlas_moved_path = os.path.join(atlas_folder, "ICBM_Template_moved.nii.gz")
    ants.image_write(atlas_moved, atlas_moved_path)
    
    t1_sitk = sitk.ReadImage(t1_path)
    mask_sitk = sitk.ReadImage(mask_path)
    mask_sitk = mask_sitk > 0
    mask_sitk = sitk.Cast(mask_sitk, t1_sitk.GetPixelID())
    t1_masked_sitk = t1_sitk * mask_sitk
    
    # Normalize and match histograms
    atlas_moved_sitk = sitk.ReadImage('../data/Patient1/atlas/ICBM_Template_moved.nii.gz')
    atlas_norm = normalize_image(atlas_moved_sitk)
    t1_norm = normalize_image(t1_masked_sitk)
    atlas_matched_sitk = histogram_matching(atlas_norm, t1_norm)
    
    # Get similarity map
    similarity_map = calculate_ncc(t1_norm, atlas_matched_sitk, window_size=(3,3,3))
    similarity_map_sitk = sitk.GetImageFromArray(similarity_map)
    similarity_map_path = os.path.join(atlas_folder, "t1_atlas_similarity.nii.gz")
    sitk.WriteImage(similarity_map_sitk, similarity_map_path)
    
    # Get atlases
    atlas_files = glob.glob(os.path.join(atlases_folder, '*.nii.gz'))
    tissue_types = ['csf', 'gray', 'white']
    selected_atlases = {}
    
    for atlas in atlas_files:
        for key in tissue_types:
            if key in os.path.basename(atlas).lower():
                selected_atlases[atlas] = key
            
    tissue_atlases = [ants.image_read(atlas) for atlas in selected_atlases]
    resampled_atlases = get_atlases(t1_masked, tissue_atlases, affine_transforms)
    
    for atlas_path, resampled_atlas in zip(selected_atlases.keys(), resampled_atlases):
        tissue_type = selected_atlases[atlas_path]
        atlas_filename = f'atlas_{tissue_type}.nii.gz'
        save_path = os.path.join(atlas_folder, atlas_filename)
        ants.image_write(resampled_atlas, save_path)
        print(f'Saved {tissue_type} atlas to {save_path}')
        

