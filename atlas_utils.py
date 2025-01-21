#!/usr/bin/env python
# coding: utf-8


import os
import glob
import ants
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from brainextractor import BrainExtractor


def apply_brain_mask(image, brain_mask):
    """
    Apply a brain mask to an image sequence using ANTs.

    Parameters:
    - image (ants.ANTsImage): The input image sequence.
    - brain_mask (ants.ANTsImage): The brain mask to be applied.

    Returns:
    - ants.ANTsImage: The masked image sequence.
    """
    # Ensure the brain mask is a binary mask
    brain_mask = brain_mask > 0

    # Apply the mask
    masked_image = image * brain_mask

    return masked_image

def extract_brain_mask(input_image, output_mask_path):
    """
    Extracts a brain mask from an MRI sequence using FSL's BET tool through fslpy.

    Parameters:
    - input_image (.nii): Input MRI image.
    - output_mask_path (str): Path where the brain mask will be saved.

    Returns:
    - None: Outputs are written directly to files specified by output_image_path and output_mask_path.
    """
    # BET command in fsl  
    bet = BrainExtractor(img=input_image)
    bet.run()
    bet.save_mask(output_mask_path)
    print("Brain mask extracted and saved to:", output_mask_path)

def rigid_registration(fixed_image, moving_image, levels=2, steps=50):
    """
    Performs rigid registration between the fixed and moving images using ANTsPy.

    Parameters:
    - fixed_image (ants.ANTsImage): The fixed (reference) image.
    - moving_image (ants.ANTsImage): The moving image to be registered to the fixed image.
    - levels (int, optional): The number of resolution levels for multi-resolution registration. Default is 2.
    - steps (int, optional): The number of iterations for registration at each resolution level. Default is 50.

    Returns:
      tuple: A tuple containing:
        - list: Forward transform file paths generated during the registration.
        - ants.ANTsImage: The warped moving image aligned to the fixed image.
    """
    # Rigid registration
    transform_type = 'Rigid'
    metric = 'Mattes'
    metric_params = {'metric_weight': 1.0, 'number_of_histogram_bins': 50, 'use_sampled_point_set': False}
    grad_step = 0.2  # Corresponds to maximum step length in ITK Versor optimizer
    shrink_factors = [2**i for i in range(levels, 0, -1)]
    smoothing_sigmas = [2**i for i in range(levels - 1, -1, -1)]

    # Perform registration using ANTsPy
    registration = ants.registration(fixed=fixed_image, moving=moving_image,
                                     type_of_transform=transform_type, 
                                     grad_step=grad_step,
                                     reg_iterations=[steps]*levels,
                                     shrink_factors=shrink_factors,
                                     smoothing_sigmas=smoothing_sigmas,
                                     metric=metric,
                                     metric_params=metric_params,
                                     verbose=True)

    # Return the transform parameters and the transformed image
    return registration['fwdtransforms'], registration['warpedmovout']

def affine_registration(fixed_image, moving_image, initial_transform=None, levels=2, steps=50):
    """
    Performs affine registration between the fixed and moving images, optionally initializing with a rigid transform.

    Parameters:
    - fixed_image (ants.ANTsImage): The fixed (reference) image.
    - moving_image (ants.ANTsImage): The moving image to be registered to the fixed image.
    - initial_transform (str, optional): Path to an initial transform file for pre-alignment. Default is None.
    - levels (int, optional): The number of resolution levels for multi-resolution registration. Default is 2.
    - steps (int, optional): The number of iterations for registration at each resolution level. Default is 50.

    Returns:
      tuple: A tuple containing:
        - list: Forward transform file paths generated during the registration.
        - ants.ANTsImage: The warped moving image aligned to the fixed image.
    """
    if initial_transform is None:
        print("\tNo initial transformation provided, performing rigid registration first.")
        initial_transform = rigid_registration(fixed_image, moving_image)['fwdtransforms'][0]

    # Set up optimizer parameters
    transform_type = 'Affine'
    metric = 'Mattes'
    metric_params = {'metric_weight': 1.0, 'number_of_histogram_bins': 50, 'use_sampled_point_set': False}
    grad_step = 0.2  # Corresponds to maximum step length in ITK
    shrink_factors = [2**i for i in range(levels, 0, -1)]
    smoothing_sigmas = [2**i for i in range(levels - 1, -1, -1)]

    # Perform affine registration using ANTsPy
    registration = ants.registration(fixed=fixed_image, moving=moving_image,
                                     initial_transform=initial_transform,
                                     type_of_transform=transform_type,
                                     metric=metric,
                                     metric_params=metric_params,
                                     grad_step=grad_step,
                                     reg_iterations=[steps]*levels,
                                     shrink_factors=shrink_factors,
                                     smoothing_sigmas=smoothing_sigmas,
                                     verbose=True)

    # Return the transformation parameters and the transformed image
    return registration['fwdtransforms'], registration['warpedmovout']

def normalize_image(image):
    """
    Normalizes a SimpleITK image to the range [0, 1].

    Parameters:
    - image (sitk.Image): The input image to be normalized.

    Returns:
    - sitk.Image: The normalized image with the same spatial metadata as the input image.
    """
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_array = (image_array - min_val) / (max_val - min_val)
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def histogram_matching(source_image, reference_image, 
                       number_of_histogram_levels=1024, number_of_match_points=35, 
                       threshold_at_mean_intensity=True):
    """
    Perform histogram matching of the source image to the reference image.

    Parameters:
    - source_image_path (str): Path to the source image.
    - reference_image_path (str): Path to the reference image.
    - output_image_path (str): Path to save the output matched image.
    - number_of_histogram_levels (int): Number of bins used when creating histograms of the images.
    - number_of_match_points (int): Number of quantile values to be matched.
    - threshold_at_mean_intensity (bool): Whether to threshold at mean intensity.
    """
    # Read the source and reference images
    #source_image = sitk.ReadImage(source_image_path, sitk.sitkFloat32)
    #reference_image = sitk.ReadImage(reference_image_path, sitk.sitkFloat32)
    
    # Initialize the histogram matching filter
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(number_of_histogram_levels)
    matcher.SetNumberOfMatchPoints(number_of_match_points)
    matcher.SetThresholdAtMeanIntensity(threshold_at_mean_intensity)
    
    # Perform histogram matching
    matched_image = matcher.Execute(source_image, reference_image)
    
    # Save the matched image
    #sitk.WriteImage(matched_image, output_image_path)
    
    return matched_image

def calculate_ncc(sequence1, sequence2, window_size):
    """
    Calculate the Normalized Cross-Correlation (NCC) values between two normalized 3D MRI sequences using convolution.

    Parameters:
    - sequence1 (numpy.ndarray): The first input normalized 3D MRI sequence.
    - sequence2 (numpy.ndarray): The second input normalized 3D MRI sequence.
    - window_size (int): The size of the sliding window for NCC calculation.

    Returns:
    - numpy.ndarray: The NCC values as a similarity map sequence.
    """

    # Ensure the sequences are numpy arrays
    sequence1 = sitk.GetArrayFromImage(sequence1).astype(np.float32)
    sequence2 = sitk.GetArrayFromImage(sequence2).astype(np.float32)

    print(f"Input sequence1 shape: {sequence1.shape}")
    print(f"Input sequence2 shape: {sequence2.shape}")

    # Compute means and standard deviations of the sequences within the window
    mean1 = uniform_filter(sequence1, size=window_size, mode='reflect')
    mean2 = uniform_filter(sequence2, size=window_size, mode='reflect')

    squared_mean1 = uniform_filter(sequence1**2, size=window_size, mode='reflect')
    squared_mean2 = uniform_filter(sequence2**2, size=window_size, mode='reflect')

    variance1 = np.maximum(squared_mean1 - mean1**2, 0)
    variance2 = np.maximum(squared_mean2 - mean2**2, 0)

    std1 = np.sqrt(variance1)
    std2 = np.sqrt(variance2)

    print(f"mean1: {mean1}")
    print(f"mean2: {mean2}")
    print(f"std1: {std1}")
    print(f"std2: {std2}")
    
    # Compute cross-correlation
    cross_corr = uniform_filter(sequence1 * sequence2, size=window_size, mode='reflect') - mean1 * mean2
    
    print(f"cross_corr shape: {cross_corr.shape}")

    # Compute NCC
    window_volume = np.prod(window_size)
    denominator = std1 * std2
    epsilon = 1e-8 
    ncc = cross_corr / (denominator + epsilon)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = std1 * std2 * window_volume # Correct denominator
        epsilon = 1e-8  # Small constant to avoid division by zero
        ncc = cross_corr / (denominator + epsilon)
        #denominator[denominator==0] = 1
    """
    ncc[np.isnan(ncc)] = 0  # Replace NaN values with 0
    ncc[np.isinf(ncc)] = 0  # Replace inf values with 0

    print(f"ncc shape: {ncc.shape}")
    print(f"Max intensity: {np.max(ncc)}")
    print(f"Min intensity: {np.min(ncc)}")

    return ncc

def get_atlases(fixed_img, mov_atlases, transform):
    """
    Resamples and refines probabilistic atlases to match the space of a fixed image.

    Parameters:
    - fixed_img (ants.ANTsImage): The fixed (reference) image.
    - mov_atlases (list of ants.ANTsImage): List of moving probabilistic atlas images to be resampled.
    - transform (str or list): Transform or list of transforms used for initial alignment.

    Returns:
    - list of ants.ANTsImage: A list of resampled and refined atlas images in the fixed image's space.
    """

    atlases = []
    print("GetAtlases")

    # Check the type of transform and resample all probabilistic atlases
    for mov_atlas in mov_atlases:
        try:
            # Resample the moving atlas to the space of the fixed image using the provided transform
            resampled_atlas = ants.apply_transforms(fixed=fixed_img, moving=mov_atlas, 
                                                    transformlist=transform, interpolator='linear', default_value=0)
            
            refining_transform = ants.registration(fixed=fixed_img, moving=resampled_atlas, type_of_transform='Affine')
            refined_atlas = ants.apply_transforms(fixed=fixed_img, moving=resampled_atlas, 
                                        transformlist=refining_transform['fwdtransforms'], interpolator='linear', default_value=0)
            atlases.append(refined_atlas)
            print(f"\tAtlas resampled successfully.")
        except Exception as err:
            print("\tExceptionObject caught!")
            print(err)
            continue  # or return here depending on error handling

    return atlases

