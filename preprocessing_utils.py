#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import ants
import glob
import psutil
import argparse
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.transform import rescale
from brainextractor import BrainExtractor



def downsample_image_data(image, factor):
    """ Downsamples the given 3D image by the specified factor. """
    zoom_factors = (factor, factor, factor)
    return zoom(image, zoom_factors, order=1)


def downsample_image(image_path, save_path, scale_factor=0.5):
    # Load the image
    image = nib.load(image_path)
    data = image.get_fdata()
    
    # Downsample the image data
    downsampled_data = rescale(data, scale=(scale_factor, scale_factor, scale_factor), 
                               mode='reflect', anti_aliasing=True)
    
    # Create a new NIfTI image, using the original header
    new_header = image.header.copy()
    new_affine = image.affine.copy()
    
    # Adjust the affine matrix to account for the change in voxel size
    new_affine[:3, :3] *= 1/scale_factor
    
    # Create the new NIfTI image with the modified affine matrix and header
    downsampled_image = nib.Nifti1Image(downsampled_data, affine=new_affine, header=new_header)
    downsampled_image.to_filename(save_path)
    return downsampled_image
  
  
def print_memory_usage(description="Current memory"):
    process = psutil.Process()
    print(f"{description} - Memory Usage: {process.memory_info().rss / 1024 ** 3:.2f} GB")
    
    
def extract_brain_mask(input_image, output_mask_path, frac=0.5):
    """
    Extracts a brain mask from an MRI sequence using FSL's BET tool through fslpy.

    Parameters:
    - input_image (.nii): Input MRI image.
    - output_mask_path (str): Path where the brain mask will be saved.

    Returns:
    - None: Outputs are written directly to files specified by output_image_path and output_mask_path.
    """
    #image_data = input_image.get_fdata()
    #downsampled_image_data = downsample_image_data(image_data, 0.5)  # Downsample by a factor of 2
    #downsampled_image = nib.Nifti1Image(downsampled_image_data, affine=input_image.affine)

    # BET command in fsl  
    bet = BrainExtractor(img=input_image)
    bet.run()
    bet.save_mask(output_mask_path)
    print("Brain mask extracted and saved to:", output_mask_path)

def anisotropic_diffusion(image_path, output_path, time_step=0.0625, conductance=2.5, iterations=5):
    """
    Apply anisotropic diffusion filter to a stack of MRI images.

    Parameters:
    - input_image (str): Path to the input MRI image.
    - time_step (float, optional): Time step for the diffusion process. Default is 0.0625.
    - conductance (float, optional): Conductance parameter for edge preservation. Default is 3.0.
    - iterations (int, optional): Number of iterations for the diffusion process. Default is 5.

    Returns:
    sitk.Image: Filtered MRI image.
    """

    image = sitk.ReadImage(image_path)
    print(image_path)
    print("Before processing:")
    print("Spacing:", image.GetSpacing())
    print("Origin:", image.GetOrigin())
    print("Direction:", image.GetDirection())
    
    image = sitk.Cast(image, sitk.sitkFloat32)
    print("Before downsampling:")
    print("Spacing:", image.GetSpacing())
    print("Size:", image.GetSize())
    filtered_image = sitk.CurvatureAnisotropicDiffusion(image, timeStep=time_step, conductanceParameter=conductance, numberOfIterations=iterations)

    print("After processing:")
    print("Spacing:", filtered_image.GetSpacing())
    print("Origin:", filtered_image.GetOrigin())
    print("Direction:", filtered_image.GetDirection())
    
    sitk.WriteImage(filtered_image, output_path)
    print('Smooth image saved to: ', output_path)

def n4_bias_correction(image, mask, n_iter=50, n_levels=1):
    """
    Apply N4 bias field correction to a given image with a mask.

    Parameters:
    - image (ants.ANTsImage): Input image.
    - mask (ants.ANTsImage): Mask image.
    - n_iter (int, optional): Number of iterations. Default is 50.
    - n_levels (int, optional): Number of fitting levels. Default is 1.

    Returns:
    - ants.ANTsImage: Corrected image.
    """
    # Ensure the image and mask are in float format
    corrector = ants.n4_bias_field_correction(image, mask=mask, shrink_factor=2, 
                                              convergence={'iters': [n_iter], 'tol': 1e-6})
    return corrector




