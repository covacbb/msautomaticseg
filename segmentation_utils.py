#!/usr/bin/env python
# coding: utf-8


import os
import glob
import math
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.stats import norm
import matplotlib.pyplot as plt


def normalize_image_intensities(img_data, normalization_type='min-max'):
    """
    Normalize the intensities of a nibabel image.
    
    Parameters:
    - image_path (str): Path to the input image file.
    - output_path (str): Path to save the normalized image. If None, the normalized image is not saved.
    - normalization_type (str): Type of normalization ('min-max' or 'z-score'). Defaults to 'min-max'.
    
    Returns:
    - nib.Nifti1Image: Normalized image.
    """
    if normalization_type == 'min-max':
        # Min-max normalization
        img_min = np.min(img_data)
        img_max = np.max(img_data)
        normalized_data = (img_data - img_min) / (img_max - img_min)
    elif normalization_type == 'z-score':
        # Z-score normalization
        img_mean = np.mean(img_data)
        img_std = np.std(img_data)
        normalized_data = (img_data - img_mean) / img_std
    else:
        raise ValueError("Invalid normalization_type. Choose 'min-max' or 'z-score'.")

    return normalized_data



def plotSoftPosterior(posteriors, nComponents, slices=[115, 115, 75]):
    for n in range(nComponents):
        tmp = np.zeros(brainImage.shape)
        tmp[maskIndices] = posteriors[:, n]
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
        fig.suptitle('Class: ' + str(n+1) )
        ax1.imshow(tmp[slices[0], :, :], cmap='gray')
        ax1.axis('off')
        ax1.set_title('Axial')
        ax2.imshow(tmp[::-1,slices[1],::-1], cmap='gray')
        ax2.axis('off')
        ax2.set_title('Coronal')
        ax3.imshow(tmp[::-1,:,slices[2]], cmap='gray')
        ax3.axis('off')
        ax3.set_title('Sagittal')
        plt.show()

def plotHardPosterior(hardSegmentation, it, slices=[115, 115, 75]):
    tmp = np.zeros(brainImage.shape)
    tmp[maskIndices] = hardSegmentation + 1
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    fig.suptitle('Segmentation at iteration ' + str(it))
    ax1.imshow(tmp[slices[0], :, :], cmap='magma')
    ax1.axis('off')
    ax1.set_title('Axial')
    ax2.imshow(tmp[::-1,slices[1],::-1], cmap='magma')
    ax2.axis('off')
    ax2.set_title('Coronal')
    ax3.imshow(tmp[::-1,:,slices[2]], cmap='magma')
    ax3.axis('off')
    ax3.set_title('Sagittal')
    plt.show()

def plotLikelihood(likelihoodHistory):
    if len(likelihoodHistory) > 1:
        plt.plot(likelihoodHistory)
        plt.xlabel('iteration')
        plt.ylabel('log-likelihood')
        plt.show

def plotOriginalImage(slices):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,4),)
    fig.suptitle('Original Image, slices: ' + str(slices) )
    ax1.imshow(restored_image[slices[0], :, :], cmap='gray')
    ax1.axis('off')
    ax1.set_title('Axial')
    ax2.imshow(restored_image[::-1,slices[1],::-1], cmap='gray')
    ax2.axis('off')
    ax2.set_title('Coronal')
    ax3.imshow(restored_image[::-1,:,slices[2]], cmap='gray')
    ax3.axis('off')
    ax3.set_title('Sagittal')
    plt.show()

def plotHistWithGMM(brainIntensities, GMM_weights, GMM_means, GMM_variances, nComponents, bins):
    plt.figure(figsize=(8, 4))
    val, binsH, _ = plt.hist(brainIntensities.ravel(), bins=bins) 
    area =  sum(np.diff(binsH)*val)
    plt.title("Brain image histogram")
    minIntensity = brainIntensities.min()
    maxIntensity = brainIntensities.max()
    x = np.linspace(minIntensity, maxIntensity, bins)
    gmmNorm = np.zeros(x.shape)
    for n in range(nComponents):
        plt.plot(x, area * GMM_weights[n] * norm.pdf(x, GMM_means[n], np.sqrt(GMM_variances[n])), label='class ' + str(n + 1))
        gmmNorm +=  area * GMM_weights[n] * norm.pdf(x, GMM_means[n], np.sqrt(GMM_variances[n]))
    plt.plot(x, gmmNorm, label='GMM')
    plt.xlabel('Frequency')
    plt.xlabel('Intensity')
    plt.legend()
    plt.show()

def initialize_parameters(brainIntensities, atlases, threshold):
    
    nComponents = len(atlases)
    # Let's create the GMM parameters
    GMM_means = np.zeros([nComponents, 1])
    GMM_variances = np.zeros([nComponents, 1])
    GMM_weights = np.full([nComponents], 1/nComponents) #each component of the mixture model contributes equally at the start

    minIntensity = brainIntensities.min()
    maxIntensity = brainIntensities.max()
    initialWidth = (maxIntensity - minIntensity) / nComponents
    
    for n in range(nComponents):
        atlas = atlases[n]
        masked_intensities = brainIntensities[atlas > threshold]
        GMM_means[n] = np.mean(masked_intensities) if len(masked_intensities) > 0 else  minIntensity + (n + 1) * (initialWidth)
        GMM_variances[n] = np.var(masked_intensities) if len(masked_intensities) > 0 else initialWidth**2

    return GMM_means, GMM_variances, GMM_weights

def expectation_maximization(brainIntensities, atlases, threshold, maxIteration=200, minDifference=1e-4):
    """
    Performs the Expectation-Maximization algorithm for brain image segmentation using 
    Gaussian Mixture Models (GMM) with atlas priors.

    Parameters:
    - brainIntensities (ndarray): 1D array of brain voxel intensities.
    - atlases (list of ndarrays): List of prior probability maps for each tissue component.
    - threshold (float): Threshold for initializing GMM parameters.
    - maxIteration (int, optional): Maximum number of iterations for the EM algorithm. Default is 200.
    - minDifference (float, optional): Minimum change in log-likelihood for convergence. Default is 1e-4.

    Returns:
    - GMM_means (ndarray): Array of mean values for each GMM component.
    - GMM_variances (ndarray): Array of variance values for each GMM component.
    - GMM_weights (ndarray): Array of weight values for each GMM component.
    - posteriors (ndarray): Posterior probabilities of each voxel belonging to each component.
    """
    nComponents = len(atlases)
    GMM_means, GMM_variances, GMM_weights = initialize_parameters(brainIntensities, atlases, threshold)

    # Compute posteriors
    posteriors = np.zeros([len(brainIntensities), nComponents])
    
    showEveryX = 10
    likelihoodHistory = []
    
    # Start EM
    it = 0
    stopCondition = False
    softPlots = True
    
    while(it < maxIteration + 1 and not stopCondition):
        
        # Expectation step with atlas priors
        # Update parameters based on the current classification
        for n in range(nComponents):
            prior = atlases[n].flatten()
            likelihood = norm.pdf(brainIntensities, GMM_means[n], np.sqrt(GMM_variances[n]))
            posteriors[:, n] = GMM_weights[n] * likelihood * prior

        # Compute likelihood
        current_likelihood = np.sum(np.log(np.sum(posteriors, axis=1)+ np.finfo(float).eps))
        likelihoodHistory.append(current_likelihood)

        # Normalization of posteriors (sum of post probabilities across all components for each data point equals 1)
        posteriors_before = posteriors.copy()
        posteriors /= (np.sum(posteriors, axis=1, keepdims=True) + np.finfo(float).eps)

        if it == 0:  # Plot the first iteration posteriors before and after normalization
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.hist(posteriors_before[:, 0], bins=50, label='CSF', alpha=0.5)
            plt.hist(posteriors_before[:, 1], bins=50, label='GM', alpha=0.5)
            plt.hist(posteriors_before[:, 2], bins=50, label='WM', alpha=0.5)
            plt.title("Unnormalized Posteriors")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(posteriors[:, 0], bins=50, label='CSF', alpha=0.5)
            plt.hist(posteriors[:, 1], bins=50, label='GM', alpha=0.5)
            plt.hist(posteriors[:, 2], bins=50, label='WM', alpha=0.5)
            plt.title("Normalized Posteriors")
            plt.legend()
            plt.show()
            
        # Maximization step
        for n in range(nComponents):
            softSum = np.sum(posteriors[:, n])
            GMM_means[n] = (posteriors[:, n].T @ brainIntensities) / (softSum)
            GMM_variances[n] = (posteriors[:, n].T @ (brainIntensities - GMM_means[n])**2) / (softSum)
            GMM_weights[n] = softSum / len(brainIntensities)
            print(f"Component {n}: softSum : {softSum}, Mean = {GMM_means[n]}, Variance = {GMM_variances[n]}, Weight = {GMM_weights[n]}")

            #GMM_means[n] = np.sum(posteriors[:, n] * brainIntensities) / softSum
            #GMM_variances[n] = np.sum(posteriors[:, n] * (brainIntensities - GMM_means[n])**2) / softSum
    
        if it % (showEveryX ) == 0:
            if softPlots:
                plotSoftPosterior(posteriors, nComponents, slices=[115, 115, 75])
            hardSegmentation = np.argmax(posteriors, axis=1)
            plotHardPosterior(hardSegmentation, it, slices=[115, 115, 75])
            plotOriginalImage(slices=[115, 115, 75])
            plotHistWithGMM(brainIntensities, GMM_weights, GMM_means, GMM_variances, nComponents, bins=100)
            plotLikelihood(likelihoodHistory)
    
        if it > 1 and np.abs(likelihoodHistory[-1] - likelihoodHistory[-2]) < minDifference:
            print("Algorithm converges since cost per iteration is smaller than minDifference")
            stopCondition = True
    
        it = it + 1
    return GMM_means, GMM_variances, GMM_weights, posteriors


def full_probability_volumes(brainImage, posteriors, nComponents):
    """
    Generates full probability volumes for each tissue component and creates 
    a hard segmentation map based on posterior probabilities.

    Parameters:
    - brainImage (ndarray): 3D array representing the brain image.
    - posteriors (ndarray): A 2D array where each column corresponds to the posterior 
      probabilities of a specific tissue component.
    - nComponents (int): Number of tissue components (GM, WM, CSF).

    Returns:
    - full_volumes (ndarray): 4D array containing probability volumes for each component.
    - hard_segmentation (ndarray): 3D array containing the hard segmentation map.
    """
    full_volumes = np.zeros((brainImage.shape[0], brainImage.shape[1], brainImage.shape[2], nComponents))
    hard_segmentation = np.zeros(brainImage.shape, dtype=int)
    for i in range(nComponents):
        full_volume = np.zeros(brainImage.shape)
        full_volume[maskIndices] = posteriors[:, i]
        full_volumes[..., i] = full_volume
    return full_volumes




def numpy_solution_from_pr(probability_volumes, mask_array, reference_image):
    """
    Generate a hard segmentation image from probability volumes using a mask, 
    ensuring that the output image retains the metadata of a reference image.

    Parameters:
    - probability_volumes (numpy.ndarray): A 4D array where each slice along the last dimension is a probability map for a class.
    - mask_array (numpy.ndarray): A 3D binary array representing the brain mask.
    - reference_image (SimpleITK.Image): The reference image to copy metadata from.

    Returns:
    - SimpleITK.Image: The segmentation image with the metadata from the reference image.
    """
    
    # Find the index of the maximum probability for each voxel
    labels = np.argmax(probability_volumes, axis=-1) + 1  # +1 to convert from 0-based to 1-based labels

    # Apply the brain mask
    labels *= mask_array.astype(np.int32)

    # Convert the result back to a SimpleITK image
    label_image = sitk.GetImageFromArray(np.transpose(labels, (2, 1, 0)))

    label_image.CopyInformation(reference_image)

    return label_image



def lesion_new(tissue, flair, alpha, factor, min_size, ce_radius, dil_radius, wm_threshold, gm_wm_threshold):  
    """
    Detects and refines brain lesions in FLAIR images based on tissue labels, thresholds, 
    and surrounding tissue characteristics.

    Parameters:
    - tissue (SimpleITK.Image): A labeled brain tissue image where different regions (e.g., GM, WM) are identified.
    - flair (SimpleITK.Image): The FLAIR image to process for lesion detection.
    - alpha (float): Scaling factor for the threshold calculation.
    - min_size (int): Minimum size of a lesion to be considered valid.
    - ce_radius (int): Radius for contrast enhancement processing.
    - dil_radius (int): Radius for lesion dilation during refinement.
    - wm_threshold (float): Threshold percentage for white matter surrounding lesions.
    - gm_wm_threshold (float): Threshold percentage for gray and white matter inside lesions.
    - factor (float): Scaling factor for exponential transformation.

    Returns:
    - refined_lesions_image (SimpleITK.Image): A labeled image of the refined lesions.
    - results (dict): A dictionary containing the tissue percentages for each lesion.
    """
    
    # Enhance contrast of the FLAIR image (assuming method 'ContrastEnhancement' exists or a similar one is used)
    flair_e = exponential_transformation(flair, factor)
    flair_e2 = contrast_enhancement(flair, ce_radius)
    
    # Threshold estimation
    print("Threshold estimation")
    flair_array = sitk.GetArrayFromImage(flair_e)
    tissue_array = sitk.GetArrayFromImage(tissue)
    
    # Mask the flair image with GM tissue
    gm_mask = (tissue_array == 2)
    gm_flair_values = flair_array[gm_mask]
    mu = np.mean(gm_flair_values)
    sigma = np.std(gm_flair_values)
    fwhm = calculate_fwhm(gm_flair_values)
    
    # Calculate threshold
    t = calculate_threshold(mu, alpha, sigma, fwhm, factor)
    
    # Thresholding FLAIR image
    print("Thresholding and refinement")
    flair_thresholded = sitk.BinaryThreshold(flair_e2, lowerThreshold=t, upperThreshold=float("inf"))
    
    # Initialize the fill holes filter
    fill_holes_filter = sitk.BinaryFillholeImageFilter()
    fill_holes_filter.SetFullyConnected(False)
    
    filled_lesions_img = sitk.Image(flair_thresholded.GetSize(), flair_thresholded.GetPixelIDValue())
    filled_lesions_img.CopyInformation(flair_thresholded)
    filled_lesions_img = fill_holes_filter.Execute(flair_thresholded)
    
    # Connected Component Analysis to identify individual lesions
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    components = connected_component_filter.Execute(sitk.Cast(filled_lesions_img, sitk.sitkUInt8))
    print(f"Number of initial components: {connected_component_filter.GetObjectCount()}")
    
    # Relabel components based on size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(min_size)
    relabelled_components = relabel_filter.Execute(components)
    print(f"Number of relabelled components (size-filtered): {relabel_filter.GetNumberOfObjects()}")
    
    # Components array
    lesions_array = sitk.GetArrayFromImage(relabelled_components)
    unique_labels = np.unique(lesions_array[lesions_array > 0]) 
    
    # Create an array to store the refined lesions and a dictionary to store their characteristics
    refined_lesions = np.zeros_like(lesions_array)
    results = {}
    
    # Analyze each component in the original image
    for label in unique_labels:
            
        # Mask for current lesion
        lesion_mask = lesions_array == label
    
        # Convert the NumPy array mask back to a SimpleITK image for processing
        lesion_image = sitk.GetImageFromArray((lesion_mask).astype(np.uint8))

        # Dilation of lesions to find surrounding tissues
        dilation_filter = sitk.BinaryDilateImageFilter()
        dilation_filter.SetKernelRadius(dil_radius)
        dilation_filter.SetKernelType(sitk.sitkBall)
        dilated_component = dilation_filter.Execute(lesion_image)
        
        # Convert to arrays for manipulation
        dilated_array = sitk.GetArrayFromImage(dilated_component)
        
        # Calculate surrounding region by subtracting original components from dilated components
        surrounding_mask = np.logical_and(dilated_array, np.logical_not(lesion_mask))
    
        # Calculate percentages of WM in surrounding area
        wm_surrounding = np.sum(tissue_array[surrounding_mask] == 3) # 3: wm label index in label_image
        total_surrounding = np.sum(surrounding_mask)
        wm_surrounding_percentage = (wm_surrounding / total_surrounding * 100) if total_surrounding > 0 else 0
        print(f'White matter surrounding percentage: {wm_surrounding_percentage}')
    
        # Calculate percentages of WM and GM inside the lesion
        wm_inside = np.sum(tissue_array[lesion_mask] == 3) # 3: wm label index
        gm_inside = np.sum(tissue_array[lesion_mask] == 2) # 2: gm label index
        total_inside = np.sum(lesion_mask)
        gm_wm_inside_percentage = ((wm_inside + gm_inside) / total_inside * 100) if total_inside > 0 else 0
        print(f'White and gray matter inside percentage: {gm_wm_inside_percentage}')
    
        # Evaluate against thresholds
        results[label] = {
            'wm_surrounding_percentage': wm_surrounding_percentage,
            'gm_wm_inside_percentage': gm_wm_inside_percentage,
            'wm_surrounding_valid': wm_surrounding_percentage > wm_threshold,
            'gm_wm_inside_valid': gm_wm_inside_percentage > gm_wm_threshold
        }

        num_lesions = 0
        #Check if the lesion meets the specified conditions
        if (wm_surrounding_percentage > wm_threshold and 
            gm_wm_inside_percentage > gm_wm_threshold and 
            np.sum(lesion_mask) >= min_size):
            refined_lesions[lesion_mask] = label
            num_lesions += 1
    
    # Convert final components back to an image
    # refined_lesions_image = sitk.GetImageFromArray(refined_lesions)
    refined_lesions_image = relabelled_components
    refined_lesions_image.CopyInformation(flair)
   
    return refined_lesions_image, results


def contrast_enhancement(image, radius):
    """
    Enhances the contrast of an image using morphological operations.

    Parameters:
    - image (SimpleITK.Image): The input image.
    - radius (int): The radius of the structuring element used in dilation and erosion.

    Returns:
    - SimpleITK.Image: The contrast-enhanced image.
    """
    # Define the structuring element as a ball
    structuring_element = sitk.sitkBall
    radius = [radius] * image.GetDimension()  # Ensure the radius is defined for all dimensions

    # Apply grayscale dilation
    dilated_image = sitk.GrayscaleDilate(image, radius, structuring_element)

    # Apply grayscale erosion
    eroded_image = sitk.GrayscaleErode(image, radius, structuring_element)

    # Initialize the result image
    enhanced_image = sitk.Image(image.GetSize(), image.GetPixelIDValue())
    enhanced_image.CopyInformation(image)

    # Apply the logic described in your original algorithm
    # We need to iterate over all pixels, so convert images to arrays for easier processing
    dilated_array = sitk.GetArrayFromImage(dilated_image)
    eroded_array = sitk.GetArrayFromImage(eroded_image)
    image_array = sitk.GetArrayFromImage(image)                
    enhanced_array = sitk.GetArrayFromImage(enhanced_image)   

    # Enhance contrast based on the conditions specified
    for idx in np.ndindex(image_array.shape):
        original = image_array[idx]
        dilated = dilated_array[idx]
        eroded = eroded_array[idx]

        if dilated - original <= original - eroded:
            enhanced_array[idx] = dilated
        elif original - eroded <= dilated - original:
            enhanced_array[idx] = eroded

    # Convert the numpy array back to SimpleITK Image
    enhanced_image = sitk.GetImageFromArray(enhanced_array)
    enhanced_image.CopyInformation(image)

    return enhanced_image



def calculate_fwhm(array):
    """
    Calculates the Full Width at Half Maximum (FWHM) from the histogram of the input array.

    Parameters:
    - array (ndarray): A 1D or 3D array representing the data for which FWHM is calculated.
    Returns:
    - fwhm (float or None): The calculated FWHM value. Returns None if the calculation cannot be performed.
    """
    data = array.flatten()

    # Generate histogram
    y, x = np.histogram(data, bins=100, density=True)
    x = (x[:-1] + x[1:]) / 2  # Convert bin edges to centers

    max_y = max(y)
    half_max = max_y / 2

    # Find indices where the histogram first and last cross half maximum
    indices = np.where(y > half_max)[0]
    if len(indices) > 0:
        fwhm = x[indices[-1]] - x[indices[0]]
    else:
        fwhm = None

    return fwhm


# Calculate threshold
def calculate_threshold(mu, alpha, sigma, fwhm, factor):
    """
    Following the threshold equation based on (mu, alpha, sigma) of a given image,
    calculates a new threshold value after applying an exponential transformation 
    to the input image. 
    The full-width half maximum (FWHM) is used to calculate sigma. If the FWHM value
    is not provided, the sigma from the distribution is used.

    Parameters:
    - mu (float): The mean of the original distribution.
    - alpha (float): A scaling factor applied to the standard deviation.
    - sigma (float): The original standard deviation of the distribution.
    - fwhm (float): The full-width half maximum, used to estimate sigma if provided.

    Returns:
    - t_new (float): The calculated threshold after the transformation.

    """
    if fwhm is not None:
        sigma_from_fwhm = fwhm / 2.35482
        print("FWHM:", fwhm, "Estimated sigma from FWHM:", sigma_from_fwhm)
        sigma_final = sigma_from_fwhm
    else:
        sigma_final = sigma
        print("sigma:", sigma)
        
    # Calculate new threshold after exponential transformation
    print("Calculating threshold after exponential transformation...") 
    # New mean
    mu_new = math.exp(mu * factor + 0.5 * (sigma_final * factor) ** 2)
    # New standard deviation
    sigma_new = math.sqrt(
        (math.exp((sigma_final * factor) ** 2) - 1) * math.exp(2 * mu * factor + (sigma_final * factor) ** 2))
    # New threshold
    t_new = mu_new + alpha * sigma_new

    print(f"\tNew Threshold: {t_new}")
    print(f"\t(mu_new: {mu_new}, sigma_new: {sigma_new}, alpha: {alpha})")
      
    return t_new


def exponential_transformation(image, factor):
    """
    Apply the exponential transformation to the image histogram.

    Parameters:
    - image (SimpleITK.Image): The input image.
    - factor (float): Factor to scale the exponential function, controlling the rate of growth.

    Returns:
    - impleITK.Image: The image after applying the exponential transformation.
    """
    # Calculate the minimum and maximum intensities in the image
    statistics_filter = sitk.StatisticsImageFilter()
    statistics_filter.Execute(image)
    min_intensity = statistics_filter.GetMinimum()
    max_intensity = statistics_filter.GetMaximum()

    # Normalize the image intensities to [0, 1]
    normalized_image = sitk.RescaleIntensity(image, outputMinimum=0.0, outputMaximum=1.0)

    # Apply the exponential function scaled by the factor
    exp_image = sitk.Exp(normalized_image * factor)

    # Rescale intensities back to the original range
    exp_image = sitk.RescaleIntensity(exp_image, outputMinimum=min_intensity, outputMaximum=max_intensity)

    return exp_image


def plot_label_image(label_image, title='Labeled Image'):
    """
    Plot slices from the 3D labeled image across different orientations.
    Parameters:
    - label_image (SimpleITK.Image): The labeled segmentation image.
    - title (str): Title for the plots.
    """
    # Convert SimpleITK image to a NumPy array for easy slicing
    label_array = sitk.GetArrayFromImage(label_image)

    # Selecting one slice from each dimension to display
    axial_slice = label_array[label_array.shape[0] // 2, :, :]
    sagittal_slice = label_array[:, :, label_array.shape[2] // 2]
    coronal_slice = label_array[:, label_array.shape[1] // 2, :]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    # Axial
    im1 = axes[0].imshow(axial_slice, cmap='jet')
    axes[0].set_title('Axial')
    axes[0].axis('off')

    # Coronal
    im2 = axes[1].imshow(coronal_slice, cmap='jet')
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Sagittal
    im3 = axes[2].imshow(sagittal_slice, cmap='jet')
    axes[2].set_title('Sagittal')
    axes[2].axis('off')

    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_ticks(range(len(labels)))
    cbar.set_ticklabels([labels[i] for i in range(len(labels))])


    plt.show()


def plot_mri_histogram(image, lib):

    if lib=='ants':
        data = image.numpy()
    elif lib=='sitk':
        data = sitk.GetArrayFromImage(image)
    if lib == 'nib':
        data = image
    else:
        print('lib incorrecta')
    # Flatten the data to a 1D array
    flattened_data = data.flatten()
    non_zero_data = flattened_data[flattened_data != 0]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_data, bins=100, color='blue', alpha=0.7 )
    plt.title("Histogram of MRI Intensities")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return non_zero_data



def plot_labels(label_image):
    relabelled_array = sitk.GetArrayFromImage(label_image)
    
    # Select a specific slice (e.g., the middle slice)
    slice_index = relabelled_array.shape[0] // 2
    slice_data = relabelled_array[70, :, :]
    
    # Plot the selected slice of the relabeled components
    plt.figure(figsize=(7, 7))
    plt.imshow(slice_data, cmap='nipy_spectral')  # 'nipy_spectral' is a good colormap for distinct colors
    plt.colorbar()
    plt.title('Relabeled Components - Slice')
    plt.axis('off')
    plt.show()

